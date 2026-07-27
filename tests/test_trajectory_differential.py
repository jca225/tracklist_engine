"""Trajectory migration differential — core.timebase vs the legacy path_decode impl.

The legacy `_pieces`/`_ref_at` bodies (pre-Trajectory, commit adb175b^) are
pinned here verbatim; the shims in path_decode must produce IDENTICAL values
over the real BB11+BB12 ground-truth rows (every span, dense samples,
including out-of-range extrapolation points).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alignment.path_decode import _gt_pieces, _pieces, _ref_at

_REPO = Path(__file__).resolve().parents[1]
_GT_YAMLS = sorted((_REPO / "labeling" / "fixtures").glob("*_ground_truth.yaml"))


# --- legacy implementation, pinned verbatim ---------------------------------


def _legacy_pieces(seg_list, span_start, span_end, default_slope):
    out = []
    for i, (ms, rs, re) in enumerate(seg_list):
        me = seg_list[i + 1][0] if i + 1 < len(seg_list) else span_end
        dur = max(me - ms, 1e-6)
        slope = (re - rs) / dur if (re - rs) else default_slope
        out.append((ms, me, rs, slope))
    return out


def _legacy_gt_pieces(row):
    segs = row.get("ref_segments")
    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    slope = float(row.get("tempo_ratio") or 1.0)
    if segs:
        seq = [
            (float(s["mix_start_s"]), float(s["ref_start_s"]), float(s["ref_end_s"]))
            for s in segs
        ]
        return _legacy_pieces(seq, s0, s1, slope)
    return [(s0, s1, float(row["ref_start_s"]), slope)]


def _legacy_ref_at(pieces, t):
    for ms, me, rs, slope in pieces:
        if ms <= t <= me:
            return rs + (t - ms) * slope
    if t < pieces[0][0]:
        ms, _me, rs, slope = pieces[0]
        return rs + (t - ms) * slope
    ms, me, rs, slope = pieces[-1]
    return rs + (t - ms) * slope


# --- differential over real GT ----------------------------------------------


def _gt_rows():
    rows = []
    for p in _GT_YAMLS:
        doc = yaml.safe_load(p.read_text())
        rows.extend(
            r
            for r in doc["tracks"]
            if str(r.get("slot_label")) != "mix" and r.get("ref_start_s") is not None
        )
    return rows


@pytest.mark.skipif(not _GT_YAMLS, reason="no GT fixtures in this checkout")
def test_differential_all_gt_spans():
    rows = _gt_rows()
    assert len(rows) > 200, "expected BB11+BB12 GT corpora"
    checked = 0
    for row in rows:
        new = _gt_pieces(row)
        old = _legacy_gt_pieces(row)
        assert len(new) == len(old)
        for a, b in zip(new, old):
            assert a == pytest.approx(b)
        s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
        span = max(s1 - s0, 1.0)
        # dense in-range samples + out-of-range extrapolation on both sides
        ts = [s0 + span * f for f in (-0.2, 0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.2)]
        for t in ts:
            assert _ref_at(new, t) == pytest.approx(_legacy_ref_at(old, t), abs=1e-9)
            checked += 1
    assert checked > 1600


@pytest.mark.skipif(not _GT_YAMLS, reason="no GT fixtures in this checkout")
def test_differential_pieces_zero_ref_span_quirk():
    # a dst-degenerate segment (stab) must take default_slope, exactly as legacy
    segs = [(10.0, 50.0, 50.0), (14.0, 60.0, 68.0)]
    assert _pieces(segs, 10.0, 20.0, 1.07) == _legacy_pieces(segs, 10.0, 20.0, 1.07)
