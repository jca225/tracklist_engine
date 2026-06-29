"""Falsify-first: do GT segment boundaries land on metrical PHRASE lines?

Lever B (redirected-WS2, beat entrainment) only makes sense if DJs cut/loop on
metrical phrase boundaries. Test it on GT before building anything: for every GT
ref_segment boundary (its ``mix_start_s``), measure the distance to the nearest
mix downbeat and to the nearest 4-/8-/16-bar phrase line (from the MERT bundle's
measure grid, `mert_store.load_bb12_mert` → mix_series.start_s), and compare to a
random-time null. If boundaries cluster on phrase lines well above chance, the
phrase-snap decoder is justified; if not, kill Lever B and pivot to Lever A.

Read-only, no audio: GT yaml + cached MERT measure grid only.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.neuro.phrase_premise_check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.result import Err, Ok
from labeling.ground_truth import schema
from workspaces.alignment_prototype.mert_store import load_bb12_mert

GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"
SET_ID = "1fsnxchk"


def _phrase_lines(downbeats: np.ndarray, bars: int) -> np.ndarray:
    """Every ``bars``-th downbeat = a phrase line."""
    return downbeats[:: max(1, bars)]


def _nearest_dist(t: float, lines: np.ndarray) -> float:
    return float(np.min(np.abs(lines - t)))


def main() -> int:
    r = schema.load(GT_YAML)
    if not r.is_ok():
        print(f"GT load failed: {r.error}", file=sys.stderr)
        return 2
    gt = r.value

    match load_bb12_mert(SET_ID):
        case Err(msg):
            print(f"need MERT bundle for measure grid: {msg}", file=sys.stderr)
            return 2
        case Ok((_sid, mix_series, _ref_series)):
            pass

    downbeats = np.asarray(mix_series.start_s, dtype=float)
    downbeats.sort()
    if downbeats.size < 8:
        print("too few measures", file=sys.stderr)
        return 2
    bar_dur = float(np.median(np.diff(downbeats)))
    print(
        f"mix measures: {downbeats.size}, median bar={bar_dur:.2f}s "
        f"(span {downbeats[0]:.0f}-{downbeats[-1]:.0f}s)"
    )

    # collect GT boundaries: each segment's mix_start_s (skip the very first
    # boundary of a span — that's placement, not a within-span cut)
    bounds: list[float] = []
    for t in gt.tracks:
        segs = sorted(t.ref_segments, key=lambda s: s.mix_start_s)
        for s in segs[1:]:  # internal cut/loop points only
            if downbeats[0] <= s.mix_start_s <= downbeats[-1]:
                bounds.append(s.mix_start_s)
    print(f"internal GT segment boundaries in grid range: {len(bounds)}")
    if len(bounds) < 20:
        print("too few boundaries to test", file=sys.stderr)
        return 2

    rng = np.random.default_rng(0)
    null = rng.uniform(downbeats[0], downbeats[-1], size=20000)

    grids = {
        "downbeat (1 bar)": downbeats,
        "4-bar phrase": _phrase_lines(downbeats, 4),
        "8-bar phrase": _phrase_lines(downbeats, 8),
        "16-bar phrase": _phrase_lines(downbeats, 16),
    }
    print(
        f"\n{'grid':18}{'n_lines':>8}{'GT med dist':>13}{'null med':>10}"
        f"{'GT<0.5bar%':>12}{'null<0.5bar%':>13}"
    )
    tol = 0.5 * bar_dur  # within half a bar of a phrase line
    for name, lines in grids.items():
        gd = np.array([_nearest_dist(b, lines) for b in bounds])
        nd = np.array([_nearest_dist(b, lines) for b in null])
        gt_hit = 100.0 * np.mean(gd <= tol)
        null_hit = 100.0 * np.mean(nd <= tol)
        print(
            f"{name:18}{lines.size:8d}{np.median(gd):12.2f}s{np.median(nd):9.2f}s"
            f"{gt_hit:11.0f}%{null_hit:12.0f}%"
        )

    print(
        "\nread: GT med dist << null med, and GT-hit% >> null-hit% ⇒ boundaries "
        "are phrase-locked ⇒ Lever B justified. Parity with null ⇒ kill Lever B."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
