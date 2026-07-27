"""Phase B feasibility gate — can lyric CONTEXT discriminate the right repeat
instance?

The decoder lands on the wrong instance of a repeated section (HuBERT emission
is flat across identical repeats). Phase B would add a lyric-consistency term to
the decode emission. Before building that, measure the ceiling: on the actual
strict-miss span-seconds, does a lyric-window match score rank the GT ref-time
ABOVE the (wrong) predicted ref-time? This is discriminative (argmax over two
offsets, full ~12s window context incl. the distinguishing surrounding words) —
NOT the equivalence test that made lyric FIBERS over-merge.

  gt-preferred fraction high  -> build the emission fusion
  ~chance / pred-preferred    -> lyrics don't disambiguate these; close early

    venvs/audio/bin/python -m alignment.phaseb_probe \
        --pred looptrace/out/pred_1fsnxchk_hubert.json \
        --gt labeling/fixtures/bb12_ground_truth.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from alignment.evals.fiber_ab import _load_gt_rows, _match_gt
from alignment.path_decode import _gt_pieces, _pieces, _ref_at

HALF = 6.0  # +/- window (~ the 12 s matched-filter window)


def _mix_vocals_for(ref_audio: str) -> str | None:
    """ref .../aligning/<SET>/stems/<slot>/vocals.flac -> <SET>/mix_vocals.flac."""
    p = Path(ref_audio)
    for anc in p.parents:
        if anc.name == "stems":
            return str(anc.parent / "mix_vocals.flac")
    return None


def _flat_bigrams(audio_path: str) -> list[tuple[str, float]]:
    from alignment.lyrics_align import (
        _bigram_times,
        _norm,
        transcribe_words,
    )

    seq = _norm(transcribe_words(audio_path))
    return [(bg, t) for bg, ts in _bigram_times(seq).items() for t in ts]


def _bg_at(flat: list[tuple[str, float]], t: float, half: float = HALF) -> set[str]:
    return {bg for (bg, wt) in flat if abs(wt - t) <= half}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--stems", default="acappella")
    args = p.parse_args()
    stems = set(args.stems.split(","))

    import json

    spans = json.loads(args.pred.read_text())["spans"]
    gt_by_tid: dict[str, list[dict]] = defaultdict(list)
    for r in _load_gt_rows(args.gt):
        if r.get("track_id"):
            gt_by_tid[str(r["track_id"])].append(r)

    mix_cache: dict[str, list] = {}
    ref_cache: dict[str, list] = {}
    tal: Counter = Counter()

    for s in spans:
        if (s.get("claimed_stem") or "regular") not in stems:
            continue
        g = _match_gt(s, gt_by_tid)
        if g is None:
            continue
        ref = s["ref_audio"]
        mixp = _mix_vocals_for(ref)
        if mixp is None or not Path(mixp).is_file():
            continue
        if mixp not in mix_cache:
            mix_cache[mixp] = _flat_bigrams(mixp)
        if ref not in ref_cache:
            ref_cache[ref] = _flat_bigrams(ref)
        mix_flat, ref_flat = mix_cache[mixp], ref_cache[ref]

        s0, s1 = float(g["set_start_s"]), float(g["set_end_s"])
        slope = float(g.get("tempo_ratio") or 1.0)
        pred = _pieces([(s0 + ms, rs, re) for (ms, rs, re) in s["segs"]], s0, s1, slope)
        gtp = _gt_pieces(g)
        for t in np.arange(s0, s1, 1.0):
            pr, gr = _ref_at(pred, t), _ref_at(gtp, t)
            if abs(pr - gr) < 2.0:
                continue  # strict hit — not a failure
            mbg = _bg_at(mix_flat, float(t))
            if len(mbg) < 3:
                tal["no_mix_lyric"] += 1
                continue
            sc_gt = len(mbg & _bg_at(ref_flat, float(gr)))
            sc_pr = len(mbg & _bg_at(ref_flat, float(pr)))
            tal["evaluable"] += 1
            if sc_gt > sc_pr and sc_gt >= 3:
                tal["gt_preferred"] += 1
            elif sc_pr > sc_gt and sc_pr >= 3:
                tal["pred_preferred"] += 1
            else:
                tal["tie_or_weak"] += 1

    ev = tal["evaluable"]
    print(
        f"=== phaseb feasibility pred={args.pred.name} stems={stems} half={HALF}s ==="
    )
    print(
        f"  strict-miss seconds with mix lyrics (evaluable): {ev}  "
        f"(no-mix-lyric skipped: {tal['no_mix_lyric']})"
    )
    if ev:
        print(
            f"  GT-offset preferred by lyrics : {tal['gt_preferred']:3d}  "
            f"({tal['gt_preferred'] / ev:.1%})   <- would-fix ceiling"
        )
        print(
            f"  WRONG (pred) preferred        : {tal['pred_preferred']:3d}  "
            f"({tal['pred_preferred'] / ev:.1%})"
        )
        print(
            f"  tie / weak (<3 shared)        : {tal['tie_or_weak']:3d}  "
            f"({tal['tie_or_weak'] / ev:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
