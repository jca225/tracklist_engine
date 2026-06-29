"""End-to-end fingerprint placement eval: decode_placements vs GT.

Runs the committed placement pipeline (offset_candidates -> monotonic decode over
tracklist order) on a labeled set and scores set_start / set_end against GT. This
is the placement stage of the aligner as a single runnable, validating the
2026-06-28 reframe (the ~30s set_start "wall" was conflating set_start with the
alignment offset; DJs start mid-song). See project_placement_wall_was_decomposition_error.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.eval_placement \
        [--gt labeling/fixtures/bb12_ground_truth.yaml] [--stems regular]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.result import Err, Ok
from workspaces.alignment_prototype.dataset import load_set
from workspaces.alignment_prototype.fp_index import FpKey, load as fp_load
from workspaces.alignment_prototype.landmark_fp import SR, constellation, hashes
from workspaces.alignment_prototype.mix_fp_hits import decode_placements, load_mix_mono
from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir


def _slot_key(s: str | None) -> tuple[int, int]:
    m = re.match(r"(\d+)(?:w(\d+))?", s or "")
    return (int(m.group(1)) if m else 9999, int(m.group(2)) if m and m.group(2) else 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--stems", default="regular")
    p.add_argument("--mix-file", default="mix.m4a")
    args = p.parse_args(argv)
    want = {s.strip() for s in args.stems.split(",") if s.strip()}

    match load_set(args.gt):
        case Err(m):
            sys.exit(f"GT load failed: {m}")
        case Ok((gt, targets)):
            pass

    set_dir = find_aligning_dir(gt.set_id)
    if set_dir is None:
        sys.exit(f"no aligning dir for {gt.set_id}")
    spans = [
        t
        for t in targets
        if (t.claimed_stem or "regular") in want
        and t.recording_id
        and t.slot_label != "mix"
    ]
    spans = sorted(spans, key=lambda t: _slot_key(t.slot_label))  # tracklist order

    print(f"hashing mix once ({args.mix_file})…", file=sys.stderr, flush=True)
    mix = load_mix_mono(set_dir / args.mix_file)
    hm = hashes(*constellation(mix))
    dur = len(mix) / SR

    kept = [(t, fp_load(FpKey(t.recording_id, "regular"))) for t in spans]
    kept = [(t, fp) for t, fp in kept if fp is not None]
    if not kept:
        sys.exit(
            "no ref fingerprints in cache — run scripts/backfill_track_fingerprints.py"
        )
    ref_fps = [fp for _t, fp in kept]
    preds = decode_placements(hm, ref_fps, mix_dur_s=dur)

    ss_err, se_err, abst = [], [], 0
    for (t, _fp), pr in zip(kept, preds):
        if pr is None:
            abst += 1
            continue
        ss_err.append(abs(pr[0] - t.set_start_s))
        se_err.append(abs(pr[1] - t.set_end_s))
    ss = np.array(ss_err)
    se = np.array(se_err)
    print(
        f"\n=== placement (set {gt.set_id}, stems={sorted(want)}, n={len(ss)}, abstain={abst}) ==="
    )
    for name, e in (("set_start", ss), ("set_end", se)):
        if not e.size:
            continue
        print(
            f"  {name:9} median={np.median(e):5.1f}s  mean={e.mean():6.1f}s  "
            f"<4s={100 * np.mean(e < 4):3.0f}%  <8s={100 * np.mean(e < 8):3.0f}%  <15s={100 * np.mean(e < 15):3.0f}%"
        )
    print(
        f"  (median/<15s are the stable metrics; mean is dominated by a few wrong-diagonal spans)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
