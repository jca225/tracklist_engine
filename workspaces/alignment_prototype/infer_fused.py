#!/usr/bin/env python3
"""infer_fused — fused-pipeline inference on a LOCAL pulled set.

Runs the fused method (validated on UnmixDB) on a real scraped set in
``~/aligning/<set>/`` with **no pi-storage and no MERT**: the tracklist's
manifest tracks are the candidate pool, the set's ``mix.m4a`` is the mix, and for
each candidate we get

  - identity / presence = landmark-fingerprint vote count + sharpness (abstain
    when weak — the open-set safety the harness showed at ~85-90% rank@1)
  - placement (set_start in the mix) = fingerprint offset histogram
    (``set_start ≈ -offset``; robust, no heavy tail)

The mix is fingerprinted ONCE; each candidate votes against it. Output is a
predicted-timeline JSON next to the review tooling's format.

    venvs/audio/bin/python -m workspaces.alignment_prototype.infer_fused \\
        --set-id 1fsnxchk [--max-tracks N] [--vote-floor 20] [--sharp-floor 1.3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.landmark_fp import (  # noqa: E402
    FHOP,
    SR,
    constellation,
    hashes,
    vote_sharpness,
    _vote_histogram,
)
from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir  # noqa: E402

OUT_DIR = _REPO / "workspaces/alignment_prototype/out"
_SLOT_RE = re.compile(r"^(\d{3}(?:w\d+)?)")


def _slot_of(track: dict) -> str:
    stem = Path(track.get("local_path", "")).stem
    m = _SLOT_RE.match(stem)
    return m.group(1) if m else (track.get("axes_key") or track.get("track_id", "?"))


def _fp_place(
    hm: dict, ty, stretches: tuple[float, ...]
) -> tuple[float, int, float, float]:
    """(set_start_s, votes, stretch, sharpness) of a candidate vs the mix hashes."""
    import librosa

    best = (0.0, 0, 1.0, 0.0)
    for st in stretches:
        if abs(st - 1.0) > 1e-3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ry = librosa.effects.time_stretch(ty, rate=1.0 / st)
        else:
            ry = ty
        votes = _vote_histogram(hm, hashes(*constellation(ry)))
        if not votes:
            continue
        off, v = max(votes.items(), key=lambda kv: kv[1])
        if v > best[1]:
            best = (max(0.0, -(off * FHOP / SR * st)), v, st, vote_sharpness(votes))
    return best


def fused_infer(
    set_id: str,
    *,
    max_tracks: int | None = None,
    stretches: tuple[float, ...] = (0.96, 1.0, 1.04),
    vote_floor: int = 20,
    sharp_floor: float = 1.3,
) -> tuple[Path, list[dict]]:
    import librosa

    set_dir = find_aligning_dir(set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    mix_path = set_dir / "mix.m4a"
    if not mix_path.is_file():
        raise FileNotFoundError(f"no mix.m4a in {set_dir}")

    print(f"fingerprinting mix {mix_path.name} (once)…", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        my, _ = librosa.load(str(mix_path), sr=SR, mono=True)
    hm = hashes(*constellation(my))

    tracks = manifest["tracks"]
    if max_tracks:
        tracks = tracks[:max_tracks]
    preds = []
    for i, t in enumerate(tracks):
        tp = t.get("local_path")
        if not tp or not Path(tp).is_file():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ty, _ = librosa.load(str(tp), sr=SR, mono=True)
        set_start, votes, st, sharp = _fp_place(hm, ty, stretches)
        present = votes >= vote_floor and sharp >= sharp_floor
        preds.append(
            dict(
                slot_label=_slot_of(t),
                recording_id=t.get("recording_id") or t.get("track_id"),
                label=t.get("label")
                or f"{t.get('artist', '')} - {t.get('title', '')}".strip(" -"),
                set_start_s=round(set_start, 2),
                votes=int(votes),
                stretch=round(st, 3),
                sharpness=round(sharp, 2),
                present=bool(present),
            )
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(tracks)}", flush=True)

    preds.sort(key=lambda p: p["set_start_s"])
    return set_dir, preds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--set-id", required=True)
    p.add_argument("--max-tracks", type=int, default=None)
    p.add_argument("--vote-floor", type=int, default=20)
    p.add_argument("--sharp-floor", type=float, default=1.3)
    args = p.parse_args(argv)

    set_dir, preds = fused_infer(
        args.set_id,
        max_tracks=args.max_tracks,
        vote_floor=args.vote_floor,
        sharp_floor=args.sharp_floor,
    )
    present = [p for p in preds if p["present"]]
    print(
        f"\n{len(present)}/{len(preds)} candidates present (vote>={args.vote_floor}, "
        f"sharp>={args.sharp_floor}); rest abstained\n"
    )
    print(f"{'slot':6} {'set_start':>9} {'votes':>6} {'sharp':>5}  label")
    for p in preds:
        mark = " " if p["present"] else "·"
        print(
            f"{mark}{p['slot_label']:5} {p['set_start_s']:9.1f} {p['votes']:6d} "
            f"{p['sharpness']:5.2f}  {p['label'][:46]}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.set_id}_fused_timeline.json"
    out_path.write_text(
        json.dumps({"set_id": args.set_id, "method": "fused", "spans": preds}, indent=1)
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
