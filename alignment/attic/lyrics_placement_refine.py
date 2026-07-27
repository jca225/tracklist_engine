#!/usr/bin/env python3
"""Wire the lyrics-ASR channel's set_start into the predicted timeline.

The lyrics channel (lyrics_align.py) places acappella spans at 2.3 s median
vs GT (BB12, n=49) — ~18x better than the acoustic placement path — but was
never wired into production: the timelines the ref-decoders consume still
carry acoustic placement (measured tens of seconds median error on
acappella), which is the binding end-to-end constraint (oracle 43-44% vs
real 13-14% trajectory accuracy).

This pass consumes a predicted timeline (NOT ground truth): for each
acappella span it takes the span's assigned recording's vocal-stem
transcript (cache only — warm it with lyrics_align --transcribe), finds
word-bigram diagonals against the mix transcript, runs the same
position-prior monotonic decode with the TIMELINE's own set_start as the
prior, and rewrites set_start/set_end (window shift, duration kept) for
non-abstained placements within a sanity leash. Identity is taken from the
timeline, never re-decided here.

Usage:
    venvs/audio/bin/python -m alignment.lyrics_placement_refine \
        --set-id 1fsnxchk [--timeline out/<set>_predicted_timeline.json] \
        [--out out/<set>_predicted_timeline_lyr.json] [--leash-s 120]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.lyrics_align import (  # noqa: E402
    _bigram_times,
    _norm,
    candidate_diagonals,
    load_cached,
    monotonic_decode,
)
from alignment.refine_ref_offsets import (  # noqa: E402
    find_aligning_dir,
)

OUT_DIR = Path(__file__).resolve().parent / "out"


def vocal_stem_for(track: dict, set_dir: Path) -> Path | None:
    """Vocal stem path with a DISK fallback: the manifest's `stems` field is
    only populated for tracks whose stems pi-storage considers canonical
    (BB11: 69/147 tracks) — but the aligning folder holds stem dirs for
    nearly every slot, named by the slot prefix. Resolve by exact basename,
    then by slot prefix (unique per slot)."""
    import glob as _glob

    vp = (track.get("stems") or {}).get("vocals")
    if vp and Path(vp).is_file():
        return Path(vp)
    lp = track.get("local_path")
    if not lp:
        return None
    base = Path(lp).stem
    hits = _glob.glob(str(set_dir / "stems" / (_glob.escape(base) + "*") / "vocals.flac"))
    if not hits:
        slot = base.split("__")[0]
        hits = _glob.glob(str(set_dir / "stems" / (slot + "__*") / "vocals.flac"))
    return Path(sorted(hits)[0]) if hits else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--timeline", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--transcribe",
        action="store_true",
        help="warm missing vocal-stem transcripts for this timeline's "
        "acappella spans (Whisper, ~2-3 min/stem on MPS) instead of refining",
    )
    p.add_argument(
        "--leash-s",
        type=float,
        default=120.0,
        help="reject a lyrics placement further than this from the acoustic "
        "prior (sanity catch; acoustic p90 is ~70 s)",
    )
    args = p.parse_args(argv)

    tl_path = args.timeline or (OUT_DIR / f"{args.set_id}_predicted_timeline.json")
    timeline = json.loads(tl_path.read_text())
    spans = timeline["spans"]

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_words = load_cached(set_dir / "mix_vocals.flac")
    if not mix_words:
        sys.exit(
            "mix_vocals transcript not cached — run lyrics_align --transcribe first"
        )
    mix_bt = _bigram_times(_norm(mix_words))

    # acappella spans in mix order (the monotonic decode's tracklist order)
    aca = [
        (i, s)
        for i, s in enumerate(spans)
        if (s.get("claimed_stem") or "regular") == "acappella"
    ]
    aca.sort(key=lambda q: float(q[1]["set_start_s"]))

    dec_in, meta = [], []
    n_nostem = n_notrans = n_nocands = 0
    for idx, s in aca:
        track = by_tid.get(s["recording_id"])
        vpath = vocal_stem_for(track, set_dir) if track else None
        if not vpath:
            n_nostem += 1
            continue
        if args.transcribe:
            from alignment.lyrics_align import transcribe_words

            if not load_cached(vpath):
                print(f"transcribing {vpath.name} ({vpath.parent.name}) …")
                transcribe_words(vpath)
            continue
        cw = load_cached(vpath)
        if not cw:
            n_notrans += 1
            continue
        cands = candidate_diagonals(_norm(cw), mix_bt)
        if not cands:
            n_nocands += 1
            continue
        dec_in.append((cands, float(s["set_start_s"])))
        meta.append(idx)

    if args.transcribe:
        print("transcription warm-up done")
        return 0
    chosen = monotonic_decode(dec_in) if dec_in else []
    n_upd = n_abstain = n_leash = 0
    for (ss, rs), idx in zip(chosen, meta):
        s = spans[idx]
        if ss is None:
            n_abstain += 1
            continue
        prior = float(s["set_start_s"])
        if abs(ss - prior) > args.leash_s:
            n_leash += 1
            continue
        dur = float(s["set_end_s"]) - prior
        s.setdefault("set_start_acoustic", prior)
        s["set_start_s"] = round(float(ss), 3)
        s["set_end_s"] = round(float(ss) + dur, 3)
        s["ref_start_s"] = round(float(rs), 3)  # decoder refines further
        s["placement"] = "lyrics"
        n_upd += 1

    timeline["lyrics_placement"] = {
        "updated": n_upd,
        "abstained": n_abstain,
        "leash_rejected": n_leash,
        "no_stem": n_nostem,
        "no_transcript": n_notrans,
        "no_candidates": n_nocands,
    }
    dest = args.out or tl_path
    dest.write_text(json.dumps(timeline, indent=2))
    print(
        f"lyrics placement: updated {n_upd}/{len(aca)} acappella spans "
        f"(abstain {n_abstain}, leash {n_leash}, no-stem {n_nostem}, "
        f"no-transcript {n_notrans}, no-candidates {n_nocands})"
    )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
