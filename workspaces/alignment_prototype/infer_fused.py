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


def _resolve_ref(track: dict, set_dir: Path) -> Path | None:
    """On-disk ref path for a manifest track, tolerant of unambiguous renames.

    ``tag_aligning_folder.py`` appends ``[NNNbpm KK]`` tags to the track
    filenames *after* the manifest is written, so ``local_path`` goes stale.
    Prefer ``audio_index.json`` by ``track_audio_id``, then a slot-prefix glob
    only when it identifies exactly one file; ambiguous copies abstain rather
    than silently selecting the first one.
    """
    from labeling.identity.audio_index import load_audio_index, lookup_ref

    p = track.get("local_path")
    if p and Path(p).is_file():
        return Path(p)
    indexed = lookup_ref(load_audio_index(set_dir), track.get("track_audio_id"))
    if indexed is not None:
        return indexed
    slot = _slot_of(track)
    candidates = [
        hit
        for hit in sorted((set_dir / "tracks").glob(f"{slot}__*"))
        if hit.suffix != ".asd" and hit.is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        warnings.warn(
            "ambiguous reference fallback: "
            f"track_audio_id={track.get('track_audio_id', '?')} "
            f"slot={slot} candidates={len(candidates)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


def _fp_place(
    hm: dict, ty, stretches: tuple[float, ...]
) -> tuple[float, int, float, float, float]:
    """(set_start_s, votes, stretch, sharpness, diag_b) of a candidate vs the mix.

    ``diag_b`` is the UNCLAMPED mix-time where the reference's t=0 lands on the
    fingerprint diagonal ``mix_t = stretch * ref_t + diag_b`` — negative when the
    DJ dropped the track mid-song. It's what lets us recover the ref window for a
    render; ``set_start_s`` is the same value clamped to >=0 for display.
    """
    import librosa

    best = (0.0, 0, 1.0, 0.0, 0.0)
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
            b = -(off * FHOP / SR * st)
            best = (max(0.0, b), v, st, vote_sharpness(votes), b)
    return best


def _add_render_windows(
    preds: list[dict], *, entry_floor: int = 40, min_span_s: float = 20.0
) -> None:
    """Fill set_end_s / ref_start_s / ref_end_s in place from the fp diagonal.

    A span runs from its entry until the next confident entry (votes>=entry_floor),
    capped at where the reference naturally ends; the ref window is read off the
    diagonal ``mix_t = stretch*ref_t + diag_b`` so span stretch stays consistent.
    """
    entries = sorted(p["set_start_s"] for p in preds if p["votes"] >= entry_floor)
    for p in preds:
        b, st, dur, ss = p["diag_b"], p["stretch"], p["ref_dur_s"], p["set_start_s"]
        natural_end = b + st * dur
        nxt = next((e for e in entries if e > ss + 1.0), None)
        set_end = (
            natural_end if nxt is None else min(natural_end, max(nxt, ss + min_span_s))
        )
        set_end = max(set_end, ss + min_span_s)
        p["set_end_s"] = round(set_end, 2)
        p["ref_start_s"] = round(max(0.0, (ss - b) / st), 2)
        p["ref_end_s"] = round(min(dur, (set_end - b) / st), 2)


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
        rp = _resolve_ref(t, set_dir)
        if rp is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ty, _ = librosa.load(str(rp), sr=SR, mono=True)
        set_start, votes, st, sharp, diag_b = _fp_place(hm, ty, stretches)
        present = votes >= vote_floor and sharp >= sharp_floor
        preds.append(
            dict(
                slot_label=_slot_of(t),
                recording_id=t.get("recording_id") or t.get("track_id"),
                label=t.get("label")
                or f"{t.get('artist', '')} - {t.get('title', '')}".strip(" -"),
                claimed_stem="regular",
                set_start_s=round(set_start, 2),
                votes=int(votes),
                stretch=round(st, 3),
                sharpness=round(sharp, 2),
                present=bool(present),
                diag_b=round(diag_b, 3),
                ref_dur_s=round(len(ty) / SR, 2),
                ref_path=str(rp),
            )
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(tracks)}", flush=True)

    preds.sort(key=lambda p: p["set_start_s"])
    _add_render_windows(preds)
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
