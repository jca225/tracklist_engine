#!/usr/bin/env python3
"""Stem-correction pass: fix the SCRAPED tracklist stem (set_track_slots.
claimed_stem — the aligner's INPUT) before alignment, validated against the
hand GT (authoritative truth).

Heuristics #2/#6. BB12 measured the input is wrong for 44% of tracks (scrape
calls overlay-acappellas 'regular' because the tracklist text lacks an
"(Acappella)" suffix). Two signals fuse here:

  * STRUCTURAL prior — `is_concurrent`/sided 'w' rows are ~79% acap/instr
    (mostly acappella); main-line rows are NEVER acappella (full or instr host
    bed). This alone lifts stem accuracy ~56%->~74% on BB12.
  * AUDIO — debiased cross-channel matched filter (`_xchan_job` z-score = peak
    in units of the channel's own score noise). Overrides the structural
    default only when a channel is a confident, clear winner (ZABS + ZMARGIN),
    so noisy audio can't undo the strong prior. DETECT-AND-FLAG, never silently
    rewrite ([[feedback_correctness_vs_accuracy]]).

Produce the slots dump first:
    ssh pi-storage 'sqlite3 -json /mnt/.../music_database.db \\
      "SELECT row_index,slot_label,track_id,recording_id,claimed_stem,
       is_concurrent,full_name FROM set_track_slots WHERE set_id=\\"<sid>\\""' \\
      > /tmp/<sid>_slots.json

    venvs/audio/bin/python -m workspaces.alignment_prototype.stem_correct \\
        --set-id 1fsnxchk --slots-json /tmp/1fsnxchk_slots.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.continuity_refine import (  # noqa: E402
    _CHANNELS, _grid_stretches, _probe_offsets, _windows_from, _xchan_job,
)
from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP, SR, _MIX_SOURCE, chroma, find_aligning_dir,
)

ZABS = 3.0     # a channel must clear this z (clear matched-filter detection)
ZMARGIN = 1.0  # ...and beat the structural default by this, to override it
STEMS = ("regular", "acappella", "instrumental")


def _truth_stem(t) -> str:
    return (getattr(t, "claimed_stem", None) or "regular")


def correct_stem(is_conc: bool, full_name: str, z: dict) -> tuple[str, str]:
    """(predicted_stem, reason) from text + structural prior + audio.

    Audio discriminator is INSTRUMENTAL-PRESENCE, not channel-argmax: vocals
    match for both acappellas and full tracks, so the vocal channel can't tell
    them apart. The song's instrumental, however, is present only if it was
    actually played — so a confident instrumental-channel match (mix_instr vs
    ref instr stem) means NOT a pure acappella. Absence (low instr z) + the
    sided prior => acappella."""
    name = (full_name or "").lower()
    if "acappella" in name or "acapella" in name:
        return "acappella", "text"
    if "instrumental" in name:
        return "instrumental", "text"

    def ok(x):
        return x is not None and not math.isnan(x)

    zi = z.get("instrumental")
    zv = z.get("acappella")  # vocal channel
    instr_present = ok(zi) and zi >= ZABS
    vocal_present = ok(zv) and zv >= ZABS

    if not is_conc:
        # main-line: never acappella. Instrumental if its stem is present and
        # vocals are not; else the full track (regular).
        if instr_present and not vocal_present:
            return "instrumental", "audio"
        return "regular", "prior"

    # sided/concurrent overlay: acappella prior. Demote only if the song's
    # instrumental is actually present in the mix.
    if instr_present:
        return ("regular", "audio") if vocal_present else ("instrumental", "audio")
    return "acappella", "prior"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--slots-json", type=Path, required=True)
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--probes", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    import librosa
    from core.result import Err, Ok
    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    match load_set(args.gt):
        case Err(msg):
            sys.exit(f"GT load failed: {msg}")
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Err(msg):
            sys.exit(f"MERT bundle (grids) load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass

    slots = json.loads(args.slots_json.read_text())
    scrape_by_tid = {s["track_id"]: s for s in slots if s.get("track_id")}
    print(f"set={gt.set_id} GT spans={len(targets)} scrape slots={len(slots)}")

    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_chroma: dict[str, np.ndarray] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        f = set_dir / fname
        if f.is_file():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _sr = librosa.load(str(f), sr=SR, mono=True)
            mix_chroma[stem] = chroma(y)

    n = int(args.window_s * SR / HOP)
    jobs, meta, seen = [], [], set()
    for i, t in enumerate(targets):
        if t.slot_label == "mix" or t.recording_id in seen:
            continue
        scr = scrape_by_tid.get(t.recording_id)
        track = by_tid.get(t.recording_id)
        if scr is None or track is None:
            continue
        seen.add(t.recording_id)
        span_len = max(0.0, t.set_end_s - t.set_start_s)
        dts = _probe_offsets(span_len, args.window_s, args.probes)
        stretches = _grid_stretches(t, mix_series, ref_series)
        chans = []
        for name, mix_key, ref_stem in _CHANNELS:
            mc = mix_chroma.get(mix_key)
            if mc is None:
                continue
            ref_path = (track.get("local_path") if ref_stem is None
                        else (track.get("stems") or {}).get(ref_stem))
            if not ref_path or not Path(ref_path).is_file():
                continue
            wl = _windows_from(mc, t.set_start_s, dts, n)
            if wl:
                chans.append((name, str(ref_path), wl, stretches, 0))
        jobs.append((i, t.ref_start_s, chans))
        meta.append((t, scr))

    print(f"scoring {len(jobs)} unique tracks (audio channels per track)…")
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_xchan_job, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    # evaluate: B0 raw scrape, B1 structural-only, B2 structural+audio
    rows = []
    for (i, _g, _c), (t, scr) in zip(jobs, meta):
        truth = _truth_stem(t)
        is_conc = bool(scr.get("is_concurrent"))
        z = {c["ch"]: c["z"] for c in res[i]["channels"]}
        b0 = scr["claimed_stem"]
        b1, _ = correct_stem(is_conc, scr.get("full_name") or "", {})  # no audio
        b2, why = correct_stem(is_conc, scr.get("full_name") or "", z)
        rows.append((t, scr, truth, is_conc, b0, b1, b2, why, z))

    def acc(idx, sel=None):
        rr = sel if sel is not None else rows
        return sum(1 for r in rr if r[idx] == r[2]) / len(rr) if rr else 0.0

    # explicitly-tagged truth only (untagged GT rows default to 'regular' and
    # pollute the regular class — a high-vocal/low-instr audio sig on a
    # "regular" truth is often an untagged acappella)
    explicit = [r for r in rows if getattr(r[0], "claimed_stem", None)]
    print(f"\n=== stem accuracy vs GT (n={len(rows)} tracks) ===")
    print(f"  B0 raw scrape            : {100*acc(4):.0f}%")
    print(f"  B1 structural prior only : {100*acc(5):.0f}%")
    print(f"  B2 structural + audio    : {100*acc(6):.0f}%")
    print(f"  -- on explicitly-tagged truth only (n={len(explicit)}): "
          f"B0 {100*acc(4,explicit):.0f}%  B1 {100*acc(5,explicit):.0f}%  "
          f"B2 {100*acc(6,explicit):.0f}%")

    def prf(idx, cls):
        tp = sum(1 for r in rows if r[idx] == cls and r[2] == cls)
        fp = sum(1 for r in rows if r[idx] == cls and r[2] != cls)
        fn = sum(1 for r in rows if r[idx] != cls and r[2] == cls)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return pr, rc, tp, fp, fn

    print("\nper-class (priority = acappella):")
    for cls in STEMS:
        for name, idx in (("B0", 4), ("B2", 6)):
            pr, rc, tp, fp, fn = prf(idx, cls)
            print(f"  {cls:12} {name}  prec {100*pr:3.0f}%  recall {100*rc:3.0f}% "
                  f" (tp{tp} fp{fp} fn{fn})")

    flips_fixed = [r for r in rows if r[4] != r[2] and r[6] == r[2]]
    flips_broke = [r for r in rows if r[4] == r[2] and r[6] != r[2]]
    print(f"\nB2 fixed {len(flips_fixed)} scrape errors, broke {len(flips_broke)}:")
    for t, scr, truth, ic, b0, b1, b2, why, z in flips_broke:
        zs = " ".join(f"{k}={z[k]:.1f}" for k in z if not math.isnan(z[k]))
        print(f"  BROKE {scr['slot_label']:6} scrape={b0:11} truth={truth:11} "
              f"pred={b2:11} via {why}  [{zs}]  {(scr.get('full_name') or '')[:34]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
