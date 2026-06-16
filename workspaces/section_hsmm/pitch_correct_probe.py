#!/usr/bin/env python3
"""Does undoing the pitch shift raise verifiability? (oracle test for John's key idea)

Pitch is the #1 unverifiability driver (why_unverifiable.py). DJs harmonic-mix,
so the acappella is pitched to the host key. This test pitch-corrects the MIX
window by the (oracle GT) shift — back to the candidate's native key — then
re-ranks all candidates at their native keys. If the pitch-shifted events become
verifiable, pitch is fixable, and the oracle shift gets replaced by John's
key-prediction (host_key - candidate_key, both on disk as Camelot tags).

Correcting the mix side (not the candidates) means distractors gain no extra
matching freedom — the clean version of the fix.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.pitch_correct_probe --set-id 1fsnxchk
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    SR, STRETCHES, detect_offset, find_aligning_dir, ref_audio_for,
)
from workspaces.section_hsmm.similarity_probe import _feat, _mfcc  # noqa: E402


def _win_mfcc(path, s0, dur, n_steps):
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, offset=max(0.0, s0), duration=dur)
        if abs(n_steps) > 0:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
    return _mfcc(y)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--max-win-s", type=float, default=15.0)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if r.get("claimed_stem") == "acappella" and r.get("track_id")]
    pool = [t for t in by_tid if ref_audio_for({"claimed_stem": "acappella"},
                                               by_tid[t]) is not None]
    mix_vocals = set_dir / "mix_vocals.flac"

    print(f"pitch-correction probe ({len(rows)} events, {len(pool)} pool) …", file=sys.stderr)
    res = []  # (pitch!=0, verif_uncorrected, verif_corrected)
    for r in rows:
        tid = str(r["track_id"])
        if tid not in pool:
            continue
        s0 = float(r["set_start_s"])
        dur = min(float(r["set_end_s"]) - s0, args.max_win_s)
        if dur < 2:
            continue
        pshift = int(r.get("pitch_shift_semi") or 0)

        def rank(win):
            peaks = {}
            for c in pool:
                rf = _feat(ref_audio_for({"claimed_stem": "acappella"}, by_tid[c]),
                           f"ref_{c}_voc", "mfcc")
                if rf.shape[1] > win.shape[1] >= 8:
                    peaks[c] = detect_offset(win, rf, STRETCHES)[1]
            if tid not in peaks:
                return None
            return all(peaks[tid] >= v for k, v in peaks.items() if k != tid)

        w_un = _win_mfcc(mix_vocals, s0, dur, 0)
        w_co = _win_mfcc(mix_vocals, s0, dur, -pshift) if pshift else w_un
        vu, vc = rank(w_un), rank(w_co)
        if vu is None or vc is None:
            continue
        res.append((pshift != 0, vu, vc))

    n = len(res)
    shifted = [r for r in res if r[0]]
    print(f"\n=== pitch-correction probe ({args.set_id}, {n} events) ===")
    print(f"{'subset':>18} {'n':>4} {'uncorrected':>12} {'pitch-corrected':>16}")
    print(f"{'ALL':>18} {n:>4} {100*np.mean([r[1] for r in res]):11.0f}% "
          f"{100*np.mean([r[2] for r in res]):15.0f}%")
    if shifted:
        print(f"{'pitch-shifted only':>18} {len(shifted):>4} "
              f"{100*np.mean([r[1] for r in shifted]):11.0f}% "
              f"{100*np.mean([r[2] for r in shifted]):15.0f}%")
        gained = sum(1 for r in shifted if r[2] and not r[1])
        lost = sum(1 for r in shifted if r[1] and not r[2])
        print(f"\non pitch-shifted events: {gained} newly verifiable, {lost} lost "
              f"(net {gained-lost:+d} of {len(shifted)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
