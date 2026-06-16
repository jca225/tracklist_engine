#!/usr/bin/env python3
"""Vocal-activity (singing) detection feasibility — does harmonicity separate
real singing from separation artifacts?

The hypothesis (John): mix_vocals has artifact/noise during instrumental
sections, which the matcher hallucinates onto (the sticky-distractor false
positives). Real singing is HARMONIC (clear F0, low spectral flatness);
artifacts are NOISE-like (flat). This computes per-frame voicedness features and
asks the decisive question: are the aligner's FALSE-positive spans in
lower-voicedness regions than its TRUE spans? If so, a VAD gate suppresses them.

Features: RMS energy, spectral flatness (noise=high), and pyin voiced-probability
(principled: is there a pitch?). Validated against the fused timeline + GT.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.vad_probe --set-id 1fsnxchk
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

from workspaces.alignment_prototype.refine_ref_offsets import SR, find_aligning_dir  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402

HOP = 1024
FPS = SR / HOP
OUT = Path(__file__).resolve().parent / "out"


def _features(path, set_id):
    cf = _CACHE / f"{set_id}_vad_feats.npz"
    if cf.is_file():
        z = np.load(cf)
        return z["rms"], z["flat"], z["vprob"]
    import librosa
    print("loading mix_vocals + computing VAD features (pyin is slow) …", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0]
        print("  pyin …", file=sys.stderr)
        f0, voiced, vprob = librosa.pyin(y, fmin=65, fmax=1200, sr=SR,
                                         hop_length=HOP, frame_length=4096)
    vprob = np.nan_to_num(vprob, nan=0.0)
    m = min(len(rms), len(flat), len(vprob))
    rms, flat, vprob = rms[:m], flat[:m], vprob[:m]
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.save  # noqa
    np.savez(cf, rms=rms, flat=flat, vprob=vprob)
    return rms, flat, vprob


def _auc(pos, neg):
    """Mann-Whitney AUC: P(pos > neg)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    args = p.parse_args(argv)
    set_dir = find_aligning_dir(args.set_id)
    rms, flat, vprob = _features(set_dir / "mix_vocals.flac", args.set_id)
    t = np.arange(len(rms)) / FPS

    def span_mean(feat, s0, s1):
        m = (t >= s0) & (t < s1)
        return float(feat[m].mean()) if m.any() else float("nan")

    tl = json.loads((OUT / f"{args.set_id}_fused_timeline.json").read_text())
    voc = [s for s in tl["spans"] if s["channel"] == "overlay"]
    import yaml
    gt = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if r.get("claimed_stem") == "acappella" and r.get("track_id")]

    def ov(s):
        return any(str(r["track_id"]) == s["recording_id"]
                   and s["set_start_s"] < float(r["set_end_s"]) + 3
                   and s["set_end_s"] > float(r["set_start_s"]) - 3 for r in gt)

    feats = {"voiced_prob": vprob, "1-flatness": 1 - flat / (flat.max() + 1e-9),
             "rms": rms}
    true_sp = [s for s in voc if ov(s)]
    false_sp = [s for s in voc if not ov(s)]
    # GT vocal-present mask vs gap mask (frame-level)
    pos_mask = np.zeros(len(rms), bool)
    for r in gt:
        pos_mask |= (t >= float(r["set_start_s"])) & (t < float(r["set_end_s"]))

    print(f"=== VAD feasibility ({args.set_id}) ===")
    print(f"true spans: {len(true_sp)}  false spans: {len(false_sp)}  "
          f"GT vocal frames: {pos_mask.mean()*100:.0f}%\n")
    print(f"{'feature':>14}  {'true-span':>10} {'false-span':>11}  "
          f"{'AUC T>F':>8}  {'AUC pos>gap':>11}")
    for name, f in feats.items():
        tv = [span_mean(f, s["set_start_s"], s["set_end_s"]) for s in true_sp]
        fv = [span_mean(f, s["set_start_s"], s["set_end_s"]) for s in false_sp]
        tv = [x for x in tv if np.isfinite(x)]; fv = [x for x in fv if np.isfinite(x)]
        auc_tf = _auc(tv, fv)
        auc_pg = _auc(f[pos_mask], f[~pos_mask])
        print(f"{name:>14}  {np.mean(tv):10.3f} {np.mean(fv):11.3f}  "
              f"{auc_tf:8.2f}  {auc_pg:11.2f}")
    print("\nAUC T>F > 0.5 means false positives sit in lower-voicedness regions "
          "(a VAD gate would suppress them). AUC pos>gap = frame-level "
          "vocal-present vs gap separability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
