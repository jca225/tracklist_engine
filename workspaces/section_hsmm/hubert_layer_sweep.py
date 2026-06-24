#!/usr/bin/env python3
"""HuBERT layer sweep for the overlay (acappella) channel abstain gate (BB12).

L9 was a default, not tuned. HuBERT computes all hidden states in one forward
pass, so this extracts layers 6-10 in a single pass per file (reusing any
existing per-layer frame cache, e.g. L9 from the pre-test), then runs the
abstain overlay decode per layer and prints a compact precision@coverage
summary. Decode is seconds once frame caches exist.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.hubert_layer_sweep \
        --set-id 1fsnxchk [--layers 6 7 8 9 10] [--frame-s 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import HOP, SR  # noqa: E402
from workspaces.section_hsmm.abstain_eval import _decode_channel  # noqa: E402
from workspaces.section_hsmm.decode_hsmm import _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.decode_v2 import CHANNELS  # noqa: E402
from workspaces.section_hsmm.decode_v3 import build_overlay_vocab_mert  # noqa: E402
from workspaces.section_hsmm.similarity_probe import (  # noqa: E402
    _HUBERT_CHUNK_S, _HUBERT_SR, _hubert_model, _resample_cols,
)
from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402

TAUS = (-1.0, 0.0, 0.02, 0.05, 0.10)


def _hubert_multi(y: np.ndarray, layers: list[int]) -> dict[int, np.ndarray]:
    """One forward pass -> {layer: (768, frames@SR/HOP)} L2-normed per frame."""
    import librosa
    import torch
    model, fe, dev = _hubert_model()
    y16 = librosa.resample(y, orig_sr=SR, target_sr=_HUBERT_SR)
    step = int(_HUBERT_CHUNK_S * _HUBERT_SR)
    acc: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    with torch.no_grad():
        for i in range(0, len(y16), step):
            chunk = y16[i:i + step]
            if len(chunk) < 400:
                continue
            iv = fe(chunk, sampling_rate=_HUBERT_SR, return_tensors="pt")
            hs = model(iv.input_values.to(dev), output_hidden_states=True).hidden_states
            for l in layers:
                acc[l].append(hs[l][0].float().cpu().numpy())
    n_out = max(1, int(round(len(y) / HOP)))
    out: dict[int, np.ndarray] = {}
    for l in layers:
        if not acc[l]:
            out[l] = np.zeros((768, 0), np.float32)
            continue
        h = _resample_cols(np.concatenate(acc[l], axis=0).T, n_out)
        out[l] = (h / (np.linalg.norm(h, axis=0, keepdims=True) + 1e-8)).astype(np.float32)
    return out


def _prepopulate(items: list[tuple[Path, str]], layers: list[int]) -> None:
    import librosa
    import warnings
    _CACHE.mkdir(parents=True, exist_ok=True)
    for i, (path, key) in enumerate(items, 1):
        missing = [l for l in layers if not (_CACHE / f"{key}_hubertL{l}.npy").is_file()]
        if not missing:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=SR, mono=True)
        feats = _hubert_multi(y, missing)
        for l in missing:
            np.save(_CACHE / f"{key}_hubertL{l}.npy", feats[l])
        if i % 10 == 0 or i == len(items):
            print(f"  frame-cache [{i}/{len(items)}] (+{len(missing)} layers)", file=sys.stderr)


def _summary(dec_tid, conf, frame_s, gt_rows):
    rows = [r for r in gt_rows
            if (r.get("claimed_stem") or "regular") in CHANNELS["overlay"]["stems"]]

    def active(ts):
        return {str(r["track_id"]) for r in rows
                if float(r["set_start_s"]) - 2 <= ts <= float(r["set_end_s"]) + 2}

    frames = []
    for i, tid in enumerate(dec_tid):
        act = active((i + 0.5) * frame_s)
        if act:
            frames.append((tid in act, float(conf[i])))
    n = len(frames)
    out = {"acc": np.mean([c for c, _ in frames]) if n else 0.0, "n": n, "pc": {}}
    for tau in TAUS:
        pred = [(c, cf) for c, cf in frames if cf >= tau]
        out["pc"][tau] = (len(pred) / n, np.mean([c for c, _ in pred])) if pred else (0.0, None)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--layers", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    gt_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]
    stem_by_tid: dict[str, str] = {}
    for r in gt_rows:
        stem_by_tid.setdefault(str(r["track_id"]), r.get("claimed_stem") or "regular")
    for tid, t in by_tid.items():
        t["_stem"] = stem_by_tid.get(tid, "regular")
    gt_tids = _gt_track_ids(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")

    # one multi-layer forward pass over mix + overlay refs
    key_of, items = build_overlay_vocab_mert(by_tid, gt_tids, args.frame_s, args.layers[0])
    items = items + [(set_dir / CHANNELS["overlay"]["mix_file"], f"{args.set_id}_mix_vocals")]
    print(f"prepopulating frame caches for layers {args.layers} …", file=sys.stderr)
    _prepopulate(items, args.layers)

    print(f"\n=== overlay abstain layer sweep ({args.set_id}) ===")
    print(f"{'layer':>5} {'no-abst':>8} | "
          + " ".join(f"m>={t:<4g}" for t in TAUS) + "   (cov%/prec%)")
    for layer in args.layers:
        dec_tid, conf = _decode_channel("overlay", set_dir, by_tid, gt_tids,
                                        args.frame_s, layer, "hubert")
        s = _summary(dec_tid, conf, args.frame_s, gt_rows)
        cells = []
        for tau in TAUS:
            cov, prec = s["pc"][tau]
            cells.append(f"{100*cov:3.0f}/{'--' if prec is None else f'{100*prec:3.0f}':>3}")
        print(f"L{layer:<4} {100*s['acc']:7.0f}% | " + " ".join(f"{c:>8}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
