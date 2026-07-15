"""Stage-1 scorer-invariance triage for ensemble reduction (streaming_mir).

The ensemble-reduction A/B (speed_quality_ab.py) showed p2=1.27x / p1=2.42x but
lower stem SDR vs the p3 reference. SDR alone is not the deploy gate — the vocal
stems feed HuBERT vocal probes, so the question is whether reduction moves the
vocals in the SPACE THE PROBE READS (HuBERT L9 frames), not in raw waveform.

This measures, per track, the frame-wise cosine similarity of HuBERT-L9 features
between each reduced arm's vocals and the p3 reference's vocals. High cosine
(≈1.0) ⇒ the probe cannot see the difference ⇒ votes cannot move ⇒ gate leans
PASS. A drop ⇒ escalate to the full capture_votes gate on the GT sets.

Reuses the harness HuBERT extractor (workspaces.section_hsmm.similarity_probe._hubert)
so the features are identical to what the live probe computes.

Run on a CUDA/MPS box:
  /venv/main/bin/python workspaces/streaming_mir/hubert_invariance.py \
      --root /workspace/sq_ab_out --ref-arm p3_fp32 --arms p2_fp32 p1_fp32
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LAYER = 9


def hubert_frames(path: Path) -> np.ndarray:
    import librosa

    # _hubert expects y at the harness SR (it resamples SR->16k internally);
    # load at that exact rate or the internal resample is wrong.
    from workspaces.alignment_prototype.refine_ref_offsets import SR
    from workspaces.section_hsmm.similarity_probe import _hubert

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return _hubert(y, LAYER)  # (768, T)


def frame_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-frame cosine over the shared frame span (same source audio,
    so frames are already time-aligned; separators only change content)."""
    t = min(a.shape[1], b.shape[1])
    a, b = a[:, :t], b[:, :t]
    an = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-9)
    return float((an * bn).sum(axis=0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ref-arm", default="p3_fp32")
    ap.add_argument("--arms", nargs="+", default=["p2_fp32", "p1_fp32"])
    ap.add_argument("--stem", default="vocals", choices=["vocals", "instrumental"])
    args = ap.parse_args()

    track_dirs = sorted(
        p.name for p in (args.root / args.ref_arm).iterdir() if p.is_dir()
    )
    print(f"tracks: {track_dirs} stem={args.stem}\n")

    ref_feats = {}
    for t in track_dirs:
        ref_feats[t] = hubert_frames(args.root / args.ref_arm / t / f"{args.stem}.flac")

    print(f"{'arm':10s} {'per-track cosine vs ' + args.ref_arm:40s} mean")
    for arm in args.arms:
        cs = []
        for t in track_dirs:
            f = hubert_frames(args.root / arm / t / f"{args.stem}.flac")
            cs.append(frame_cosine(ref_feats[t], f))
        pertrack = " ".join(f"{c:.4f}" for c in cs)
        print(f"{arm:10s} {pertrack:40s} {sum(cs) / len(cs):.4f}")
    print(
        "\nread: >0.98 mean ⇒ probe-invisible, gate leans PASS (escalate only if"
        " borderline);\n<0.95 ⇒ real perturbation, run full capture_votes on GT."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
