#!/usr/bin/env python3
"""v7 pre-test — does VOCAL similarity separate the true acappella from others?

John's reframe: stop searching 1-of-N by weak embeddings; instead VERIFY a
proposed acappella by matching the isolated mix vocals against the candidate's
actual vocal (near-exact, same recording). This probes whether that signal is
discriminative at all, before building a verification decoder.

For each GT acappella span: take mix_vocals at the GT mix-time, matched-filter it
(MFCC features — vocal timbre, not chroma) against the TRUE candidate's vocal
stem and against N distractor tracks' vocals. If the true candidate's peak
reliably outranks distractors, verification works.

Label-agnostic by design (John's 2nd point: the 'acappella' tag is scraped and
often missing/wrong) — distractors are drawn from ALL set tracks, not just
acappella-tagged ones.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.similarity_probe \
        --set-id 1fsnxchk [--n-distractors 15] [--feature mfcc|chroma]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP, SR, chroma, detect_offset, find_aligning_dir, ref_audio_for,
)
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402

FPS = SR / HOP


def _mfcc(y: np.ndarray) -> np.ndarray:
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=20)[1:]  # drop energy
    return (m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-8)).astype(np.float32)


def _feat(audio_path: Path, cache_key: str, feature: str) -> np.ndarray:
    cf = _CACHE / f"{cache_key}_{feature}.npy"
    if cf.is_file():
        return np.load(cf)
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    f = _mfcc(y) if feature == "mfcc" else chroma(y)
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cf, f)
    return f


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--n-distractors", type=int, default=15)
    p.add_argument("--feature", choices=["mfcc", "chroma"], default="mfcc")
    p.add_argument("--max-win-s", type=float, default=15.0)
    args = p.parse_args(argv)

    import json
    import yaml
    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    rows = [r for r in yaml.safe_load(args.gt.read_text())["tracks"]
            if (r.get("claimed_stem") == "acappella") and r.get("track_id")
            and not r.get("is_loop") and not r.get("ref_segments")]
    # label-agnostic distractor pool: ALL set tracks with a usable vocal stem
    pool = [t for t in by_tid if ref_audio_for({"claimed_stem": "acappella"},
                                               by_tid[t]) is not None]

    print(f"chroma/mfcc({args.feature}) of mix_vocals …", file=sys.stderr)
    mix = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals", args.feature)

    ranks, margins, true_peaks, dist_peaks = [], [], [], []
    for r in rows:
        tid = str(r["track_id"])
        if tid not in pool:
            continue
        a = int(float(r["set_start_s"]) * FPS)
        n = int(min(float(r["set_end_s"]) - float(r["set_start_s"]), args.max_win_s) * FPS)
        a = min(a, max(0, mix.shape[1] - n))
        win = mix[:, a:a + n]
        if win.shape[1] < 8:
            continue
        # distractors: first N pool tracks != true (deterministic)
        cands = [tid] + [t for t in pool if t != tid][:args.n_distractors]
        peaks = {}
        for c in cands:
            rp = ref_audio_for({"claimed_stem": "acappella"}, by_tid[c])
            rf = _feat(rp, f"ref_{c}_voc", args.feature)
            if rf.shape[1] <= win.shape[1]:
                continue
            _, peak, _ = detect_offset(win, rf)
            peaks[c] = peak
        if tid not in peaks or len(peaks) < 3:
            continue
        tp = peaks[tid]
        others = sorted((v for k, v in peaks.items() if k != tid), reverse=True)
        rank = 1 + sum(1 for v in others if v > tp)
        ranks.append(rank)
        margins.append(tp - others[0])
        true_peaks.append(tp)
        dist_peaks.append(others[0])

    n = len(ranks)
    if not n:
        print("no scorable spans", file=sys.stderr)
        return 1
    ranks = np.array(ranks)
    print(f"=== v7 vocal-verification pre-test ({args.set_id}, {args.feature}, "
          f"{n} acappella spans, 1 true vs {args.n_distractors} distractors) ===")
    print(f"retrieval@1 (true is top match): {100*(ranks==1).mean():.0f}%  "
          f"@3: {100*(ranks<=3).mean():.0f}%  median rank: {int(np.median(ranks))}")
    print(f"true-candidate peak:  median={np.median(true_peaks):.2f}")
    print(f"best-distractor peak: median={np.median(dist_peaks):.2f}")
    print(f"separation margin (true - best distractor): median={np.median(margins):+.3f}  "
          f"positive in {100*(np.array(margins)>0).mean():.0f}% of spans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
