"""Ensemble-reduction deploy gate: HuBERT vocal-probe discrimination on GT spans.

The real deploy question for ensemble reduction (p3->p2/p1) is NOT waveform SDR
but whether the HuBERT vocal probe still picks the CORRECT reference for each
acappella span. For acappella slots the reference is a pre-isolated acappella
recording (never separated), so ensemble reduction only perturbs the MIX side —
this script isolates exactly that.

Per BB12 GT acappella span: separate the mix region under each arm (p3 ref, p1),
compute HuBERT-L9 frames, score (normalized matched-filter peak) against every
candidate acappella reference, and check whether the correct reference wins and
by what margin. If p1 preserves the winner and margin vs p3 across spans -> the
probe is invariant to the reduction -> GATE PASS.

Runs on the box (has pi access + CUDA). Reads gate_job.json, writes gate_result.json.
  sudo /venv/main/bin/python workspaces/streaming_mir/ensemble_gate_discrim.py \
      --job /workspace/gate_job.json --out /workspace/gate_result.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.adapters import roformer_chain_adapter  # noqa: E402
from analysis.roformer_config import RoformerChainConfig  # noqa: E402

PI = "pi-storage"
LAYER = 9


def arms() -> dict[str, RoformerChainConfig]:
    base = RoformerChainConfig.default()  # p3 = shipped 3-pass
    bs_only = tuple(m for m in base.vocal_models if m.model_type == "bs_roformer")
    p1 = replace(base, vocal_models=bs_only, instrumental_models=bs_only)
    return {"p3": base, "p1": p1}


def pull(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-q", f"{PI}:{remote}", str(local)], check=True)


def clip(src: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            str(src),
            "-ar",
            "44100",
            "-ac",
            "2",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def hubert(path: Path) -> np.ndarray:
    import librosa

    from workspaces.alignment_prototype.refine_ref_offsets import SR
    from workspaces.section_hsmm.similarity_probe import _hubert

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return _hubert(y, LAYER)  # (768, T), L2-normed per frame


def matched_peak(win: np.ndarray, ref: np.ndarray) -> float:
    """Max normalized matched-filter score of win slid over ref (both L2-normed
    per frame already, so this is a windowed cosine peak)."""
    from scipy.signal import fftconvolve

    n = win.shape[1]
    if ref.shape[1] < n or n == 0:
        return 0.0
    w = win / (np.linalg.norm(win) + 1e-9)
    num = fftconvolve(ref, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((ref**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[n:] - e[:-n], 1e-9))
    return float((num / den).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/workspace/gate_work"))
    args = ap.parse_args()
    job = json.loads(args.job.read_text())
    args.work.mkdir(parents=True, exist_ok=True)

    # 1. Pull mix + refs, HuBERT the (separator-independent) acappella refs once.
    mix = args.work / "mix.m4a"
    if not mix.exists():
        pull(job["mix_pi_path"], mix)
    ref_feat = {}
    for rid, rpath in job["refs"].items():
        rp = args.work / "refs" / f"{rid}.m4a"
        if not rp.exists():
            pull(rpath, rp)
        ref_feat[rid] = hubert(rp)
        print(f"ref {rid}: {ref_feat[rid].shape[1]} frames", flush=True)
    rids = list(job["refs"])

    # 2. Per span: clip mix region, separate under each arm, HuBERT, score vs all refs.
    cfgs = arms()
    results = []
    for sp in job["spans"]:
        raw = args.work / "spans" / f"{sp['slot']}.m4a"
        if not raw.exists():
            clip(mix, sp["set_start"], sp["set_end"], raw)
        row = {"slot": sp["slot"], "correct": sp["rid"], "name": sp["name"], "arms": {}}
        for arm, cfg in cfgs.items():
            load_r = roformer_chain_adapter.load(cfg, device="cuda")
            if not load_r.is_ok():
                row["arms"][arm] = {"error": load_r.error.detail}
                continue
            with tempfile.TemporaryDirectory() as td:
                sr = roformer_chain_adapter.separate(load_r.value, raw, Path(td), 0)
                if not sr.is_ok():
                    row["arms"][arm] = {"error": sr.error.detail}
                    continue
                voc = next(a.path for a in sr.value.stems if a.stem_name == "vocals")
                wf = hubert(Path(voc))
            scores = {rid: matched_peak(wf, ref_feat[rid]) for rid in rids}
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            winner, top = ranked[0]
            margin = top - ranked[1][1]
            row["arms"][arm] = {
                "winner": winner,
                "correct_won": winner == sp["rid"],
                "correct_score": round(scores[sp["rid"]], 4),
                "top_score": round(top, 4),
                "margin": round(margin, 4),
            }
            print(
                f"{sp['slot']} {arm}: winner={winner} correct={winner == sp['rid']} "
                f"margin={margin:.3f}",
                flush=True,
            )
        results.append(row)

    # 3. Verdict
    n = len(results)
    p3_ok = sum(r["arms"].get("p3", {}).get("correct_won", False) for r in results)
    p1_ok = sum(r["arms"].get("p1", {}).get("correct_won", False) for r in results)
    flips = [
        r["slot"]
        for r in results
        if r["arms"].get("p3", {}).get("winner")
        != r["arms"].get("p1", {}).get("winner")
    ]
    args.out.write_text(
        json.dumps(
            {
                "results": results,
                "summary": {
                    "n": n,
                    "p3_correct": p3_ok,
                    "p1_correct": p1_ok,
                    "winner_flips": flips,
                },
            },
            indent=2,
        )
    )
    print(
        f"\n=== GATE: n={n} | p3 correct={p3_ok}/{n} | p1 correct={p1_ok}/{n} "
        f"| winner flips p3->p1: {len(flips)} {flips}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
