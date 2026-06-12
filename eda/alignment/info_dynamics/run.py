"""Driver: run the full M0/M1/M2 information-dynamics study on one mix.

    venvs/audio/bin/python -m eda.alignment.info_dynamics.run \
        --artifact data/analysis/1fsnxchk_mix_mert.npz \
        --gt labeling/fixtures/bb12_ground_truth.yaml \
        --out data/analysis/info_dynamics

Writes config.json, metrics.json, metrics.csv, summary.md, and per-model
overlay plots under ``<out>/plots/``.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from .baselines import run_m0, run_m1
from .data import StudyData, load_study_data
from .evaluate import (
    PeakConfig,
    best_signal_by_lift,
    best_signal_f1,
    evaluate_signalset,
    preq_nll,
)
from .plots import save_signal_overlay
from .seqmodel import run_m2
from .signals import SignalSet


def _shuffled(data: StudyData, *, seed: int) -> StudyData:
    """Temporal-shuffle control: permute frame order, keep the time grid."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.n_frames)
    return dataclasses.replace(
        data, tokens=data.tokens[perm], mert_clean=data.mert_clean[perm]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="data/analysis/1fsnxchk_mix_mert.npz")
    ap.add_argument("--gt", default="labeling/fixtures/bb12_ground_truth.yaml")
    ap.add_argument("--out", default="data/analysis/info_dynamics")
    ap.add_argument("--n-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--smooth-window", type=int, default=3)
    ap.add_argument("--min-distance-s", type=float, default=6.0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PeakConfig(
        smooth_window=args.smooth_window,
        percentile=args.percentile,
        min_distance_s=args.min_distance_s,
    )

    data = load_study_data(
        args.artifact, args.gt, n_tokens=args.n_tokens, seed=args.seed
    )
    # Common evaluable window: everything compares past M2's warm-up prefix so
    # the boundary tables are apples-to-apples across the ladder.
    eval_lo = float(data.bar_start_s[args.warmup])

    config = {
        "set_id": data.set_id,
        "artifact": args.artifact,
        "gt": args.gt,
        "n_frames": data.n_frames,
        "n_tokens": args.n_tokens,
        "seed": args.seed,
        "mert_layer": data.artifact.mert_layer,
        "mert_model": data.artifact.mert_model,
        "frame_rate_hz": round(data.n_frames / float(data.bar_end_s[-1]), 4),
        "labeled_region_s": [data.labeled_lo_s, data.labeled_hi_s],
        "n_gt_boundaries": int(len(data.gt_boundary_s)),
        "eval_lo_s": eval_lo,
        "peak_cfg": dataclasses.asdict(cfg),
        "warmup": args.warmup,
        "block": args.block,
        "tolerances_s": [3.0, 10.0],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[info] {data.set_id}: {data.n_frames} frames, {len(data.gt_boundary_s)} GT "
          f"boundaries, eval from {eval_lo:.0f}s")

    # --- run the ladder -----------------------------------------------------
    runs: list[SignalSet] = []
    print("[run] M0 memoryless …")
    runs.append(run_m0(data))
    print("[run] M1 adaptive Markov …")
    runs.append(run_m1(data))
    print("[run] M2 GRU …")
    runs.append(run_m2(data, arch="gru", warmup=args.warmup, block=args.block, seed=args.seed))
    print("[run] M2 attention (full context) …")
    runs.append(run_m2(data, arch="attention", context=None,
                       warmup=args.warmup, block=args.block, seed=args.seed,
                       label="M2-attention-ctxFull"))

    # --- context-length ablation (attention) --------------------------------
    print("[run] M2 attention context ablation …")
    ablation: list[SignalSet] = []
    for ctx in (8, 32):
        ablation.append(run_m2(
            data, arch="attention", context=ctx, warmup=args.warmup,
            block=args.block, seed=args.seed, label=f"M2-attention-ctx{ctx}",
        ))

    # --- shuffle controls (cheap, exact models) -----------------------------
    print("[run] shuffle controls (M0, M1, M2-gru) …")
    sdata = _shuffled(data, seed=args.seed)
    shuffles = [
        dataclasses.replace(run_m0(sdata), model="M0-shuffled"),
        dataclasses.replace(run_m1(sdata), model="M1-shuffled"),
        dataclasses.replace(
            run_m2(sdata, arch="gru", warmup=args.warmup, block=args.block, seed=args.seed),
            model="M2-gru-shuffled",
        ),
    ]

    # --- evaluate everything ------------------------------------------------
    metrics: dict = {"config": config, "models": {}}
    all_runs = runs + ablation + shuffles
    for ss in all_runs:
        metrics["models"][ss.model] = evaluate_signalset(
            ss, data, cfg=cfg, eval_lo_s=eval_lo
        )

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_csv(out / "metrics.csv", metrics)
    _write_summary(out / "summary.md", data, metrics, config)

    # --- plots --------------------------------------------------------------
    if not args.no_plots:
        print("[plot] overlays …")
        pdir = out / "plots"
        for ss in runs:
            save_signal_overlay(pdir / f"{ss.model}_signals.svg", ss, data, cfg=cfg)

    print(f"[done] wrote {out}/metrics.json, metrics.csv, summary.md")


def _write_csv(path: Path, metrics: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "signal", "tolerance_s", "n_peaks", "n_gt", "n_pred",
            "tp", "precision", "recall", "f1", "chance_f1", "lift",
            "prequential_nll",
        ])
        for model, mres in metrics["models"].items():
            nll = mres.get("prequential_nll", float("nan"))
            for sig, info in mres["signals"].items():
                for tol_key, sc in info["scores"].items():
                    w.writerow([
                        model, sig, sc["tolerance_s"], info["n_peaks"],
                        sc["n_gt"], sc["n_pred"], sc["tp"], sc["precision"],
                        sc["recall"], sc["f1"], sc["chance_f1"], sc["lift"],
                        round(nll, 4),
                    ])


def _sig_cell(mres: dict, sig: str, tol: str) -> str:
    sc = mres["signals"][sig]["scores"][tol]
    return f"{sc['f1']:.3f} (chance {sc['chance_f1']:.3f}, lift {sc['lift']:+.3f})"


def _write_summary(path: Path, data: StudyData, metrics: dict, config: dict) -> None:
    n_gt = metrics["models"][next(iter(metrics["models"]))]["n_gt"]
    L: list[str] = []
    L.append(f"# Information-dynamics study — {data.set_id}\n")
    L.append(f"- Frames: {config['n_frames']} bars (~{config['frame_rate_hz']} Hz), "
             f"MERT layer {config['mert_layer']}, K={config['n_tokens']} tokens")
    L.append(f"- GT boundaries scored (≥{config['eval_lo_s']:.0f}s): {n_gt} "
             f"(~1 per {(data.labeled_hi_s-config['eval_lo_s'])/max(n_gt,1):.0f}s)")
    L.append("")
    L.append("> **Read the lift, not the raw F1.** With ~1 boundary per 20 s, a "
             "±10 s window tiles most of the timeline, so random peaks already "
             "score high — the `chance_f1` columns make this explicit. The honest "
             "signal is **lift = F1 − chance**, and the ±3 s column.\n")

    L.append("## Prediction: does memory help? (prequential NLL, lower = better)\n")
    L.append("| Model | Preq NLL (nats) |")
    L.append("|---|---|")
    main = ["M0", "M1", "M2-gru", "M2-attention-ctxFull"]
    for m in main:
        if m in metrics["models"]:
            nll = metrics["models"][m].get("prequential_nll", float("nan"))
            L.append(f"| {m} | {nll:.3f} |" if np.isfinite(nll) else f"| {m} | — |")
    L.append(f"\n(uniform-token baseline = log {config['n_tokens']} = "
             f"{np.log(config['n_tokens']):.3f} nats)\n")

    L.append("## Boundary localization — best signal per model\n")
    L.append("| Model | Best signal (by ±3s lift) | F1@3s (chance/lift) | F1@10s (chance/lift) |")
    L.append("|---|---|---|---|")
    for m in main:
        if m not in metrics["models"]:
            continue
        mres = metrics["models"][m]
        bsig, _ = best_signal_by_lift(mres, tolerance_key="tol_3s")
        L.append(f"| {m} | {bsig} | {_sig_cell(mres, bsig, 'tol_3s')} | {_sig_cell(mres, bsig, 'tol_10s')} |")
    L.append("")

    L.append("## Context-length ablation (attention) — prequential NLL\n")
    L.append("| Context | Preq NLL | Best ±3s lift |")
    L.append("|---|---|---|")
    for m in ("M2-attention-ctx8", "M2-attention-ctx32", "M2-attention-ctxFull"):
        if m in metrics["models"]:
            mres = metrics["models"][m]
            _, lift = best_signal_by_lift(mres, tolerance_key="tol_3s")
            L.append(f"| {m.split('-ctx')[-1]} | {mres['prequential_nll']:.3f} | {lift:+.3f} |")
    L.append("")

    L.append("## Shuffle control — ±3s lift should collapse to ~0\n")
    L.append("| Model · signal | real F1@3s | shuffled F1@3s | real lift | shuffled lift |")
    L.append("|---|---|---|---|---|")
    pairs = [("M0", "persist_cosdist"), ("M0", "surprisal"),
             ("M1", "surprisal"), ("M1", "mir"), ("M1", "pir_proxy"),
             ("M2-gru", "surprisal")]
    for base, sig in pairs:
        sm, ss = base, f"{base}-shuffled"
        if sm in metrics["models"] and ss in metrics["models"]:
            r = metrics["models"][sm]["signals"][sig]["scores"]["tol_3s"]
            s = metrics["models"][ss]["signals"][sig]["scores"]["tol_3s"]
            L.append(f"| {base} · {sig} | {r['f1']:.3f} | {s['f1']:.3f} | "
                     f"{r['lift']:+.3f} | {s['lift']:+.3f} |")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
