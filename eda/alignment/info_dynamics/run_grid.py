"""Factorial study: {full mix, acappella, instrumental} × {codebook, continuous}.

Tests the hypothesis that information dynamics is correct but was being run on
the wrong representation: a polyphonic mashup through a 24-symbol quantizer.
Each cell runs a memoryless null + a model-with-memory, and the verdict per cell
is whether the *model* surprise localizes GT song transitions **better than its
temporal-shuffle null** at ±3 s (lift over the random-peak chance floor).

    venvs/audio/bin/python -m eda.alignment.info_dynamics.run_grid

Sources are auto-detected from data/analysis/1fsnxchk_mix{,_instrumental,_vocals}_mert.npz.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .baselines import run_m0, run_m1
from .continuous import pca_whiten, run_m0_continuous, run_m2_continuous
from .data import StudyData, load_study_data
from .evaluate import PeakConfig, best_signal_by_lift, evaluate_signalset
from .seqmodel import run_m2
from .signals import SignalSet

GT = "labeling/fixtures/bb12_ground_truth.yaml"
SOURCES = {
    "full": "data/analysis/1fsnxchk_mix_mert.npz",
    "acappella": "data/analysis/1fsnxchk_mix_vocals_mert.npz",
    "instrumental": "data/analysis/1fsnxchk_mix_instrumental_mert.npz",
}
OUT = Path("data/analysis/info_dynamics_grid")
WARMUP, BLOCK, SEED, NTOK, PCA_D = 128, 32, 0, 24, 64


def _shuffle_tokens(data: StudyData) -> StudyData:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(data.n_frames)
    return dataclasses.replace(data, tokens=data.tokens[perm], mert_clean=data.mert_clean[perm])


def _shuffle_z(z: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)
    return z[rng.permutation(z.shape[0])]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PeakConfig()
    rows: list[dict] = []
    full_metrics: dict = {}

    for source, art in SOURCES.items():
        if not Path(art).is_file():
            print(f"[skip] {source}: {art} not found")
            continue
        data = load_study_data(art, GT, n_tokens=NTOK, seed=SEED)
        eval_lo = float(data.bar_start_s[WARMUP])
        z = pca_whiten(data, d=PCA_D)
        print(f"\n=== {source} ({data.n_frames} bars) ===")

        runs: dict[str, SignalSet] = {}
        # --- codebook (discrete) ---
        runs["disc-M0"] = run_m0(data)
        runs["disc-M1"] = run_m1(data)
        # --- continuous ---
        runs["cont-M0c"] = run_m0_continuous(data, z)
        runs["cont-M2c-gru"] = run_m2_continuous(
            data, z, arch="gru", warmup=WARMUP, block=BLOCK, seed=SEED)

        # --- shuffle nulls for the two model-with-memory cells ---
        sd = _shuffle_tokens(data)
        runs["disc-M1-shuf"] = dataclasses.replace(run_m1(sd), model="disc-M1-shuf")
        runs["cont-M2c-gru-shuf"] = dataclasses.replace(
            run_m2_continuous(data, _shuffle_z(z), arch="gru",
                              warmup=WARMUP, block=BLOCK, seed=SEED),
            model="cont-M2c-gru-shuf")

        evals = {k: evaluate_signalset(ss, data, cfg=cfg, eval_lo_s=eval_lo)
                 for k, ss in runs.items()}
        full_metrics[source] = {k: v for k, v in evals.items()}

        # collate: each method's best signal, its ±3s lift, and shuffle lift
        def cell(method: str, shuf: str | None) -> dict:
            bname, blift = best_signal_by_lift(evals[method], tolerance_key="tol_3s")
            sc = evals[method]["signals"][bname]["scores"]
            row = {
                "source": source, "method": method, "best_signal": bname,
                "preq_nll": round(evals[method].get("prequential_nll", float("nan")), 3),
                "f1_3s": sc["tol_3s"]["f1"], "chance_3s": sc["tol_3s"]["chance_f1"],
                "lift_3s": sc["tol_3s"]["lift"], "lift_10s": sc["tol_10s"]["lift"],
            }
            if shuf:
                _, slift = best_signal_by_lift(evals[shuf], tolerance_key="tol_3s")
                row["shuf_lift_3s"] = round(slift, 4)
                row["beats_shuffle"] = bool(row["lift_3s"] > slift + 0.02)
            return row

        rows.append(cell("disc-M0", None))
        rows.append(cell("disc-M1", "disc-M1-shuf"))
        rows.append(cell("cont-M0c", None))
        rows.append(cell("cont-M2c-gru", "cont-M2c-gru-shuf"))
        for r in rows[-4:]:
            bs = r.get("beats_shuffle", "—")
            print(f"  {r['method']:14s} {r['best_signal']:14s} nll={r['preq_nll']:6} "
                  f"lift@3s={r['lift_3s']:+.3f} (shuf {r.get('shuf_lift_3s','—')}) "
                  f"beats_shuffle={bs}")

    (OUT / "grid_metrics.json").write_text(json.dumps(full_metrics, indent=2, default=float))
    _write_grid_summary(OUT / "grid_summary.md", rows)
    print(f"\n[done] {OUT}/grid_summary.md")


def _write_grid_summary(path: Path, rows: list[dict]) -> None:
    L = ["# Information dynamics — source × representation grid (BB12)\n"]
    L.append("Verdict per model-with-memory cell: does the surprise beat its "
             "temporal-shuffle null at ±3 s (lift gap > 0.02)?\n")
    L.append("| Source | Method | Best signal | Preq NLL | F1@3s | chance | **lift@3s** | shuffle lift | beats shuffle? |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        nll = r["preq_nll"]
        nll_s = f"{nll:.3f}" if isinstance(nll, float) and np.isfinite(nll) else "—"
        bs = r.get("beats_shuffle", None)
        bs_s = ("✅" if bs else "❌") if bs is not None else "—"
        sl = r.get("shuf_lift_3s", "—")
        L.append(f"| {r['source']} | {r['method']} | {r['best_signal']} | {nll_s} | "
                 f"{r['f1_3s']:.3f} | {r['chance_3s']:.3f} | **{r['lift_3s']:+.3f}** | {sl} | {bs_s} |")
    L.append("\n*lift = F1 − random-peak floor. A genuine localizer beats its "
             "shuffle; an artifact does not. ±10s omitted (saturated at this GT density).*")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
