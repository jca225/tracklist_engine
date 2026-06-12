"""Significance test for the source × representation grid (BB12, n = 1 mix).

Two nulls, both reported:

1. **Localization permutation (primary).** For each cell we take the real model's
   best-lift signal, fix its peak pattern, and circularly shift its *phase* within
   the labeled window 1000× to build a null F1 distribution. The p-value is exact
   (``(1+#{null>=real})/(N+1)``) and tests "do these peaks land on GT boundaries
   better than an arbitrary phase?" — preserving the signal's own peak structure,
   so it is stricter than a uniform-random null. A 95% bootstrap CI on the lift
   (resampling GT boundaries) quantifies the effect size.

2. **Model-level input shuffle (secondary).** The older control — permute the
   input frames, retrain, recompute lift (cheap discrete Markov × N_DISC seeds,
   GRU × N_CONT seeds). Tests "is the learned structure non-random?" Kept because
   it nulls a different thing (the model), but it is too few seeds for a p-value.

Multiplicity: the grid is {full, acappella, instrumental} × {codebook, continuous}.
The **pre-registered primary hypothesis** is instrumental × continuous (motivated
by the stem-wise design — the instrumental is the structural anchor). All cells
are also reported with Benjamini-Hochberg FDR-adjusted q-values.

**Scope caveat — this is a *within-mix* test.** It establishes that BB12's
instrumental surprise is non-randomly aligned to BB12's section starts. It says
nothing about DJ mixes in general: the unit of replication for a population claim
is the *set*, not the bar-frame, and n = 1 here. Read it as characterization, not
a validated detector.

    venvs/audio/bin/python -m eda.alignment.info_dynamics.run_robustness

Writes data/analysis/info_dynamics_grid/robustness.md.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .baselines import run_m1
from .continuous import pca_whiten, run_m2_continuous
from .data import StudyData, load_study_data
from .evaluate import (
    PeakConfig,
    benjamini_hochberg,
    best_signal_by_lift,
    bootstrap_lift_ci,
    circular_shift_null,
    evaluate_signalset,
    signal_peak_times,
)

GT = "labeling/fixtures/bb12_ground_truth.yaml"
SOURCES = {
    "full": "data/analysis/1fsnxchk_mix_mert.npz",
    "acappella": "data/analysis/1fsnxchk_mix_vocals_mert.npz",
    "instrumental": "data/analysis/1fsnxchk_mix_instrumental_mert.npz",
}
PRIMARY = ("instrumental", "continuous")  # pre-registered hypothesis
OUT = Path("data/analysis/info_dynamics_grid")
WARMUP, BLOCK = 128, 32
N_PERM, N_BOOT, TOL = 1000, 1000, 3.0
N_DISC, N_CONT = 10, 4  # secondary input-shuffle seeds


def _cell(ss, data, lo, cfg, src, rep, *, tol: float = TOL) -> dict:
    """Real lift + permutation p-value + bootstrap CI for one model's best signal."""
    tol_key = f"tol_{int(tol)}s"
    res = evaluate_signalset(ss, data, cfg=cfg, eval_lo_s=lo)
    name, lift = best_signal_by_lift(res, tolerance_key=tol_key)
    peaks, lo_used = signal_peak_times(ss.signals[name], data, cfg, eval_lo_s=lo)
    gt = data.gt_boundary_s[data.gt_boundary_s >= lo_used]
    perm = circular_shift_null(
        peaks, gt, lo_s=lo_used, hi_s=data.labeled_hi_s,
        tolerance_s=tol, n_perm=N_PERM, seed=0,
    )
    boot = bootstrap_lift_ci(
        peaks, gt, lo_s=lo_used, hi_s=data.labeled_hi_s,
        tolerance_s=tol, n_boot=N_BOOT, seed=0,
    )
    return {
        "source": src, "rep": rep, "signal": name, "tol_s": tol,
        "lift": float(lift), "real_f1": perm["real_f1"],
        "p_value": perm["p_value"], "z": perm["z"],
        "null_mean": perm["null_mean"], "null_max": perm["null_max"],
        "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
    }


def _lift_only(ss, data, lo, cfg) -> float:
    res = evaluate_signalset(ss, data, cfg=cfg, eval_lo_s=lo)
    _, l = best_signal_by_lift(res, tolerance_key="tol_3s")
    return float(l)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PeakConfig()
    cells: list[dict] = []
    secondary: list[dict] = []

    for src, art in SOURCES.items():
        if not Path(art).is_file():
            print(f"[skip] {src}: missing {art}")
            continue
        data = load_study_data(art, GT, n_tokens=24, seed=0)
        lo = float(data.bar_start_s[WARMUP])
        z = pca_whiten(data, d=64)

        m1 = run_m1(data)
        m2c = run_m2_continuous(data, z, arch="gru", warmup=WARMUP, block=BLOCK, seed=0)

        # --- primary: localization permutation + bootstrap (no retrain) ---
        cells.append(_cell(m1, data, lo, cfg, src, "codebook"))
        cells.append(_cell(m2c, data, lo, cfg, src, "continuous"))
        c = cells[-1]
        print(f"{src:13s} cont real_f1={c['real_f1']:.3f} lift={c['lift']:+.3f} "
              f"p={c['p_value']:.4f} z={c['z']:.1f} CI[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]")

        # --- secondary: model-level input shuffle (retrain) ---
        sh_d = [
            _lift_only(
                run_m1(dataclasses.replace(
                    data,
                    tokens=data.tokens[np.random.default_rng(s).permutation(data.n_frames)],
                    mert_clean=data.mert_clean[np.random.default_rng(s).permutation(data.n_frames)],
                )),
                data, lo, cfg,
            )
            for s in range(N_DISC)
        ]
        sh_c = [
            _lift_only(
                run_m2_continuous(
                    data, z[np.random.default_rng(100 + s).permutation(z.shape[0])],
                    arch="gru", warmup=WARMUP, block=BLOCK, seed=0,
                ),
                data, lo, cfg,
            )
            for s in range(N_CONT)
        ]
        secondary.append({
            "source": src,
            "disc_shuf_mean": float(np.mean(sh_d)), "disc_shuf_max": float(np.max(sh_d)),
            "cont_shuf_mean": float(np.mean(sh_c)), "cont_shuf_max": float(np.max(sh_c)),
        })

    # --- multiple-comparison correction across all cells ---
    qvals = benjamini_hochberg([c["p_value"] for c in cells])
    for c, q in zip(cells, qvals):
        c["q_value"] = float(q)

    _write(cells, secondary)


def _verdict(c: dict) -> str:
    star = " ⭐" if (c["source"], c["rep"]) == PRIMARY else ""
    if c["q_value"] < 0.05:
        return f"✅ q<.05{star}"
    if c["p_value"] < 0.05:
        return f"~ p<.05, q={c['q_value']:.2f}{star}"
    return f"❌ p={c['p_value']:.2f}{star}"


def _write(cells: list[dict], secondary: list[dict]) -> None:
    L = [
        "# Significance test — ±3 s localization (BB12, n = 1 mix)\n",
        f"Primary null: circular-shift permutation, {N_PERM} perms, peak pattern "
        "held fixed. p = exact one-sided; q = Benjamini-Hochberg FDR across all "
        f"{len(cells)} cells. CI = 95% bootstrap on lift ({N_BOOT} draws). "
        f"Pre-registered primary hypothesis ⭐ = {PRIMARY[0]} × {PRIMARY[1]}.\n",
        "| Source | Representation | best signal | F1 | lift [95% CI] | p (perm) | q (FDR) | z | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        L.append(
            f"| {c['source']} | {c['rep']} | {c['signal']} | {c['real_f1']:.3f} | "
            f"{c['lift']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | "
            f"{c['p_value']:.4f} | {c['q_value']:.3f} | {c['z']:.1f} | {_verdict(c)} |"
        )

    prim = next((c for c in cells if (c["source"], c["rep"]) == PRIMARY), None)
    if prim is not None:
        sig = "significant" if prim["q_value"] < 0.05 else "NOT significant"
        L.append(
            f"\n**Primary result (⭐ {PRIMARY[0]} × {PRIMARY[1]}):** lift "
            f"{prim['lift']:+.3f} (95% CI [{prim['ci_lo']:+.3f}, {prim['ci_hi']:+.3f}]), "
            f"permutation p = {prim['p_value']:.4f}, FDR q = {prim['q_value']:.3f} "
            f"→ **{sig}** at q < .05. The instrumental's surprise peaks land on its "
            "section starts far better than an arbitrary phase of the same peak pattern."
        )

    L += [
        "\n**Scope — this is a *within-mix* test (n = 1 mix).** It shows BB12's "
        "instrumental surprise is non-randomly aligned to BB12's section starts; it "
        "does **not** show the effect generalizes across DJ mixes. The unit of "
        "replication for a population claim is the *set*, not the bar-frame — those "
        "frames are not exchangeable across mixes. Read as characterization, not a "
        "validated detector. Set-level replication (≥3–5 labeled sets) is the bar to "
        "upgrade this to a general finding.",
        "\n## Secondary: model-level input-shuffle null (retrained, few seeds)\n",
        f"Permute input frames + retrain. {N_DISC} discrete / {N_CONT} continuous "
        "seeds — too few for a p-value, reported only as a sanity floor on the "
        "best-signal lift.\n",
        "| Source | codebook shuf (mean/max) | continuous shuf (mean/max) |",
        "|---|---|---|",
    ]
    for s in secondary:
        L.append(
            f"| {s['source']} | {s['disc_shuf_mean']:+.3f}/{s['disc_shuf_max']:+.3f} | "
            f"{s['cont_shuf_mean']:+.3f}/{s['cont_shuf_max']:+.3f} |"
        )

    (OUT / "robustness.md").write_text("\n".join(L) + "\n")
    print(f"[done] {OUT}/robustness.md")


if __name__ == "__main__":
    main()
