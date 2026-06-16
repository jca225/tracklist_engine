"""Generalized info-dynamics significance test for ANY set vs its tracklist cues.

Parameterized successor to `run_bb11`: scores a set's mix (full + separated stems)
against its scraped 1001tracklists cue times, with the same circular-shift
permutation p-value + bootstrap CI + Benjamini-Hochberg FDR machinery (findings
§v5/v6). Use for the cross-set replication sweep + the popularity-correlation
study (does which-cell-wins track SoundCloud plays, or is it noise?).

Inputs per set (all under data/analysis/):
  - {set_id}_tracklist_boundaries.json   list[int] cue seconds (extract on pi)
  - {set_id}_mix_mert.npz                full mix MERT (required)
  - {set_id}_mix_vocals_mert.npz         acappella stem (optional)
  - {set_id}_mix_instrumental_mert.npz   instrumental stem (optional)

    venvs/audio/bin/python -m eda.alignment.info_dynamics.run_set --set-id pwgrrb1

Writes data/analysis/info_dynamics_grid/{set_id}_robustness.md and appends a
one-row summary to data/analysis/info_dynamics_grid/cross_set_summary.tsv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import run_m1
from .continuous import pca_whiten, run_m2_continuous
from .data import study_data_from_boundaries
from .evaluate import PeakConfig, benjamini_hochberg
from .run_robustness import WARMUP, BLOCK, N_PERM, _cell

ANALYSIS = Path("data/analysis")
OUT = ANALYSIS / "info_dynamics_grid"
TOLERANCES = (10.0, 3.0)  # ±3 s discriminates; ±10 s saturates on dense GT
SOURCES = {
    "full": "{sid}_mix_mert.npz",
    "acappella": "{sid}_mix_vocals_mert.npz",
    "instrumental": "{sid}_mix_instrumental_mert.npz",
}


def _load_boundaries(set_id: str) -> np.ndarray:
    p = ANALYSIS / f"{set_id}_tracklist_boundaries.json"
    if not p.is_file():
        raise SystemExit(f"missing boundaries: {p} (extract cues on pi first)")
    return np.asarray(sorted(set(json.loads(p.read_text()))), dtype=float)


def run_set(set_id: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = PeakConfig()
    bounds = _load_boundaries(set_id)
    if bounds.size < 5:
        raise SystemExit(f"{set_id}: only {bounds.size} cues — too few to test")
    gap = float(np.median(np.diff(bounds)))
    print(f"[{set_id}] {bounds.size} cues, {bounds.min():.0f}..{bounds.max():.0f}s, "
          f"median gap {gap:.0f}s")

    cells: list[dict] = []
    for src, tmpl in SOURCES.items():
        art = ANALYSIS / tmpl.format(sid=set_id)
        if not art.is_file():
            print(f"[skip] {src}: {art.name} absent")
            continue
        data = study_data_from_boundaries(str(art), bounds, n_tokens=24, seed=0)
        lo = float(data.bar_start_s[WARMUP])
        z = pca_whiten(data, d=64)
        models = {
            "codebook": run_m1(data),
            "continuous": run_m2_continuous(
                data, z, arch="gru", warmup=WARMUP, block=BLOCK, seed=0
            ),
        }
        for rep, ss in models.items():
            for tol in TOLERANCES:
                c = _cell(ss, data, lo, cfg, src, rep, tol=tol)
                cells.append(c)
                print(f"{src:13s} {rep:10s} ±{int(tol):2d}s  F1={c['real_f1']:.3f} "
                      f"lift={c['lift']:+.3f} p={c['p_value']:.4f} z={c['z']:.1f}")

    if not cells:
        raise SystemExit(f"{set_id}: no MERT artifacts found — run GPU pipeline first")
    for c, q in zip(cells, benjamini_hochberg([c["p_value"] for c in cells])):
        c["q_value"] = float(q)

    _write(set_id, cells, bounds, gap)
    return {"set_id": set_id, "n_cues": int(bounds.size), "cells": cells}


def _verdict(c: dict) -> str:
    if c["q_value"] < 0.05:
        return "✅ q<.05"
    return f"~ p<.05 q={c['q_value']:.2f}" if c["p_value"] < 0.05 else f"❌ p={c['p_value']:.2f}"


def _write(set_id: str, cells: list[dict], bounds: np.ndarray, gap: float) -> None:
    present = sorted({c["source"] for c in cells})
    L = [
        f"# {set_id} — info-dynamics vs tracklist cues\n",
        f"{bounds.size} cue times ({bounds.min():.0f}..{bounds.max():.0f}s, median "
        f"gap {gap:.0f}s). Sources: {', '.join(present)}. Circular-shift permutation "
        f"({N_PERM}), bootstrap 95% CI, BH-FDR over {len(cells)} cells. ±3 s is the "
        "discriminating tolerance.\n",
        "| Source | repr. | tol | F1 | lift [95% CI] | p | q | z | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        L.append(
            f"| {c['source']} | {c['rep']} | ±{int(c['tol_s'])}s | {c['real_f1']:.3f} | "
            f"{c['lift']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | {c['p_value']:.4f} | "
            f"{c['q_value']:.3f} | {c['z']:.1f} | {_verdict(c)} |"
        )
    (OUT / f"{set_id}_robustness.md").write_text("\n".join(L) + "\n")

    # one-row-per-cell append to the cross-set summary (for the popularity study)
    summ = OUT / "cross_set_summary.tsv"
    header = "set_id\tn_cues\tsource\trep\ttol_s\tF1\tlift\tci_lo\tci_hi\tp\tq\n"
    lines = [] if summ.exists() else [header]
    for c in cells:
        lines.append(
            f"{set_id}\t{bounds.size}\t{c['source']}\t{c['rep']}\t{int(c['tol_s'])}\t"
            f"{c['real_f1']:.4f}\t{c['lift']:.4f}\t{c['ci_lo']:.4f}\t{c['ci_hi']:.4f}\t"
            f"{c['p_value']:.4f}\t{c['q_value']:.4f}\n"
        )
    with summ.open("a") as f:
        f.writelines(lines)
    print(f"[done] {OUT}/{set_id}_robustness.md  (+ cross_set_summary.tsv)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    run_set(p.parse_args(argv).set_id)


if __name__ == "__main__":
    main()
