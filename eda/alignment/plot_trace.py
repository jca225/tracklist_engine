"""Save information-dynamics trace plots (optional matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eda.alignment.adaptive_markov import MarkovTrace
from eda.alignment.artifacts import MixMertArtifact


def save_trace_plot(
    path: Path,
    artifact: MixMertArtifact,
    trace: MarkovTrace,
    *,
    gt_boundary_bars: tuple[int, ...] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    t_min = artifact.bar_start_s / 60.0
    mir = trace.series("model_information_rate")
    surp = trace.series("surprisingness")
    pir = trace.series("predictive_information_rate")
    unc = trace.series("predictive_uncertainty")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, y, title in (
        (axes[0], mir, "Model information rate (Bayesian surprise)"),
        (axes[1], surp, "Surprisingness −log P(token|context)"),
        (axes[2], pir, "Predictive information rate"),
        (axes[3], unc, "Predictive uncertainty H(X|context)"),
    ):
        ax.plot(t_min, y, linewidth=0.6, color="#2563eb")
        ax.set_ylabel(title, fontsize=8)
        ax.grid(True, alpha=0.2)
        if gt_boundary_bars:
            for b in gt_boundary_bars:
                if 0 <= b < len(t_min):
                    ax.axvline(t_min[b], color="#dc2626", alpha=0.25, linewidth=0.8)

    axes[-1].set_xlabel("Mix time (minutes)")
    fig.suptitle(f"Mix structure probe — {artifact.set_id}", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
