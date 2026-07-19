"""Heatmap PNGs and contrast table for the placement ridge diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# True-diag mean at least 2× background — a *looking aid*, not a statistical claim.
# Human still reads the heatmaps; Task 5 exposes this as ``--contrast-threshold``.
DEFAULT_CONTRAST_THRESHOLD = 2.0

_CONTRAST_COLUMNS = (
    "case_id",
    "set_id",
    "slot",
    "claimed_stem",
    "span_class",
    "place_err_s",
    "channel",
    "ridge_contrast",
    "verdict_channel",
)

_GT_OVERLAY_COLOR = "#00e5ff"


def save_heatmap(
    M: np.ndarray,
    mask: np.ndarray,
    path: Path,
    *,
    title: str,
    contrast: float,
) -> None:
    """Write similarity matrix heatmap with GT diagonal contour overlay."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        M,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    if np.any(mask):
        ax.contour(
            mask.astype(np.float32),
            levels=[0.5],
            colors=_GT_OVERLAY_COLOR,
            linewidths=1.2,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Ref bin")
    ax.set_ylabel("Mix bin")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_contrast_table(
    rows: list[dict],
    path: Path,
    *,
    contrast_threshold: float = DEFAULT_CONTRAST_THRESHOLD,
) -> None:
    """Write TSV of ridge contrast per (case, channel) with verdict labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(_CONTRAST_COLUMNS),
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            contrast = float(row["ridge_contrast"])
            verdict = (
                "ridge_present" if contrast >= contrast_threshold else "ridge_absent"
            )
            out = {col: row.get(col, "") for col in _CONTRAST_COLUMNS}
            out["ridge_contrast"] = contrast
            out["verdict_channel"] = verdict
            writer.writerow(out)
