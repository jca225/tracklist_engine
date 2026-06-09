"""Single-token-stream information-dynamics probe."""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import numpy as np

from eda.alignment.adaptive_markov import AdaptiveMarkovChain, MarkovTrace
from eda.alignment.artifacts import MixMertArtifact
from eda.alignment.boundaries import pick_local_peaks, pick_peaks, score_boundaries
from eda.alignment.information_summary import information_summary


PeakMode = Literal["global", "local", "both"]


def probe_token_stream(
    artifact: MixMertArtifact,
    labels: np.ndarray,
    gt_boundary_bars: tuple[int, ...] | None,
    *,
    stream_name: str,
    n_tokens: int,
    peak_percentile: float = 75.0,
    peak_min_distance: int = 4,
    local_window: int = 32,
    local_z: float = 1.5,
    tolerance_bars: int = 2,
    peak_mode: PeakMode = "both",
) -> tuple[dict, MarkovTrace]:
    chain = AdaptiveMarkovChain(int(n_tokens))
    trace = chain.run(labels)
    mir = trace.series("model_information_rate")
    surp = trace.series("surprisingness")

    peaks: dict[str, tuple[int, ...]] = {}
    scores: dict[str, dict] = {}
    if peak_mode in ("global", "both"):
        peaks["mir_global"] = pick_peaks(
            mir, min_distance=peak_min_distance, percentile=peak_percentile,
        )
        peaks["surp_global"] = pick_peaks(
            surp, min_distance=peak_min_distance, percentile=peak_percentile,
        )
    if peak_mode in ("local", "both"):
        peaks["mir_local"] = pick_local_peaks(
            mir, window=local_window, z_threshold=local_z, min_distance=peak_min_distance,
        )
        peaks["surp_local"] = pick_local_peaks(
            surp, window=local_window, z_threshold=local_z, min_distance=peak_min_distance,
        )

    if gt_boundary_bars is not None:
        for key, pred in peaks.items():
            scores[key] = asdict(
                score_boundaries(pred, gt_boundary_bars, tolerance_bars=tolerance_bars)
            )

    out: dict = {
        "stream": stream_name,
        "n_tokens": n_tokens,
        "peaks": {k: list(v) for k, v in peaks.items()},
        "scores": scores,
        "information": information_summary(artifact, trace, gt_boundary_bars),
    }
    return out, trace
