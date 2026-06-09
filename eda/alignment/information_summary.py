"""Summarize information-dynamics traces — PIR 'interestingness' vs GT."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from eda.alignment.adaptive_markov import MarkovTrace
from eda.alignment.artifacts import MixMertArtifact


@dataclass(frozen=True)
class PeakMoment:
    bar_idx: int
    time_s: float
    model_information_rate: float
    surprisingness: float
    predictive_information_rate: float
    predictive_uncertainty: float


def _bar_time(artifact: MixMertArtifact, bar_idx: int) -> float:
    return float(artifact.bar_start_s[bar_idx])


def top_peaks(
    artifact: MixMertArtifact,
    trace: MarkovTrace,
    *,
    signal: str,
    n: int = 15,
    min_distance: int = 8,
) -> tuple[PeakMoment, ...]:
    series = trace.series(signal)
    order = np.argsort(series)[::-1]
    picked: list[int] = []
    for idx in order:
        i = int(idx)
        if series[i] <= 0:
            break
        if any(abs(i - p) < min_distance for p in picked):
            continue
        picked.append(i)
        if len(picked) >= n:
            break
    picked.sort()
    out: list[PeakMoment] = []
    for i in picked:
        r = trace.readouts[i]
        out.append(
            PeakMoment(
                bar_idx=i,
                time_s=_bar_time(artifact, i),
                model_information_rate=r.model_information_rate,
                surprisingness=r.surprisingness,
                predictive_information_rate=r.predictive_information_rate,
                predictive_uncertainty=r.predictive_uncertainty,
            )
        )
    return tuple(out)


def _rank_percentile(values: np.ndarray, indices: tuple[int, ...]) -> dict[str, float]:
    if values.size == 0 or not indices:
        return {"mean_percentile": float("nan"), "median_percentile": float("nan")}
    ranks = np.argsort(np.argsort(values))
    pct = [100.0 * ranks[i] / max(len(values) - 1, 1) for i in indices if 0 <= i < len(values)]
    if not pct:
        return {"mean_percentile": float("nan"), "median_percentile": float("nan")}
    return {"mean_percentile": float(np.mean(pct)), "median_percentile": float(np.median(pct))}


def information_summary(
    artifact: MixMertArtifact,
    trace: MarkovTrace,
    gt_boundary_bars: tuple[int, ...] | None = None,
) -> dict:
    """PIR / MIR stats; whether GT boundaries sit at high-information moments."""
    mir = trace.series("model_information_rate")
    surp = trace.series("surprisingness")
    pir = trace.series("predictive_information_rate")
    unc = trace.series("predictive_uncertainty")

    # Paper: inverted-U — intermediate PIR ≈ most "interesting". Measure spread.
    pir_abs = np.abs(pir)
    pir_mid = float(np.median(pir_abs))
    in_mid_band = int(np.sum((pir_abs >= 0.25 * pir_mid) & (pir_abs <= 2.5 * pir_mid)))

    summary: dict = {
        "trace_stats": {
            "mir_mean": float(np.mean(mir)),
            "mir_p90": float(np.percentile(mir, 90)),
            "surprisingness_mean": float(np.mean(surp)),
            "surprisingness_p90": float(np.percentile(surp, 90)),
            "pir_mean": float(np.mean(pir)),
            "pir_std": float(np.std(pir)),
            "pir_abs_median": pir_mid,
            "fraction_bars_mid_pir_band": in_mid_band / max(len(pir), 1),
            "uncertainty_mean": float(np.mean(unc)),
        },
        "top_mir_moments": [asdict(p) for p in top_peaks(artifact, trace, signal="mir", n=12)],
        "top_pir_moments": [asdict(p) for p in top_peaks(artifact, trace, signal="pir", n=12)],
        "top_surprisingness_moments": [
            asdict(p) for p in top_peaks(artifact, trace, signal="surprisingness", n=12)
        ],
    }

    if gt_boundary_bars:
        gt_set = tuple(sorted(set(gt_boundary_bars)))
        summary["gt_at_mir"] = _rank_percentile(mir, gt_set)
        summary["gt_at_surprisingness"] = _rank_percentile(surp, gt_set)
        summary["gt_at_pir_abs"] = _rank_percentile(pir_abs, gt_set)
        # Bars far from any GT boundary (≥16 bars) as control
        far: list[int] = []
        for i in range(len(mir)):
            if all(abs(i - g) > 16 for g in gt_set):
                far.append(i)
        summary["interior_at_mir"] = _rank_percentile(mir, tuple(far[:200]))
        summary["gt_vs_interior_mir_lift"] = (
            summary["gt_at_mir"]["mean_percentile"]
            - summary["interior_at_mir"]["mean_percentile"]
        )
        mir_p90 = float(np.percentile(mir, 90))
        summary["gt_in_top_decile_mir"] = {
            "count": sum(1 for b in gt_set if mir[b] >= mir_p90),
            "n_gt": len(gt_set),
            "rate": sum(1 for b in gt_set if mir[b] >= mir_p90) / max(len(gt_set), 1),
            "random_baseline": 0.10,
        }

    return summary
