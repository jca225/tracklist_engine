"""Evaluation: second-based boundary scoring, prequential NLL, shuffle control.

Peaks are scored in *seconds* (the spec's ±3 s / ±10 s windows) rather than in
bars, because DJ transitions are gradual and the bar grid is non-uniform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import StudyData
from .signals import SignalSet, smooth


@dataclass(frozen=True)
class PeakConfig:
    smooth_window: int = 3        # bars (~5 s) — visualization + detection smoothing
    percentile: float = 90.0      # global threshold
    min_distance_s: float = 6.0   # min inter-peak spacing in seconds


@dataclass(frozen=True)
class BoundaryScore:
    tolerance_s: float
    n_gt: int
    n_pred: int
    tp: int
    precision: float
    recall: float
    f1: float


def _finite_floor(signal: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with the minimum finite value (never a peak)."""
    s = np.asarray(signal, dtype=np.float64).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.full_like(s, 0.0)
    floor = float(s[finite].min())
    s[~finite] = floor - 1.0
    return s


def pick_peaks_seconds(
    signal: np.ndarray,
    bar_start_s: np.ndarray,
    cfg: PeakConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima above a global percentile, min-spaced in seconds.

    Returns (peak_frame_indices, peak_times_s).
    """
    sm = smooth(signal, window=cfg.smooth_window)
    valid = np.isfinite(sm)
    if not valid.any():
        return np.array([], dtype=int), np.array([], dtype=float)
    s = _finite_floor(sm)
    thresh = float(np.percentile(sm[valid], cfg.percentile))

    median_bar = float(np.median(np.diff(bar_start_s))) if len(bar_start_s) > 1 else 1.8
    min_dist = max(1, int(round(cfg.min_distance_s / max(median_bar, 1e-6))))

    peaks: list[int] = []
    for i in range(1, len(s) - 1):
        if s[i] < thresh:
            continue
        if s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            if peaks and i - peaks[-1] < min_dist:
                if s[i] > s[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    idx = np.asarray(peaks, dtype=int)
    return idx, bar_start_s[idx] if idx.size else np.array([], dtype=float)


def _greedy_match(
    pred: np.ndarray, gt: np.ndarray, tolerance_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching → (pred_is_tp mask, gt_is_hit mask).

    ``pred`` and ``gt`` must be pre-sorted ascending. A pred peak claims the
    nearest still-free GT boundary within ±tolerance.
    """
    pred_tp = np.zeros(pred.shape[0], dtype=bool)
    gt_hit = np.zeros(gt.shape[0], dtype=bool)
    if gt.size == 0 or pred.size == 0:
        return pred_tp, gt_hit
    for pi, p in enumerate(pred):
        best_j, best_d = -1, tolerance_s + 1e-9
        for j, g in enumerate(gt):
            if gt_hit[j]:
                continue
            d = abs(p - g)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            gt_hit[best_j] = True
            pred_tp[pi] = True
    return pred_tp, gt_hit


def _f1_from_counts(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def score_seconds(
    pred_times_s: np.ndarray,
    gt_times_s: np.ndarray,
    *,
    tolerance_s: float,
) -> BoundaryScore:
    """Greedy one-to-one matching within ±tolerance (seconds)."""
    pred = np.sort(np.asarray(pred_times_s, dtype=float))
    gt = np.sort(np.asarray(gt_times_s, dtype=float))
    if gt.size == 0:
        return BoundaryScore(tolerance_s, 0, len(pred), 0, 0.0, 0.0, 0.0)
    pred_tp, _ = _greedy_match(pred, gt, tolerance_s)
    tp = int(pred_tp.sum())
    precision, recall, f1 = _f1_from_counts(tp, len(pred), len(gt))
    return BoundaryScore(tolerance_s, len(gt), len(pred), tp, precision, recall, f1)


def random_chance_f1(
    n_pred: int,
    gt_times_s: np.ndarray,
    *,
    lo_s: float,
    hi_s: float,
    tolerance_s: float,
    n_trials: int = 40,
    seed: int = 0,
) -> float:
    """Mean F1 of ``n_pred`` peaks placed uniformly at random in [lo, hi].

    This is the honest chance floor. With dense GT and a wide tolerance the
    windows tile the timeline, so this floor approaches the observed F1 — which
    is exactly how we detect that a tolerance is saturated/uninformative.
    """
    if n_pred <= 0 or gt_times_s.size == 0 or hi_s <= lo_s:
        return 0.0
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(n_trials):
        peaks = rng.uniform(lo_s, hi_s, size=n_pred)
        f1s.append(score_seconds(peaks, gt_times_s, tolerance_s=tolerance_s).f1)
    return float(np.mean(f1s))


def restrict_to_labeled(
    peak_times_s: np.ndarray, data: StudyData, *, pad_s: float = 0.0
) -> np.ndarray:
    lo, hi = data.labeled_lo_s - pad_s, data.labeled_hi_s + pad_s
    return peak_times_s[(peak_times_s >= lo) & (peak_times_s <= hi)]


def signal_peak_times(
    signal: np.ndarray, data: StudyData, cfg: PeakConfig, *, eval_lo_s: float | None = None
) -> tuple[np.ndarray, float]:
    """Peak times of one signal, restricted to the scored window [lo, hi]."""
    lo = data.labeled_lo_s if eval_lo_s is None else max(eval_lo_s, data.labeled_lo_s)
    _, times = pick_peaks_seconds(signal, data.bar_start_s, cfg)
    times = restrict_to_labeled(times, data)
    return times[times >= lo], lo


def circular_shift_null(
    pred_times_s: np.ndarray,
    gt_times_s: np.ndarray,
    *,
    lo_s: float,
    hi_s: float,
    tolerance_s: float,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Permutation p-value for peak->GT alignment via circular phase shift.

    Holds the predicted peak *pattern* fixed (its count and circular spacing) and
    randomizes only its absolute phase within [lo, hi]. Tests H0 — "these peaks
    carry no information about *where* boundaries fall" — against H1 — "they land
    on GT boundaries better than an arbitrary phase would." Because it reuses the
    signal's own peak structure (clustering, spacing), it is a stricter, more
    honest null than ``random_chance_f1``'s uniform-random peaks, which assume an
    unstructured detector.

    Returns the real F1, the null-distribution summary, a one-sided p-value
    ``p = (1 + #{null >= real}) / (n_perm + 1)`` (add-one so p is never 0), and a
    descriptive z = (real - null_mean) / null_std.
    """
    pt = np.sort(np.asarray(pred_times_s, dtype=float))
    pt = pt[(pt >= lo_s) & (pt <= hi_s)]
    real_f1 = score_seconds(pt, gt_times_s, tolerance_s=tolerance_s).f1
    width = hi_s - lo_s
    if pt.size == 0 or np.asarray(gt_times_s).size == 0 or width <= 0:
        return {"real_f1": float(real_f1), "null_mean": 0.0, "null_std": 0.0,
                "null_max": 0.0, "p_value": 1.0, "z": 0.0, "n_ge": n_perm,
                "n_perm": n_perm}
    rng = np.random.default_rng(seed)
    rel = pt - lo_s
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        off = rng.uniform(0.0, width)
        shifted = lo_s + np.mod(rel + off, width)
        null[i] = score_seconds(shifted, gt_times_s, tolerance_s=tolerance_s).f1
    n_ge = int(np.sum(null >= real_f1))
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    return {
        "real_f1": float(real_f1),
        "null_mean": mu,
        "null_std": sd,
        "null_max": float(null.max()),
        "p_value": float((1 + n_ge) / (n_perm + 1)),
        "z": float((real_f1 - mu) / sd) if sd > 0 else 0.0,
        "n_ge": n_ge,
        "n_perm": n_perm,
    }


def bootstrap_lift_ci(
    pred_times_s: np.ndarray,
    gt_times_s: np.ndarray,
    *,
    lo_s: float,
    hi_s: float,
    tolerance_s: float,
    n_boot: int = 1000,
    n_chance_trials: int = 40,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile-bootstrap CI for lift = F1 - random-peak chance.

    Resamples the per-event match outcomes — the ``n_pred`` peak TP/FP flags and
    the ``n_gt`` boundary hit/miss flags — with replacement, and recomputes F1
    each draw (chance floor held fixed; it is a deterministic property of GT
    density). Resampling the *outcomes* rather than the GT *timestamps* avoids
    the duplicate-timestamp bias that a naive time-resample injects into the
    one-to-one matcher, so the point estimate sits inside the CI. The CI reflects
    sampling variance in which boundaries the peaks happen to hit — the dominant
    uncertainty term at n = 1 mix.
    """
    pt = np.sort(np.asarray(pred_times_s, dtype=float))
    pt = pt[(pt >= lo_s) & (pt <= hi_s)]
    gt = np.sort(np.asarray(gt_times_s, dtype=float))
    if pt.size == 0 or gt.size == 0:
        return {"lift": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_boot": n_boot}
    chance = random_chance_f1(
        len(pt), gt, lo_s=lo_s, hi_s=hi_s, tolerance_s=tolerance_s,
        n_trials=n_chance_trials, seed=seed,
    )
    pred_tp, gt_hit = _greedy_match(pt, gt, tolerance_s)
    point = _f1_from_counts(int(pred_tp.sum()), pt.size, gt.size)[2] - chance
    rng = np.random.default_rng(seed)
    n_pred, n_gt = pt.size, gt.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # tp must be consistent across the two views; resample the matched pairs.
        p_tp = int(pred_tp[rng.integers(0, n_pred, size=n_pred)].sum())
        g_hit = int(gt_hit[rng.integers(0, n_gt, size=n_gt)].sum())
        # precision from resampled peaks, recall from resampled boundaries.
        precision = p_tp / n_pred
        recall = g_hit / n_gt
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        boots[i] = f1 - chance
    a = (1.0 - ci) / 2.0
    return {
        "lift": float(point),
        "ci_lo": float(np.quantile(boots, a)),
        "ci_hi": float(np.quantile(boots, 1.0 - a)),
        "n_boot": n_boot,
    }


def benjamini_hochberg(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values), monotone, clipped."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def evaluate_signalset(
    sigset: SignalSet,
    data: StudyData,
    *,
    cfg: PeakConfig,
    tolerances_s: tuple[float, ...] = (3.0, 10.0),
    eval_lo_s: float | None = None,
) -> dict:
    """Per-signal boundary scores at each tolerance + prequential NLL.

    ``eval_lo_s`` raises the lower scoring bound (e.g. past an M2 warm-up
    prefix) so models with different valid regions compare on identical ground;
    both predicted peaks and GT boundaries below it are dropped.
    """
    lo = data.labeled_lo_s if eval_lo_s is None else max(eval_lo_s, data.labeled_lo_s)
    gt = data.gt_boundary_s[data.gt_boundary_s >= lo]
    out: dict = {"model": sigset.model, "eval_lo_s": lo, "n_gt": int(len(gt)), "signals": {}}
    mask = data.labeled_frame_mask()
    for name, sig in sigset.signals.items():
        frames, times = pick_peaks_seconds(sig, data.bar_start_s, cfg)
        times = restrict_to_labeled(times, data)
        times = times[times >= lo]
        scores = {}
        for tol in tolerances_s:
            sc = _score_to_dict(score_seconds(times, gt, tolerance_s=tol))
            chance = random_chance_f1(
                len(times), gt, lo_s=lo, hi_s=data.labeled_hi_s, tolerance_s=tol
            )
            sc["chance_f1"] = round(chance, 4)
            sc["lift"] = round(sc["f1"] - chance, 4)
            scores[f"tol_{int(tol)}s"] = sc
        # Mean surprise of the signal restricted to labeled frames (info density).
        valid = np.isfinite(sig) & mask
        out["signals"][name] = {
            "n_peaks": int(len(times)),
            "mean_labeled": float(np.nanmean(sig[valid])) if valid.any() else float("nan"),
            "scores": scores,
        }
    out["prequential_nll"] = preq_nll(sigset, data)
    return out


def preq_nll(sigset: SignalSet, data: StudyData) -> float:
    """Mean prequential surprisal (nats) over labeled, valid frames."""
    if "surprisal" not in sigset.signals:
        return float("nan")
    s = sigset.signals["surprisal"]
    mask = data.labeled_frame_mask() & np.isfinite(s)
    return float(np.mean(s[mask])) if mask.any() else float("nan")


def _score_to_dict(s: BoundaryScore) -> dict:
    return {
        "tolerance_s": s.tolerance_s,
        "n_gt": s.n_gt,
        "n_pred": s.n_pred,
        "tp": s.tp,
        "precision": round(s.precision, 4),
        "recall": round(s.recall, 4),
        "f1": round(s.f1, 4),
    }


def best_signal_f1(eval_result: dict, *, tolerance_key: str = "tol_3s") -> tuple[str, float]:
    """Name + F1 of the best signal for a model at a given tolerance."""
    best_name, best_f1 = "", -1.0
    for name, info in eval_result["signals"].items():
        f1 = info["scores"][tolerance_key]["f1"]
        if f1 > best_f1:
            best_name, best_f1 = name, f1
    return best_name, best_f1


def best_signal_by_lift(eval_result: dict, *, tolerance_key: str = "tol_3s") -> tuple[str, float]:
    """Name + lift-over-chance of the best signal at a given tolerance.

    Lift (F1 minus the random-peak floor) is the honest ranking key — raw F1 is
    inflated by GT density at wide tolerances.
    """
    best_name, best_lift = "", -1e9
    for name, info in eval_result["signals"].items():
        lift = info["scores"][tolerance_key].get("lift", -1e9)
        if lift > best_lift:
            best_name, best_lift = name, lift
    return best_name, best_lift
