"""WS1 — per-curve *precision* (reliability) for matched-filter offset detection.

Neuro principle (Ernst & Banks optimal cue integration): the brain fuses cues
weighted by their **inverse variance** (precision), and collapses to a no-decision
when every cue is unreliable. The current arbiter (`harness/merge.py`) instead
fuses on a flat [0,1] confidence or a fixed `source_priority` order — neither
adapts per frame. To weight by reliability we need a per-result *precision*: how
sharply the winning offset stands out from the background of the matched-filter
score curve.

`refine_ref_offsets.correlate_window` already computes that full curve — it just
throws away everything but the argmax peak. This module mirrors its ~6-line score
kernel to RETURN the curve (read-only; no edit to the harness probe), then derives
scale-free sharpness proxies from it.

Pure functions, no I/O. See `precision_fusion_eval.py` for the BB12 read-only test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 22050
HOP = 512


def correlate_curve(wf: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """Full normalized cross-correlation curve of window ``wf`` sliding over ``rf``.

    Identical kernel to ``refine_ref_offsets.correlate_window`` but returns the
    whole ``scores`` array (frame -> cosine-normalized match) instead of only its
    argmax. Empty array when the ref is shorter than the window.
    """
    from scipy.signal import fftconvolve

    m = wf.shape[1]
    if rf.shape[1] <= m:
        return np.zeros(0, dtype=np.float32)
    w = wf / (np.linalg.norm(wf) + 1e-9)
    num = fftconvolve(rf, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((rf**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    return (num / den).astype(np.float32)


def detect_offset_curve(
    win_f: np.ndarray,
    ref_f: np.ndarray,
    stretches: tuple[float, ...],
) -> tuple[float, float, float, np.ndarray]:
    """``(ref_start_s, peak, stretch, curve)`` — like ``detect_offset`` but also
    returns the full score curve at the *winning* stretch (the one whose argmax
    peak is highest). The curve is what precision is read off of.
    """
    n = win_f.shape[1]
    best: tuple[float, float, float, np.ndarray] | None = None
    for st in stretches:
        m = int(round(n * st))
        idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
        c = correlate_curve(win_f[:, idx], ref_f)
        if c.size == 0:
            continue
        k = int(c.argmax())
        pk = float(c[k])
        if best is None or pk > best[1]:
            best = (k * HOP / SR, pk, st, c)
    if best is None:
        return (0.0, 0.0, 1.0, np.zeros(0, dtype=np.float32))
    return best


@dataclass(frozen=True)
class Precision:
    """Sharpness summary of one matched-filter curve.

    ``peak`` is the raw cosine height (NOT comparable across features — chroma
    runs high-everywhere on repetitive vocals; that's exactly why raw-peak fusion
    is unsound). The reliability signals are the *scale-free* ones:

    - ``margin``     = peak - runner-up (raw units; the [abstention-via-margin] axis)
    - ``z``          = (peak - bg_median) / bg_std    (peak's standout in σ)
    - ``prominence`` = (peak - runner-up) / bg_std    (scale-free margin; primary)

    ``prominence`` is the headline precision: it answers "how many background σ
    does the winner beat the SECOND-best offset by", which both normalizes the
    cross-feature scale gap and penalizes repeat-ambiguous curves (a real repeat
    raises the runner-up, shrinking prominence).
    """

    peak: float
    second: float
    margin: float
    bg_med: float
    bg_std: float
    z: float
    prominence: float
    n_bg: int

    @property
    def proxies(self) -> dict[str, float]:
        return {
            "peak": self.peak,
            "margin": self.margin,
            "z": self.z,
            "prominence": self.prominence,
        }


def precision_from_curve(
    curve: np.ndarray,
    *,
    sr: int = SR,
    hop: int = HOP,
    exclusion_s: float = 3.0,
) -> Precision:
    """Derive sharpness proxies from a matched-filter curve.

    The runner-up / background are computed OUTSIDE a ``±exclusion_s`` zone around
    the argmax so the peak's own shoulder isn't miscounted as a competitor.
    """
    if curve.size < 4:
        return Precision(0.0, 0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 0)
    k = int(curve.argmax())
    peak = float(curve[k])
    ex = max(1, int(exclusion_s * sr / hop))
    mask = np.ones(curve.size, dtype=bool)
    mask[max(0, k - ex) : min(curve.size, k + ex + 1)] = False
    bg = curve[mask]
    if bg.size < 2:
        return Precision(peak, peak, 0.0, peak, 1e-9, 0.0, 0.0, 0)
    second = float(bg.max())
    bg_med = float(np.median(bg))
    bg_std = float(bg.std()) + 1e-9
    margin = peak - second
    return Precision(
        peak=peak,
        second=second,
        margin=margin,
        bg_med=bg_med,
        bg_std=bg_std,
        z=(peak - bg_med) / bg_std,
        prominence=margin / bg_std,
        n_bg=int(bg.size),
    )


# --- selection helpers (the WS1 arbiter, applied per span) -------------------


def select_by(
    cands: list[tuple[str, float, Precision]],
    proxy: str,
) -> tuple[str, float, Precision] | None:
    """Pick the candidate with the highest value of ``proxy`` (peak/margin/z/
    prominence). ``cands`` = [(feature, pred_s, Precision), ...]. None if empty."""
    if not cands:
        return None
    return max(cands, key=lambda c: c[2].proxies[proxy])


if __name__ == "__main__":
    # Self-test on synthetic curves (no audio): a sharp unimodal peak must score
    # higher precision than a flat/repeat-ambiguous one.
    rng = np.random.default_rng(0)
    n = 2000
    x = np.arange(n)
    sharp = 0.1 * rng.standard_normal(n).astype(np.float32)
    sharp += 0.9 * np.exp(-((x - 800) ** 2) / (2 * 8.0**2))  # one tight bump
    flat = 0.1 * rng.standard_normal(n).astype(np.float32) + 0.5
    repeat = 0.1 * rng.standard_normal(n).astype(np.float32)
    for c in (700, 1300):  # two equal bumps -> ambiguous
        repeat += 0.8 * np.exp(-((x - c) ** 2) / (2 * 8.0**2))
    for name, c in (("sharp", sharp), ("flat", flat), ("repeat", repeat)):
        p = precision_from_curve(c)
        print(
            f"{name:7} peak={p.peak:.3f} margin={p.margin:.3f} "
            f"z={p.z:6.2f} prominence={p.prominence:6.2f}"
        )
    print("expect: sharp >> repeat ~ flat on prominence/z")
