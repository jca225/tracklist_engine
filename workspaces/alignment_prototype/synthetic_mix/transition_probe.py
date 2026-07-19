"""§12 transition-recovery probe — can the aligner follow a continuous tempo ride?

Renders a real ref stem through a continuous tempo ride (transition.render_ride),
then recovers `ref_time(mix_time)` two ways and scores both against the known
curve:

  * CONSTANT-warp baseline: one global chroma match over the whole mix (a single
    ref_start + stretch) — the straight-line the current constant-`tempo_ratio`
    decoder assumes. Its residual on a real ramp = the curvature it structurally
    cannot represent.
  * WINDOWED recovery: slide a chroma window along the mix; each window's local
    (ref_start, stretch) match gives a (mix_t, ref_t) point. Piecewise-linear, so
    it CAN track a ramp — the question is how well, per curvature.

The gap between the two, and the windowed MAE at increasing curvature, is the
empirical Aug-1 transition-scope answer (state-of-record decision #12).

Usage:
  venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.transition_probe \\
      --ref ~/aligning/2nvzlh2k__*/mix_instrumental.flac [--win-s 12] [--mix-s 90]
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import SR, chroma, detect_offset
from workspaces.alignment_prototype.synthetic_mix.transition import (
    TempoRide,
    monotonic_filter,
    recovered_curve_error,
    render_ride,
)

# instantaneous ratio ramps, centred on 1.0, by curvature (half-width of r0->r1)
CURVATURES = {
    "flat": (1.00, 1.00),
    "gentle": (0.96, 1.04),
    "medium": (0.90, 1.10),
    "steep": (0.82, 1.18),
}
# per-window local stretch search must span the ride's ratio range
_WIN_STRETCHES = (0.80, 0.86, 0.92, 0.96, 1.0, 1.04, 1.08, 1.14, 1.20)


def load_ref(path: str, dur_s: float) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, duration=dur_s)
    return y.astype(np.float32)


def windowed_recover(
    mix: np.ndarray, ref_f: np.ndarray, win_s: float, n_windows: int
) -> list[tuple[float, float]]:
    """Slide `n_windows` chroma windows over the mix; each yields (mix_t0, ref_t)."""
    mix_dur = len(mix) / SR
    starts = np.linspace(0.0, max(0.0, mix_dur - win_s), n_windows)
    pts: list[tuple[float, float]] = []
    for t0 in starts:
        seg = mix[int(t0 * SR) : int((t0 + win_s) * SR)]
        if len(seg) < int(win_s * SR * 0.5):
            continue
        win_f = chroma(seg)
        ref_start, peak, _stretch = detect_offset(win_f, ref_f, _WIN_STRETCHES)
        if peak > 0.0:
            pts.append((float(t0), float(ref_start)))
    return pts


@dataclass(frozen=True)
class ProbeResult:
    curvature: str
    windowed_mae_s: float
    windowed_med_s: float
    constant_mae_s: float
    n_windows: int
    n_raw: int


def probe_curvature(
    ref: np.ndarray,
    ref_f: np.ndarray,
    name: str,
    r0: float,
    r1: float,
    mix_s: float,
    win_s: float,
    n_windows: int,
) -> ProbeResult:
    ride = TempoRide(r0=r0, r1=r1, mix_dur_s=mix_s, ref_start_s=0.0)
    mix = render_ride(ref, SR, ride)

    # windowed (piecewise) recovery, then a monotonic path constraint (proxy for
    # the aligner's Viterbi) that drops catastrophic matched-filter false peaks
    pts = windowed_recover(mix, ref_f, win_s, n_windows)
    mono = monotonic_filter(pts)
    err = recovered_curve_error(ride, mono)

    # constant-warp baseline: one global match -> straight line ref_start + stretch*t
    whole_f = chroma(mix)
    c_start, c_peak, c_stretch = detect_offset(whole_f, ref_f, _WIN_STRETCHES)
    const_pts = [(t, c_start + c_stretch * t) for t, _ in mono] if mono else []
    const_err = recovered_curve_error(ride, const_pts)

    return ProbeResult(
        curvature=name,
        windowed_mae_s=err["ref_mae_s"],
        windowed_med_s=err["ref_med_s"],
        constant_mae_s=const_err["ref_mae_s"],
        n_windows=int(err["n"]),
        n_raw=len(pts),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="ref stem audio (globbable)")
    ap.add_argument("--mix-s", type=float, default=90.0, help="ride/mix duration")
    ap.add_argument("--win-s", type=float, default=12.0, help="recovery window")
    ap.add_argument("--n-windows", type=int, default=15)
    args = ap.parse_args(argv)

    matches = glob.glob(str(Path(args.ref).expanduser()))
    if not matches:
        print(f"no ref audio matched {args.ref}", file=sys.stderr)
        return 2
    ref_path = matches[0]
    # need enough ref to cover the steepest ride's ref-span (mean ratio up to ~1.1)
    ref = load_ref(ref_path, dur_s=args.mix_s * 1.3 + args.win_s)
    ref_f = chroma(ref)
    print(f"ref: {ref_path}  ({len(ref) / SR:.0f}s loaded)\n")

    print(
        f"{'curvature':>8} | {'path MAE':>9} | {'path median':>11} | "
        f"{'constant MAE':>12} | {'kept':>7} | gap"
    )
    print("-" * 70)
    for name, (r0, r1) in CURVATURES.items():
        r = probe_curvature(
            ref, ref_f, name, r0, r1, args.mix_s, args.win_s, args.n_windows
        )
        gap = r.constant_mae_s - r.windowed_mae_s
        print(
            f"{r.curvature:>8} | {r.windowed_mae_s:>7.2f}s | {r.windowed_med_s:>9.2f}s | "
            f"{r.constant_mae_s:>10.2f}s | {r.n_windows:>3}/{r.n_raw:<3} | {gap:>+.1f}s"
        )
    print(
        "\nreading: path MAE/median = piecewise decoder w/ monotonic path constraint "
        "tracking the ride; constant MAE = straight-line residual (curvature the "
        "current constant-warp decoder structurally misses). kept = path-consistent "
        "windows / raw. gap>0 = windowing recovers curvature the constant decoder can't."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
