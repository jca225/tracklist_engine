"""§12 transition recovery through the REAL decoder (`path_decode.decode_path`).

`transition_probe.py` recovers a tempo ride with a *proxy* — a windowed chroma
matched-filter + a longest-non-decreasing-subsequence path constraint. That
proxy answered the scope question (gentle/medium recoverable, steep hard) but
its MAE is inflated by outliers the LNDS proxy can't catch, and it is NOT the
decoder that actually ships.

This module closes that gap: it renders the same continuous tempo ride
(`transition.render_ride`), then recovers `ref_time(mix_time)` with the
production Viterbi (`path_decode.decode_path`) — the exact piecewise-linear
segment decoder the aligner runs on real spans — and reconstructs the recovered
curve from the decoded segment list via `core.timebase.Trajectory`. The score is
therefore the *canonical* transition number, not a proxy (TRANSITION_PROBE_FINDINGS
"Next").

A smooth ride is a curve; `decode_path` approximates it as a run of short
diagonals (each window "stays" on the best local stretch, occasionally stepping
to the next). The number of decoded segments is itself informative — a good
recovery uses several forward-consistent diagonals, not one straight line and
not a shatter of spurious jumps.

Usage:
  venvs/audio/bin/python -m alignment.synthetic_mix.transition_pathdecode \\
      --ref ~/aligning/2nvzlh2k__*/mix_instrumental.flac [--win-s 12] [--mix-s 90] [--lam 0.15]
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.timebase import Trajectory
from alignment.path_decode import FPS, decode_path
from alignment.refine_ref_offsets import SR, chroma
from alignment.synthetic_mix.transition import (
    TempoRide,
    recovered_curve_error,
    render_ride,
)
from alignment.synthetic_mix.transition_probe import CURVATURES

# The stretch band handed to the Viterbi. It must bracket the ride's ratio range
# (steep is 0.82->1.18); mirror the proxy's grid so the two are comparable.
_DECODE_STRETCHES: tuple[float, ...] = (
    0.80,
    0.86,
    0.92,
    0.96,
    1.0,
    1.04,
    1.08,
    1.14,
    1.20,
)


def curve_from_segments(
    segs: list[tuple[float, float, float]],
    mix_dur_s: float,
    default_slope: float,
    step_s: float,
) -> list[tuple[float, float]]:
    """Sample a decoded segment list into (mix_t, ref_t) points.

    ``segs`` is ``decode_path``'s output — ``(mix_start_s, ref_start_s,
    ref_end_s)`` relative to span start. We build the shared piecewise-linear
    ``Trajectory`` (the same object the scorer uses) and evaluate ref at a
    uniform mix-time grid, so the error is a genuine *trajectory* error over the
    whole ride, not just at the segment anchors."""
    if not segs:
        return []
    traj = Trajectory.from_segments(
        "mix", "ref", segs, span_end=mix_dur_s, default_slope=default_slope
    )
    ts = np.arange(0.0, mix_dur_s, step_s)
    return [(float(t), float(traj.value_at(float(t)))) for t in ts]


@dataclass(frozen=True)
class DecodeResult:
    curvature: str
    decode_mae_s: float
    decode_med_s: float
    decode_p90_s: float
    n_segments: int
    score: float


def decode_curvature(
    ref: np.ndarray,
    ref_f: np.ndarray,
    name: str,
    r0: float,
    r1: float,
    mix_s: float,
    win_s: float,
    hop_s: float,
    lam: float,
) -> DecodeResult:
    """Render one ride and recover it through the production `decode_path`."""
    ride = TempoRide(r0=r0, r1=r1, mix_dur_s=mix_s, ref_start_s=0.0)
    mix = render_ride(ref, SR, ride)

    M = chroma(mix)
    wlen = int(round(win_s * FPS))
    hop = int(round(hop_s * FPS))
    segs, score = decode_path(M, ref_f, _DECODE_STRETCHES, lam, wlen, hop)

    pts = curve_from_segments(segs, mix_s, ride.mean_ratio(), hop_s)
    err = recovered_curve_error(ride, pts)
    return DecodeResult(
        curvature=name,
        decode_mae_s=err["ref_mae_s"],
        decode_med_s=err["ref_med_s"],
        decode_p90_s=err["ref_p90_s"],
        n_segments=len(segs),
        score=float(score),
    )


def _load_ref(path: str, dur_s: float) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, duration=dur_s)
    return y.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="ref stem audio (globbable)")
    ap.add_argument("--mix-s", type=float, default=90.0, help="ride/mix duration")
    ap.add_argument("--win-s", type=float, default=12.0, help="decoder window")
    ap.add_argument("--hop-s", type=float, default=2.0, help="decoder window hop")
    ap.add_argument(
        "--lam", type=float, default=0.15, help="forward-jump penalty (path_decode)"
    )
    args = ap.parse_args(argv)

    matches = glob.glob(str(Path(args.ref).expanduser()))
    if not matches:
        print(f"no ref audio matched {args.ref}", file=sys.stderr)
        return 2
    ref_path = matches[0]
    # cover the steepest ride's ref-span (mean ratio up to ~1.09) + a margin
    ref = _load_ref(ref_path, dur_s=args.mix_s * 1.3 + args.win_s)
    ref_f = chroma(ref)
    print(f"ref: {ref_path}  ({len(ref) / SR:.0f}s loaded)  lam={args.lam}\n")

    print(
        f"{'curvature':>8} | {'decode MAE':>10} | {'decode median':>13} | "
        f"{'decode p90':>10} | {'segs':>5}"
    )
    print("-" * 62)
    for name, (r0, r1) in CURVATURES.items():
        r = decode_curvature(
            ref, ref_f, name, r0, r1, args.mix_s, args.win_s, args.hop_s, args.lam
        )
        print(
            f"{r.curvature:>8} | {r.decode_mae_s:>8.2f}s | {r.decode_med_s:>11.2f}s | "
            f"{r.decode_p90_s:>8.2f}s | {r.n_segments:>5}"
        )
    print(
        "\nreading: this is the PRODUCTION path_decode Viterbi (not the LNDS proxy) "
        "recovering the ride. median is the fair number; segs = decoded diagonals "
        "(a smooth ride decodes to several forward-consistent segments). Compare to "
        "transition_probe.py's proxy row — the Viterbi should tighten MAE toward median."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
