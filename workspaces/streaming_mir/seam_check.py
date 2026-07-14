"""WS2 real-set seam validation: score two set-stem renders against a
shifted-grid pseudo-reference.

A 60-90 min set has no full-file offline reference (VRAM-impossible), so the
reference is a THIRD render whose chunk grid is offset by half a chunk
(render_set_stems.py --grid-offset-sec): its chunk interiors span the other
renders' join points, so near a join it plays the role full-file offline
played in block_overlap_sweep.py. Metric: SDR restricted to ±win_sec of each
join (boundary-local), with chunk-midpoint windows as the interior control.

Success criterion (spec): render B (overlap 10) boundary SDR within 0.5 dB of
its interior control; render A (overlap 0) shows the seam gap.

sdr() is copied from block_overlap_sweep.py rather than imported — importing
that module drags the separator adapters in; this script needs only
numpy/soundfile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "instrumental")


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Global SDR in dB. Trims to common length; flattens channels."""
    n = min(len(ref), len(est))
    r = ref[:n].reshape(-1).astype(np.float64)
    e = est[:n].reshape(-1).astype(np.float64)
    num = float(np.sum(r * r)) + 1e-12
    den = float(np.sum((r - e) ** 2)) + 1e-12
    return 10.0 * np.log10(num / den)


def sdr_windows(
    ref: np.ndarray, est: np.ndarray, centers: list[int], half_win: int
) -> float:
    """SDR restricted to ±half_win samples around each center."""
    n = min(len(ref), len(est))
    mask = np.zeros(n, dtype=bool)
    for c in centers:
        if 0 <= c < n:
            mask[max(0, c - half_win) : min(n, c + half_win)] = True
    if not mask.any():
        return float("nan")
    return sdr(ref[:n][mask], est[:n][mask])


def per_join_sdr(
    ref: np.ndarray, est: np.ndarray, centers: list[int], half_win: int
) -> list[tuple[int, float]]:
    return [(c, sdr_windows(ref, est, [c], half_win)) for c in centers]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-a", type=Path, required=True, help="overlap-0 render dir")
    ap.add_argument(
        "--render-b", type=Path, required=True, help="overlap-10 render dir"
    )
    ap.add_argument(
        "--pseudo-ref", type=Path, required=True, help="grid-offset render dir"
    )
    ap.add_argument("--chunk-sec", type=float, default=360.0)
    ap.add_argument("--win-sec", type=float, default=2.0)
    ap.add_argument(
        "--snippets-out",
        type=Path,
        default=None,
        help="export 10s wavs around each render's worst join for ear checks",
    )
    args = ap.parse_args()

    for stem in STEMS:
        ref, sr = sf.read(args.pseudo_ref / f"{stem}.flac", always_2d=True)
        a, sr_a = sf.read(args.render_a / f"{stem}.flac", always_2d=True)
        b, sr_b = sf.read(args.render_b / f"{stem}.flac", always_2d=True)
        assert sr == sr_a == sr_b, f"sample-rate mismatch on {stem}"

        hop = int(args.chunk_sec * sr)
        n = min(len(ref), len(a), len(b))
        joins = list(range(hop, n, hop))
        mids = [j - hop // 2 for j in joins]  # interior control windows
        half = int(args.win_sec * sr)

        print(f"\n== {stem} ({len(joins)} joins, ±{args.win_sec}s windows) ==")
        for name, est in (("A(ovl=0)", a), ("B(ovl=10)", b)):
            j_sdr = sdr_windows(ref, est, joins, half)
            i_sdr = sdr_windows(ref, est, mids, half)
            print(
                f"  {name:10s} join={j_sdr:7.2f} dB  interior={i_sdr:7.2f} dB  "
                f"gap={i_sdr - j_sdr:+.2f} dB"
            )
            per = per_join_sdr(ref, est, joins, half)
            worst = min(per, key=lambda p: p[1])
            print(
                f"             worst join @ {worst[0] / sr:7.1f}s = {worst[1]:.2f} dB"
            )
            if args.snippets_out is not None:
                args.snippets_out.mkdir(parents=True, exist_ok=True)
                c = worst[0]
                snip = est[max(0, c - 5 * sr) : c + 5 * sr]
                sf.write(
                    args.snippets_out / f"{stem}_{name.split('(')[0]}_worst_join.wav",
                    snip,
                    sr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
