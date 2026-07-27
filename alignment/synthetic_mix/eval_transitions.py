"""Aggregate transition-recovery eval over a generated ride dataset.

Turns the single-stem `transition_probe` into a statistical benchmark: for every
`ride_*/label.json` in a dataset (from `generate_transitions`), recover
ref_time(mix_time) by windowed chroma matched-filter + monotonic path constraint,
score against the known curve, and aggregate recovery by curvature across all
rides × refs. This is the robust §12 number (not just 2 hand-picked stems).

Usage:
  venvs/audio/bin/python -m alignment.synthetic_mix.eval_transitions \\
      --data data/synthetic_transitions [--win-s 12] [--n-windows 15]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from alignment.refine_ref_offsets import SR, chroma
from alignment.synthetic_mix.transition import (
    TempoRide,
    monotonic_filter,
    recovered_curve_error,
)
from alignment.synthetic_mix.transition_probe import (
    windowed_recover,
)


def _load_mono(path: str, dur_s: float | None = None) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, duration=dur_s)
    return y.astype(np.float32)


def ride_from_label(label: dict) -> TempoRide:
    return TempoRide(
        r0=float(label["r0"]),
        r1=float(label["r1"]),
        mix_dur_s=float(label["mix_dur_s"]),
        ref_start_s=float(label["ref_start_s"]),
    )


def eval_one(ride_dir: Path, win_s: float, n_windows: int) -> dict | None:
    label_p = ride_dir / "label.json"
    mix_p = ride_dir / "mix.flac"
    if not label_p.exists() or not mix_p.exists():
        return None
    label = json.loads(label_p.read_text())
    ride = ride_from_label(label)

    mix = _load_mono(str(mix_p))
    ref = _load_mono(
        label["ref_path"], dur_s=ride.mix_dur_s * max(ride.r0, ride.r1) + 5.0
    )
    ref_f = chroma(ref)

    pts = windowed_recover(mix, ref_f, win_s, n_windows)
    mono = monotonic_filter(pts)
    err = recovered_curve_error(ride, mono)
    return {
        "curvature": label["curvature"],
        "median": err["ref_med_s"],
        "mae": err["ref_mae_s"],
        "kept": err["n"],
        "raw": len(pts),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/synthetic_transitions")
    ap.add_argument("--win-s", type=float, default=12.0)
    ap.add_argument("--n-windows", type=int, default=15)
    args = ap.parse_args(argv)

    root = Path(args.data).expanduser()
    ride_dirs = sorted(root.glob("ride_*"))
    if not ride_dirs:
        print(f"no ride_* under {root}", file=sys.stderr)
        return 2

    by_curv: dict[str, list[dict]] = defaultdict(list)
    for d in ride_dirs:
        r = eval_one(d, args.win_s, args.n_windows)
        if r is not None:
            by_curv[r["curvature"]].append(r)

    total = sum(len(v) for v in by_curv.values())
    print(f"evaluated {total} rides under {root}\n")
    print(
        f"{'curvature':>8} | {'n':>3} | {'median-of-medians':>17} | "
        f"{'p75 median':>10} | {'median MAE':>10}"
    )
    print("-" * 62)
    for name in ("flat", "gentle", "medium", "steep"):
        rows = by_curv.get(name, [])
        if not rows:
            continue
        meds = sorted(r["median"] for r in rows)
        maes = sorted(r["mae"] for r in rows)
        mom = statistics.median(meds)
        p75 = meds[min(len(meds) - 1, int(len(meds) * 0.75))]
        med_mae = statistics.median(maes)
        print(
            f"{name:>8} | {len(rows):>3} | {mom:>15.2f}s | "
            f"{p75:>8.2f}s | {med_mae:>8.2f}s"
        )
    print(
        "\nmedian-of-medians = typical per-ride recovery; the headline transition "
        "number. p75 = the harder quartile. See TRANSITION_PROBE_FINDINGS.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
