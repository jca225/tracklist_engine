"""Generate ride-augmented synthetic spans — continuous-tempo training data.

Operationalizes the §12 transition finding (TRANSITION_PROBE_FINDINGS.md): rides
are recoverable, so the learned decoder should TRAIN on them. This renders real
ref audio through continuous tempo rides (transition.render_ride) and writes, per
sample, `mix.flac` + `label.json` (the true ref_time(mix_time) curve + params).

Curvature is sampled weighted toward the gentle/medium range the probe showed
recoverable (steep is rarer, and abstain-eligible). Self-contained: depends only
on transition.py + a glob of ref audio (defaults to the local aligning
instrumental stems) — no scenario_v2 entanglement.

Usage:
  venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate_transitions \\
      --n 20 --out data/synthetic_transitions --seed 0
  # custom ref pool:
  ... --refs "~/aligning/*/mix_instrumental.flac"
  # summarize a generated set:
  ... --list data/synthetic_transitions
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from workspaces.alignment_prototype.synthetic_mix.transition import (
    TempoRide,
    render_ride,
)

SR = 22050
DEFAULT_REFS = "~/aligning/*/mix_instrumental.flac"
# (name, half-width, weight) — ride ratio = 1 +/- half-width; gentle/medium heavy.
CURVATURE_MIX = (
    ("gentle", 0.04, 0.45),
    ("medium", 0.10, 0.40),
    ("steep", 0.18, 0.15),
)


@dataclass(frozen=True)
class RideSpec:
    ref_path: str
    curvature: str
    r0: float
    r1: float
    mix_dur_s: float
    ref_start_s: float
    direction: str  # "accel" (r rises) or "decel"


def sample_ride(
    rng: np.random.Generator, ref_dur_s: float, mix_dur_s: float
) -> tuple[str, float, float, float, str]:
    """(curvature, r0, r1, ref_start_s, direction) — a random recoverable-ish ride."""
    names = [c[0] for c in CURVATURE_MIX]
    weights = np.array([c[2] for c in CURVATURE_MIX])
    idx = int(rng.choice(len(names), p=weights / weights.sum()))
    name, half = CURVATURE_MIX[idx][0], CURVATURE_MIX[idx][1]
    accel = rng.random() < 0.5
    r0, r1 = (1 - half, 1 + half) if accel else (1 + half, 1 - half)
    # ref_start leaves room for the ride's ref-span (mean ratio ~1) within the ref
    max_start = max(0.0, ref_dur_s - mix_dur_s * max(r0, r1) - 1.0)
    ref_start = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
    return name, float(r0), float(r1), ref_start, ("accel" if accel else "decel")


def build_specs(
    ref_paths: list[str], n: int, mix_dur_s: float, seed: int
) -> list[RideSpec]:
    """Pure: n RideSpecs over the ref pool (round-robin refs, seeded rides)."""
    rng = np.random.default_rng(seed)
    specs: list[RideSpec] = []
    for i in range(n):
        ref = ref_paths[i % len(ref_paths)]
        # ref duration is looked up at render time; assume >= 120 s here for spec,
        # clamp ref_start at render if the file is shorter.
        name, r0, r1, ref_start, direction = sample_ride(rng, 300.0, mix_dur_s)
        specs.append(
            RideSpec(ref, name, r0, r1, mix_dur_s, round(ref_start, 2), direction)
        )
    return specs


def _load(path: str, dur_s: float) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True, duration=dur_s)
    return y.astype(np.float32)


def render_spec(spec: RideSpec, out_dir: Path) -> dict:
    import soundfile as sf

    ref = _load(spec.ref_path, dur_s=spec.mix_dur_s * max(spec.r0, spec.r1) + 5.0)
    ref_dur = len(ref) / SR
    ref_start = min(spec.ref_start_s, max(0.0, ref_dur - spec.mix_dur_s))
    ride = TempoRide(
        r0=spec.r0, r1=spec.r1, mix_dur_s=spec.mix_dur_s, ref_start_s=ref_start
    )
    mix = render_ride(ref, SR, ride)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "mix.flac", mix, SR)
    label = {
        **asdict(spec),
        "ref_start_s": round(ref_start, 3),
        "ref_span_s": round(ride.ref_span_s, 3),
        "mean_ratio": round(ride.mean_ratio(), 4),
        "sr": SR,
        "curve": [[round(t, 3), round(r, 3)] for t, r in ride.label_points(33)],
    }
    (out_dir / "label.json").write_text(json.dumps(label, indent=2))
    return label


def summarize(root: Path) -> None:
    labels = sorted(root.glob("ride_*/label.json"))
    if not labels:
        print(f"no ride_*/label.json under {root}")
        return
    from collections import Counter

    curv = Counter()
    for p in labels:
        d = json.loads(p.read_text())
        curv[d["curvature"]] += 1
    print(f"{len(labels)} rides under {root}")
    for k, v in curv.most_common():
        print(f"  {k:>7}: {v}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=str, default="data/synthetic_transitions")
    ap.add_argument("--refs", type=str, default=DEFAULT_REFS)
    ap.add_argument("--mix-s", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", dest="list_dir", type=str, default=None)
    args = ap.parse_args(argv)

    if args.list_dir:
        summarize(Path(args.list_dir).expanduser())
        return 0

    ref_paths = sorted(glob.glob(str(Path(args.refs).expanduser())))
    if not ref_paths:
        print(f"no ref audio matched {args.refs}", file=sys.stderr)
        return 2
    print(f"ref pool: {len(ref_paths)} files")

    specs = build_specs(ref_paths, args.n, args.mix_s, args.seed)
    out_root = Path(args.out).expanduser()
    for i, spec in enumerate(specs):
        label = render_spec(spec, out_root / f"ride_{i:04d}")
        print(
            f"  ride_{i:04d}  {spec.curvature:>7} {spec.direction}  "
            f"r={spec.r0:.2f}->{spec.r1:.2f}  ref_span={label['ref_span_s']:.0f}s  "
            f"{Path(spec.ref_path).parent.name[:32]}"
        )
    print(f"\nwrote {len(specs)} rides -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
