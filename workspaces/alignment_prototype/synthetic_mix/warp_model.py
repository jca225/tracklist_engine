"""Empirical warp sampler — reads warp_prior.json (fitted on BB11+BB12 GT).

The low-rank finding (eda/alignment/warp_analysis): DJ warp is two orthogonal,
channel-locked axes.

  * bed channel (instrumental/regular): plays NEAR-NATIVE — tempo_ratio ~
    N(1, ~0.0115). Not a free knob; do not apply a big master-tempo jitter.
  * overlay channel (acappella/regular vocal): stretch is DERIVED from the BPM
    gap, ``tempo_ratio = mix_BPM / payload_native_BPM`` (70-75% within +/-5% on
    real GT), with a ~10% half/double octave fold.

`tempo_ratio` here is the GT convention (validated by decompose_bpm.py):
ref-seconds consumed per mix-second = mix_BPM / native_BPM. A fast acappella
(native > mix) is SLOWED (ratio < 1) — the reverse of the generator's old
``payload.bpm/bed.bpm``, which warped overlays the wrong way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PRIOR_PATH = Path(__file__).resolve().parent / "warp_prior.json"


@dataclass(frozen=True)
class WarpPrior:
    bed_sigma: float
    octave_fold_prob: float
    pitch_dist: dict[int, float]

    @classmethod
    def load(cls, path: Path = _PRIOR_PATH) -> "WarpPrior":
        d = json.loads(path.read_text())
        pitch = {int(k): float(v) for k, v in d["pitch_shift_semi"]["dist"].items()}
        return cls(
            bed_sigma=float(d["bed_tempo"]["sigma"]),
            octave_fold_prob=float(d["overlay_warp"]["octave_fold_prob"]),
            pitch_dist=pitch,
        )

    def bed_tempo_ratio(self, rng: np.random.Generator) -> float:
        """Near-native bed warp ~ N(1, sigma), clipped to a sane band."""
        return float(np.clip(rng.normal(1.0, self.bed_sigma), 0.94, 1.06))

    def overlay_tempo_ratio(
        self, mix_bpm: float, payload_native_bpm: float, rng: np.random.Generator
    ) -> float:
        """Derived overlay warp = mix_BPM / native, with a ~10% octave fold."""
        if mix_bpm <= 0 or payload_native_bpm <= 0:
            return 1.0
        tr = mix_bpm / payload_native_bpm
        if rng.random() < self.octave_fold_prob:
            tr *= 2.0 if rng.random() < 0.5 else 0.5
        return float(tr)
