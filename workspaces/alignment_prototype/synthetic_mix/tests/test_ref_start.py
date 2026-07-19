"""ref_start drop-from-top sampling — the measured sim2real fix.

Real BB spans drop a track from its start ~24% of the time (ref_start≈0);
synthetic v2 never did (acappella ref_lo = uniform(20,70)). The generator learns
a false 'ref_start is never 0' prior that pushes real eval BELOW the raw control.
`ref_start_sample` injects the drop-from-top mode at a calibrated probability."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.synthetic_mix.scenario_v2 import ref_start_sample


def test_drop_prob_zero_never_returns_top():
    rng = np.random.default_rng(0)
    xs = [ref_start_sample(rng, 20.0, 70.0, 0.0) for _ in range(200)]
    assert all(20.0 <= x <= 70.0 for x in xs)
    assert min(xs) >= 20.0  # never drops to 0


def test_drop_prob_one_always_returns_top():
    rng = np.random.default_rng(0)
    xs = [ref_start_sample(rng, 20.0, 70.0, 1.0) for _ in range(50)]
    assert all(x == 0.0 for x in xs)


def test_drop_prob_matches_calibrated_rate():
    rng = np.random.default_rng(0)
    xs = [ref_start_sample(rng, 20.0, 70.0, 0.24) for _ in range(4000)]
    frac_top = np.mean([x == 0.0 for x in xs])
    assert 0.20 < frac_top < 0.28  # ~24%, matching real BB
    # the non-top samples still respect the range
    assert all(x == 0.0 or 20.0 <= x <= 70.0 for x in xs)
