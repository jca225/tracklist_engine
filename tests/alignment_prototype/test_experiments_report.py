from __future__ import annotations
from workspaces.alignment_prototype.experiments.report import (
    mean_ci,
    paired_delta_ci,
    headline_table,
)


def test_mean_ci_brackets_the_mean():
    m, lo, hi = mean_ci([0.0, 0.5, 1.0, 0.5], seed=0)
    assert lo <= m <= hi
    assert abs(m - 0.5) < 1e-9


def test_paired_delta_ci_sign_and_seed_stability():
    a = [0.9, 0.8, 0.85, 0.95]  # "with"
    b = [0.4, 0.3, 0.35, 0.45]  # "without"
    d, lo, hi = paired_delta_ci(a, b, seed=0)
    assert d > 0 and lo > 0  # clearly positive delta, CI excludes 0
    assert (d, lo, hi) == paired_delta_ci(a, b, seed=0)  # deterministic


def test_headline_has_strict_fiber_and_gap():
    rows = [
        {
            "set_id": "2nvzlh2k",
            "driver": "classical",
            "decoder": "looptrace",
            "stem": "acappella",
            "strict": 0.12,
            "fiber": 0.31,
        },
        {
            "set_id": "2nvzlh2k",
            "driver": "classical",
            "decoder": "looptrace",
            "stem": "acappella",
            "strict": 0.10,
            "fiber": 0.29,
        },
    ]
    md = headline_table(rows)
    assert "strict" in md and "fiber" in md and "gap" in md
    assert "acappella" in md
