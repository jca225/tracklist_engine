#!/usr/bin/env python3
"""Phase-5 tests: the residual DP bonus must (a) prefer the rival whose
reference content actually matches the mix, (b) emit NOTHING on
audit-uncovered frames (the Phase-4 failure guard), (c) be zero-sum within
a rival group, and (d) group only lag-related diagonals."""

from __future__ import annotations

import numpy as np

from alignment.looptrace import residual
from alignment.looptrace.segments import Diagonal

HZ = 10.0


def _mels():
    """Reference with two similar-but-distinct instances 20 s apart, and a
    mix that plays instance 2 (ref 30..36 s) at slope 1."""
    rng = np.random.default_rng(3)
    ref = rng.standard_normal((8, 600))
    inst1 = ref[:, 100:160]
    inst2 = inst1 + 0.35 * rng.standard_normal(inst1.shape)  # distinct take
    ref[:, 300:360] = inst2
    ref = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-9)
    mix = ref[:, 300:360].copy()
    return mix, ref


def _mask(covered: bool) -> np.ndarray:
    m = np.ones(600, dtype=np.float32)
    if covered:
        m[100:160] = 0.35  # audit-covered, moderately discriminative
        m[300:360] = 0.35
    return m


DIAGS = [Diagonal(10.0, 50), Diagonal(30.0, 50)]  # instance 1 vs 2 (correct)
PAIRS = [{"a0": 10.0, "a1": 16.0, "lag_s": 20.0, "class": "distinct"}]
GRID = np.arange(0.0, 6.0, 0.5)


def test_bonus_prefers_true_instance():
    mix, ref = _mels()
    b = residual.residual_bonus(
        mix, HZ, ref, HZ, _mask(True), HZ, DIAGS, [[0, 1]], 1.0, GRID
    )
    active = b[1] != 0
    assert active.mean() > 0.5, "bonus should fire on most covered steps"
    assert (b[1][active] > 0).mean() > 0.9  # correct instance rewarded
    assert (b[0][active] < 0).mean() > 0.9  # wrong instance penalized


def test_no_bonus_on_uncovered_frames():
    mix, ref = _mels()
    b = residual.residual_bonus(
        mix, HZ, ref, HZ, _mask(False), HZ, DIAGS, [[0, 1]], 1.0, GRID
    )
    assert not b.any()  # Phase-4 guard: no audit coverage -> no vote


def test_bonus_is_zero_sum_within_group():
    mix, ref = _mels()
    b = residual.residual_bonus(
        mix, HZ, ref, HZ, _mask(True), HZ, DIAGS, [[0, 1]], 1.0, GRID
    )
    np.testing.assert_allclose(b.sum(axis=0), 0.0, atol=1e-9)


def test_rival_groups_by_audit_lag():
    diags = [Diagonal(10.0, 9), Diagonal(30.0, 9), Diagonal(47.0, 9)]
    groups = residual.rival_groups(diags, PAIRS)
    assert groups == [[0, 1]]  # 47.0 is 17 s off — not a repeat image
