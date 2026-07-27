"""TDD gate for the earliest-fiber-instance (ref-position) tie-break.

The instance-separability finding (INSTANCE_SEPARABILITY_FINDINGS.md) showed the
correct fiber instance is the EARLIEST ref-position member ~0.93 of the time and
that no learned content selector transfers. The lever is therefore a parameter-
free position prior in the Viterbi decoder that breaks a near-tie between
equivalent fiber siblings toward the earliest (smallest ref-start), WITHOUT
overriding a genuine emission difference. This is distinct from the existing
continuity/warp jump slopes (``fwd_slope``/``back_slope``), which prefer the
*nearer* instance, not the earliest-in-ref.
"""

from __future__ import annotations

import numpy as np

from alignment.path_decode import _viterbi, decode_path


def _two_instance_reward(
    early_k: int, late_k: int, k: int, tm: int, early: float, late: float
) -> np.ndarray:
    """tm windows x K states with two equal-length plateaus (fiber siblings):
    an early instance at ``early_k`` and a later one at ``late_k`` (> early_k)."""
    r = np.zeros((tm, k), dtype=np.float32)
    r[:, early_k] = early
    r[:, late_k] = late
    return r


def test_default_picks_the_slightly_stronger_late_instance():
    """Baseline (r0_slope=0): a near-tie where noise gives the LATER sibling a
    hair more emission decodes to the late instance — the position-agnostic
    baseline the tie-break must improve on."""
    reward = _two_instance_reward(3, 9, k=12, tm=5, early=1.00, late=1.02)
    _score, path = _viterbi(reward, lam_fwd=1.0, lam_back=1.0)
    assert np.all(path == 9)


def test_position_prior_flips_the_near_tie_to_earliest():
    """A positive r0_slope makes the constant early-instance path win the near-
    tie: it recovers the ~0.93 earliest-instance prior with no fit."""
    reward = _two_instance_reward(3, 9, k=12, tm=5, early=1.00, late=1.02)
    _score, path = _viterbi(reward, lam_fwd=1.0, lam_back=1.0, r0_slope=0.01)
    assert np.all(path == 3)


def test_position_prior_does_not_override_a_real_emission_gap():
    """It is a TIE-break, not an override: a genuinely stronger later instance
    (not a fiber sibling) survives a small position prior."""
    reward = _two_instance_reward(3, 9, k=12, tm=5, early=1.00, late=1.50)
    _score, path = _viterbi(reward, lam_fwd=1.0, lam_back=1.0, r0_slope=0.01)
    assert np.all(path == 9)


def test_r0_slope_zero_is_exact_baseline():
    """r0_slope defaults to 0.0 and then is a bit-for-bit no-op."""
    reward = _two_instance_reward(3, 9, k=12, tm=5, early=1.00, late=1.02)
    base_score, base_path = _viterbi(reward, lam_fwd=1.0, lam_back=1.0)
    z_score, z_path = _viterbi(reward, lam_fwd=1.0, lam_back=1.0, r0_slope=0.0)
    assert z_score == base_score
    assert np.array_equal(z_path, base_path)


def test_decode_path_forwards_r0_slope_and_zero_is_noop():
    """decode_path accepts r0_slope and r0_slope=0.0 reproduces the default."""
    rng = np.random.default_rng(0)
    d, tm, tr = 12, 300, 900
    m = rng.standard_normal((d, tm)).astype(np.float32)
    r = rng.standard_normal((d, tr)).astype(np.float32)
    m /= np.linalg.norm(m, axis=0, keepdims=True) + 1e-9
    r /= np.linalg.norm(r, axis=0, keepdims=True) + 1e-9
    base = decode_path(m, r, (1.0,), lam=0.15)
    zero = decode_path(m, r, (1.0,), lam=0.15, r0_slope=0.0)
    assert base == zero
