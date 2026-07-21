from __future__ import annotations

import random

from workspaces.pws_aligner.continuous_model import ContinuousLabelModel, FusedSpan
from workspaces.pws_aligner.votes import AbstainReason, Vote


def _vote(
    probe: str, span_id: str, rec: str | None, off: float, abstained: bool = False
) -> Vote:
    return Vote(
        probe=probe,
        span_id=span_id,
        recording_id=rec,
        offset_s=off,
        confidence=0.8,
        abstained=abstained,
        reason=AbstainReason.NO_DATA if abstained else AbstainReason.NONE,
        features=(),
    )


# (accuracy, sigma_s, inlier) — heterogeneous by design: this is the regime
# that killed Dawid-Skene (fp ~0.2s vs chroma ~seconds across 2s bins).
_PROBES = {
    "sharp": (0.90, 0.3, 0.90),
    "mid": (0.80, 2.0, 0.85),
    "blurry": (0.60, 6.0, 0.80),
}
_RECS = ("r1", "r2", "r3")


def _synthetic(n: int = 300, seed: int = 7):
    rng = random.Random(seed)
    spans, truths = [], []
    for i in range(n):
        true_r = rng.choice(_RECS)
        true_mu = rng.uniform(-120.0, 120.0)
        votes = []
        for name, (acc, sig, eta) in _PROBES.items():
            if rng.random() < 0.15:
                votes.append(_vote(name, f"s{i}", None, 0.0, abstained=True))
                continue
            if rng.random() < acc:
                off = (
                    rng.gauss(true_mu, sig)
                    if rng.random() < eta
                    else rng.uniform(-240, 240)
                )
                votes.append(_vote(name, f"s{i}", true_r, off))
            else:
                wrong = rng.choice([r for r in _RECS if r != true_r])
                votes.append(_vote(name, f"s{i}", wrong, rng.uniform(-240, 240)))
        spans.append(tuple(votes))
        truths.append((true_r, true_mu))
    return spans, truths


def test_oracle_learns_sigma_ordering_and_identity():
    # Guards: dense co-vote σ-ordering recovery (M-step correctness) — the
    # complement of test_singleton_matches_floor_sigma_uniformly.  Here every
    # span has ALL probes voting the SAME true recording (high co-vote density),
    # so _fused_mu has multiple residuals and the M-step correctly recovers σ
    # ordering.  If this test regresses, check the M-step accumulation loop.
    spans, truths = _synthetic()
    m = ContinuousLabelModel()
    m.fit(spans)
    noise = m.probe_noise()
    assert noise["sharp"].sigma_s < noise["mid"].sigma_s < noise["blurry"].sigma_s
    assert noise["sharp"].sigma_s < 1.0
    for name, (acc, _sig, _eta) in _PROBES.items():
        assert abs(noise[name].accuracy - acc) < 0.15
    correct = 0
    errs = []
    for votes, (true_r, true_mu) in zip(spans, truths):
        fused = m.predict(votes)
        if fused.recording_id == true_r:
            correct += 1
            errs.append(abs(fused.offset_s - true_mu))
    assert correct / len(spans) >= 0.85
    errs.sort()
    assert errs[len(errs) // 2] < 0.8  # median fused error ~ sharp-probe scale


def test_heterogeneous_precision_no_null_collapse():
    """THE regression for the v2b kill-gate: three probes agreeing on the
    recording but spread across different 2s bins must fuse, not go NULL."""
    votes = (
        _vote("fp", "s0", "rX", 100.1),
        _vote("hubert", "s0", "rX", 102.5),
        _vote("chroma", "s0", "rX", 104.5),
    )
    fused = ContinuousLabelModel().predict(votes)  # pre-fit: init priors
    assert fused.recording_id == "rX"
    assert abs(fused.offset_s - 100.1) < 1.5  # pulled toward the sharp probe


def test_all_abstain_is_null():
    votes = (_vote("fp", "s0", None, 0.0, abstained=True),)
    fused = ContinuousLabelModel().predict(votes)
    assert fused.recording_id is None
    assert fused.n_votes == 0


def test_single_vote_beats_null():
    fused = ContinuousLabelModel().predict((_vote("fp", "s0", "rY", 42.0),))
    assert fused.recording_id == "rY"
    assert abs(fused.offset_s - 42.0) < 1e-6


def test_unseen_probe_uses_default_priors():
    spans, _ = _synthetic(n=50)
    m = ContinuousLabelModel()
    m.fit(spans)
    fused = m.predict((_vote("brand_new_probe", "s0", "rZ", 10.0),))
    assert fused.recording_id == "rZ"


def test_singleton_matches_shrink_to_prior_not_floor():
    """v4 fix for the Gate-v3 singleton degeneracy (supersedes the old
    test_singleton_matches_floor_sigma_uniformly).

    When each span's votes match DIFFERENT recordings — every recording-match is
    a singleton (no two probes agree on the same recording) — _fused_mu returns
    the sole vote's own offset as μ, so resid = offset − μ is exactly 0.  Such
    matches are σ-UNINFORMATIVE.  v4 M-step must therefore NOT let singletons drive
    σ to the floor: it skips singleton matches for σ and shrinks toward the
    supervised per-probe PRIOR.  Under an all-singleton corpus the model has no
    co-vote evidence, so σ must reflect the prior ordering (fp tight … chroma
    loose), NOT collapse uniformly to _SIGMA_FLOOR_S.
    """
    rng = random.Random(42)
    spans = []
    # Real probe names so the supervised prior applies: prior σ fp < hubert < chroma.
    recs_pool = [f"r{i}" for i in range(30)]
    probes = ["fp", "hubert", "chroma"]
    true_sigmas = {"fp": 0.3, "hubert": 2.0, "chroma": 6.0}
    for i in range(100):
        # Shuffle recs so each probe gets a unique recording on this span → all singletons.
        per_probe_recs = rng.sample(recs_pool, len(probes))
        votes = []
        for probe, rec in zip(probes, per_probe_recs):
            off = rng.gauss(float(i), true_sigmas[probe])
            votes.append(_vote(probe, f"s{i}", rec, off))
        spans.append(tuple(votes))

    m = ContinuousLabelModel(max_iter=100)
    m.fit(spans)
    noise = m.probe_noise()
    sigmas = {p: noise[p].sigma_s for p in probes}

    # Prior ordering preserved (NOT uniformly floored):
    assert sigmas["fp"] < sigmas["hubert"] < sigmas["chroma"], sigmas
    # And σ is not collapsed to the floor for the looser probes:
    from workspaces.pws_aligner.continuous_model import _SIGMA_FLOOR_S

    assert max(sigmas.values()) > _SIGMA_FLOOR_S * 3, sigmas
