"""Tests for the learning harness (Bayesian-bandit precision model) + new probes."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.agentic.auditory import (
    common_fate,
    comodulation_residual,
    tempogram,
    ENV_FPS,
)
from workspaces.alignment_prototype.agentic.events import Event
from workspaces.alignment_prototype.agentic.learning import (
    Beta,
    PrecisionModel,
    gt_correct_fn,
)


def test_beta_update_moves_mean_toward_evidence():
    b = Beta(2.0, 2.0)  # mean 0.5
    for _ in range(20):
        b = b.update(True)
    assert b.mean > 0.85 and b.n == 24.0  # 2+2 prior + 20 successes


def test_seeded_model_reflects_registry_and_validated_confidence():
    from workspaces.alignment_prototype.agentic.actions import REGISTRY

    m = PrecisionModel.seeded()
    # a validated probe seeds near its (current) registry value, high confidence
    assert abs(m.precision("fp") - REGISTRY["fp"].precision) < 0.02
    assert m.confidence("fp") == 20.0
    # unvalidated onset_align seeds weak (explorable)
    assert m.confidence("onset_align") == 4.0


def test_calibration_can_demote_an_overrated_probe():
    m = PrecisionModel.seeded()
    start = m.precision("surprise")
    for _ in range(30):  # surprise keeps being wrong
        m.observe("surprise", correct=False)
    assert m.precision("surprise") < start - 0.2


def test_calibration_promotes_a_proven_provisional_probe():
    m = PrecisionModel.seeded()
    start = m.precision("onset_align")  # ~0.60, weak prior
    for _ in range(40):  # onset_align keeps being right
        m.observe("onset_align", correct=True)
    assert m.precision("onset_align") > start + 0.15
    assert m.precision("onset_align") > m.precision(
        "mert_decode"
    )  # overtook a validated mid-probe


def test_thompson_rank_prefers_cheap_high_precision():
    m = PrecisionModel.seeded()
    rng = np.random.default_rng(0)
    # fp (0.90 / cost 0.1) should usually top mert_decode (0.55 / cost 0.5)
    tops = [m.rank(("fp", "mert_decode"), rng)[0] for _ in range(50)]
    assert tops.count("fp") > 40


def test_fit_from_events_learns_from_log():
    m = PrecisionModel.seeded()
    events = tuple(
        Event(
            seq=i,
            span_key="001",
            kind="observe",
            payload={
                "probe": "onset_align",
                "set_start_s": 100.0 + (0 if i % 2 else 30),
            },
        )
        for i in range(10)
    )
    # GT at 100; half the proposals (100.0) are right, half (130.0) wrong
    gt = {"recX": [{"set_start_s": 100.0}]}
    fn = gt_correct_fn(gt, id_of=lambda k: "recX", tol_s=15.0)
    n = m.fit_from_events(events, fn)
    assert n == 10
    b = m.posteriors["onset_align"]
    assert b.alpha > 4.0 and b.beta > 4.0  # both right and wrong seen


def test_precision_model_roundtrips(tmp_path):
    m = PrecisionModel.seeded()
    m.observe("fp", True)
    p = tmp_path / "model.json"
    m.save(p)
    m2 = PrecisionModel.load(p)
    assert abs(m2.precision("fp") - m.precision("fp")) < 1e-9


def test_tempogram_finds_beat():
    # onset envelope with an impulse every 0.5 s = 120 BPM
    n = int(ENV_FPS * 10)
    env = np.zeros(n)
    step = int(ENV_FPS * 0.5)
    env[::step] = 1.0
    res = tempogram(env)
    assert res is not None
    bpm, strength = res
    assert abs(bpm - 120.0) < 6.0 and strength > 0.5


def test_resolve_with_model_uses_calibrated_precision():
    from workspaces.alignment_prototype.agentic.actions import REGISTRY
    from workspaces.alignment_prototype.agentic.events import EventLog
    from workspaces.alignment_prototype.agentic.loop import SpanCtx, resolve

    from workspaces.alignment_prototype.agentic.belief import Observation

    m = PrecisionModel.seeded()
    for _ in range(200):  # drive fp's learned precision far below its 0.90 prior
        m.observe("fp", correct=False)
    learned_fp = m.precision("fp")
    assert learned_fp < 0.4

    def _abstain(name, prec):
        return lambda d: Observation(name, None, 0.0, prec)

    spans = [SpanCtx("001", "r1", "regular", {"slot": "001"})]
    runners = {
        "cue_prior": _abstain("cue_prior", 0.5),
        "mert_decode": _abstain("mert_decode", 0.55),
        "fp": lambda d: Observation("fp", 100.0, 0.9, REGISTRY["fp"].precision),
        "chroma_refine": _abstain("chroma_refine", 0.7),
    }
    log = EventLog()
    res = resolve(spans, runners, log, model=m, rng=np.random.default_rng(0))
    # fp fired; its recorded precision must be the LEARNED value, not 0.90
    fp_ev = [
        e for e in log.events if e.kind == "observe" and e.payload["probe"] == "fp"
    ]
    assert fp_ev and abs(fp_ev[0].payload["precision"] - learned_fp) < 1e-9
    # with fp calibrated low, this span should NOT auto-commit on fp alone
    assert "001" not in res.committed


def test_common_fate_high_when_coherent_low_when_independent():
    rng = np.random.default_rng(3)
    shared = rng.standard_normal(300)
    coherent = np.stack([shared + 0.05 * rng.standard_normal(300) for _ in range(4)])
    independent = rng.standard_normal((4, 300))
    assert common_fate(coherent) > 0.8
    assert abs(common_fate(independent)) < 0.3


def test_comodulation_residual_pops_an_independent_overlay():
    # comodulation masking release: a bed carrier shared across bands is
    # suppressed; an overlay entering mid-signal (independent modulation in a
    # subset of bands) survives as a residual spike.
    rng = np.random.default_rng(5)
    n = 400
    bed = np.abs(rng.standard_normal(n))  # the shared across-band modulator
    bands = np.stack([bed + 0.03 * rng.standard_normal(n) for _ in range(6)])
    overlay = np.abs(rng.standard_normal(n))
    bands[0, n // 2 :] += overlay[n // 2 :]  # overlay in 2/6 bands, 2nd half only
    bands[1, n // 2 :] += overlay[n // 2 :]
    s = comodulation_residual(bands)
    assert s.shape == (n,)
    assert s[: n // 2].mean() < 0.25  # coherent bed half: residual ~0
    assert s[n // 2 :].mean() > 3 * s[: n // 2].mean()  # overlay half pops


def test_comodulation_residual_flat_when_all_coherent():
    rng = np.random.default_rng(6)
    bed = np.abs(rng.standard_normal(300))
    bands = np.stack([bed + 0.02 * rng.standard_normal(300) for _ in range(5)])
    s = comodulation_residual(bands)
    # no independent overlay anywhere ⇒ no sustained pop (peak-normed, so the
    # curve exists but carries no salient structure): low mean
    assert s.mean() < 0.5
