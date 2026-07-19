"""Tests for the agentic aligner harness (belief / events / actions / policy / loop)."""

from __future__ import annotations

from workspaces.alignment_prototype.agentic.actions import REGISTRY, bind, plan_for
from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
from workspaces.alignment_prototype.agentic.events import EventLog, replay_beliefs
from workspaces.alignment_prototype.agentic.loop import SpanCtx, resolve
from workspaces.alignment_prototype.agentic.policy import Ladder, Mode


def _obs(probe, start, conf=1.0, prec=0.9, cost=0.1):
    return Observation(
        probe=probe, set_start_s=start, confidence=conf, precision=prec, cost=cost
    )


def _belief(*obs):
    b = SpanBelief("001", "rec1", "acappella")
    for o in obs:
        b = b.observe(o)
    return b


def test_belief_clusters_agreeing_probes():
    b = _belief(
        _obs("fp", 100.0), _obs("lyrics", 103.0), _obs("mert_decode", 400.0, prec=0.55)
    )
    top = b.best()
    assert top is not None
    assert 100.0 <= top.set_start_s <= 103.0  # fp+lyrics agree within tol
    assert set(top.probes) == {"fp", "lyrics"}
    assert b.quality() > 0.6  # dominant cluster, high-precision member


def test_belief_conflict_lowers_quality():
    agree = _belief(_obs("fp", 100.0), _obs("lyrics", 102.0))
    conflict = _belief(_obs("fp", 100.0), _obs("lyrics", 400.0))
    assert conflict.quality() < agree.quality()


def test_fp_cluster_preferred_over_mert_surprise_pileup():
    """bb12_42w5 failure mode: surprise+mert co-cluster outvotes correct fp.

    Registry-like weights: fp 0.53×0.8=0.424; surprise 0.45×1.0=0.45;
    mert 0.55×0.7=0.385. mert+surprise share a cluster (~3468) heavier than
    lone fp (~3519), but fp is within FP_CLUSTER_MARGIN of that weight and
    must win — otherwise agentic:surprise overwrites a GT-correct diagonal.
    """
    b = _belief(
        _obs("fp", 3519.5, conf=0.8, prec=0.53),
        _obs("mert_decode", 3470.4, conf=0.7, prec=0.55),
        _obs("surprise", 3466.08, conf=1.0, prec=0.45),
    )
    top = b.best()
    assert top is not None
    assert "fp" in top.probes
    assert abs(top.set_start_s - 3519.5) < 1.0


def test_weak_fp_cluster_does_not_steal_strong_mert_pileup():
    """A low-weight stray fp diagonal must not beat a much heavier mert cluster."""
    b = _belief(
        _obs("fp", 100.0, conf=0.2, prec=0.53),  # weight ≈ 0.106
        _obs("mert_decode", 400.0, conf=1.0, prec=0.55),  # 0.55
        _obs("surprise", 402.0, conf=1.0, prec=0.45),  # 0.45 → pileup ≈ 1.0
    )
    top = b.best()
    assert top is not None
    assert "fp" not in top.probes
    assert 400.0 <= top.set_start_s <= 402.0


def test_belief_all_abstain_is_zero():
    b = _belief(Observation("lyrics", None, 0.0, 0.9))
    assert b.quality() == 0.0
    assert b.best() is None


def test_combine_rewards_independent_agreement():
    # three independent mediocre probes agreeing should clear a higher bar than
    # any one of them — that's the whole point of the combine gate.
    b = _belief(
        _obs("cue_prior", 100.0, prec=0.50),
        _obs("mert_decode", 101.0, prec=0.55),
        _obs("fp", 102.0, prec=0.53),
    )
    base = b.quality()  # max-precision gate
    comb = b.quality(combine=True)  # noisy-OR over independent groups
    assert abs(base - 0.55) < 1e-9  # share 1.0 × best precision
    assert abs(comb - (1 - 0.50 * 0.45 * 0.47)) < 1e-9  # ≈ 0.894
    assert comb > base


def test_combine_single_probe_is_unchanged():
    b = _belief(_obs("fp", 100.0, prec=0.53))
    assert abs(b.quality(combine=True) - b.quality()) < 1e-9  # noisy-OR of one = itself


def test_combine_does_not_double_count_correlated_probes():
    # surprise snaps to mert's centre — same independence group, so agreeing they
    # combine to the STRONGER of the two, not a noisy-OR product.
    b = _belief(
        _obs("mert_decode", 100.0, prec=0.55), _obs("surprise", 100.5, prec=0.60)
    )
    assert abs(b.quality(combine=True) - 0.60) < 1e-9  # not 1-(.45)(.40)=0.82


def test_event_log_replay_roundtrip(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    b = SpanBelief("007", "recX", "regular")
    b = log.observe(b, _obs("fp", 55.0))
    b = log.observe(b, _obs("chroma_refine", 57.0, prec=0.7))
    # fresh log object reads the same file; replay reconstructs the belief
    events = EventLog(path).events
    beliefs = replay_beliefs(events, {"007": ("recX", "regular")})
    assert beliefs["007"].quality() == b.quality()
    assert beliefs["007"].best().set_start_s == b.best().set_start_s


def test_dominance_plans_prune_dominated_probes():
    vocal = [a.name for a in plan_for("acappella")]
    host = [a.name for a in plan_for("regular")]
    assert "lyrics" in vocal and vocal.index("lyrics") < vocal.index("fp")
    assert "fp" in host and "lyrics" not in host  # lyrics dominated on hosts
    # surprise is the tie-breaker tail: it fires only on spans the rest of the
    # plan failed to resolve (a weak vote diluting an already-confident belief
    # costs auto coverage — measured), and mert_decode must precede it (it
    # supplies the w-row band center)
    assert vocal[-1] == "surprise" and host[-1] == "surprise"
    assert vocal.index("mert_decode") < vocal.index("surprise")
    # before the tail, plans are information-ordered, most expensive probe last
    assert vocal[-2] == "stem_hubert" and host[-2] == "chroma_refine"


def test_bind_rejects_unknown_actions():
    try:
        bind({"warp_drive": lambda ctx: None})
    except ValueError as e:
        assert "warp_drive" in str(e)
    else:
        raise AssertionError("bind accepted an unregistered action")


def test_ladder_rungs():
    ladder = Ladder()
    strong = _belief(_obs("lyrics", 100.0), _obs("mert_decode", 101.0, prec=0.55))
    weak = _belief(
        _obs("cue_prior", 100.0, prec=0.5, conf=0.4),
        _obs("mert_decode", 500.0, prec=0.55),
    )
    silent = _belief(Observation("lyrics", None, 0.0, 0.9))
    assert ladder.mode(strong) is Mode.AUTO_COMMIT
    assert ladder.mode(weak) in (Mode.REVIEW, Mode.SUGGEST)
    assert ladder.mode(silent) is Mode.ESCALATE


def test_loop_early_termination_and_queues():
    calls: list[str] = []

    def runner(name, start, prec=0.9):
        def _run(data):
            calls.append(f"{name}:{data['slot']}")
            return _obs(name, start, prec=prec, cost=REGISTRY[name].cost)

        return _run

    spans = [
        SpanCtx("easy", "r1", "acappella", {"slot": "easy"}),
        SpanCtx("hard", "r2", "acappella", {"slot": "hard"}),
    ]
    runners = {
        "cue_prior": runner("cue_prior", 100.0, prec=0.5),
        "mert_decode": runner("mert_decode", 100.0, prec=0.55),
        # lyrics agrees on "easy" (auto-commit early) …
        "lyrics": lambda d: _obs(
            "lyrics", 100.0 if d["slot"] == "easy" else None, cost=1.0
        ),
        # … so stem_hubert must NEVER run for "easy"
        "stem_hubert": runner("stem_hubert", 100.0, prec=0.75),
    }
    res = resolve(spans, runners, EventLog(), ladder=Ladder())
    assert "easy" in res.committed
    assert "stem_hubert:easy" not in calls  # early termination saved the expensive call
    assert "stem_hubert:hard" in calls  # hard span escalated up the plan
    assert (
        res.actions_run == len(calls) + 2
    )  # +2: the lyrics lambda (untracked) ran for both spans


def test_loop_budget_stops_expensive_actions():
    spans = [SpanCtx("s1", "r1", "acappella", {"slot": "s1"})]
    runners = {
        "cue_prior": lambda d: _obs("cue_prior", 100.0, prec=0.5, cost=0.0),
        "mert_decode": lambda d: _obs("mert_decode", 300.0, prec=0.55, cost=0.5),
        "lyrics": lambda d: _obs("lyrics", None, cost=1.0),
        "stem_hubert": lambda d: _obs("stem_hubert", 100.0, prec=0.75, cost=3.0),
    }
    res = resolve(spans, runners, EventLog(), budget=2.0)  # can't afford hubert (3.0)
    assert res.cost_spent <= 2.0
    assert (
        "stem_hubert"
        not in next(
            iter(
                res.escalated.values() or res.review.values() or res.suggested.values()
            )
        ).probes_run()
        if (res.escalated or res.review or res.suggested)
        else True
    )
