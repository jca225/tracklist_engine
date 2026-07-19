"""Tests for the agentic aligner harness (belief / events / actions / policy / loop)."""

from __future__ import annotations

from pathlib import Path

from workspaces.alignment_prototype.agentic.actions import REGISTRY, bind, plan_for
from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
from workspaces.alignment_prototype.agentic.events import EventLog, replay_beliefs
from workspaces.alignment_prototype.agentic.live_runners import LiveContext
from workspaces.alignment_prototype.agentic.loop import SpanCtx, resolve
from workspaces.alignment_prototype.agentic.policy import Ladder, Mode
from workspaces.alignment_prototype.drivers.agentic import (
    pseudo_acceptance,
    refine_agentic_spans,
)


def _obs(probe, start, conf=1.0, prec=0.9, cost=0.1):
    return Observation(
        probe=probe, set_start_s=start, confidence=conf, precision=prec, cost=cost
    )


def _belief(*obs, stem: str = "acappella"):
    b = SpanBelief("001", "rec1", stem)
    for o in obs:
        b = b.observe(o)
    return b


def _timeline_span(slot: str, recording_id: str, *, stem: str = "regular") -> dict:
    return {
        "slot_label": slot,
        "recording_id": recording_id,
        "claimed_stem": stem,
        "set_start_s": 10.0,
        "set_end_s": 40.0,
        "ref_segments": [{"set_start_s": 10.0, "ref_start_s": 20.0}],
    }


def test_pseudo_acceptance_requires_independence_or_high_precision():
    independent = _belief(
        _obs("mert_decode", 100.0, prec=0.70),
        _obs("lyrics", 101.0, prec=0.70),
    )
    correlated = _belief(
        _obs("fp", 100.0, prec=0.80),
        _obs("chroma_refine", 101.0, prec=0.80),
    )
    single_high = _belief(_obs("lyrics", 100.0, prec=0.90))
    fiber_ambiguous = _belief(
        Observation(
            "lyrics",
            100.0,
            confidence=1.0,
            precision=0.90,
            ref_start_s=20.0,
            detail="[fiber×2 amb]",
        )
    )

    assert pseudo_acceptance(independent) == (True, None)
    assert pseudo_acceptance(correlated) == (False, "g2_independence")
    assert pseudo_acceptance(single_high) == (True, None)
    assert pseudo_acceptance(fiber_ambiguous) == (False, "g3_fiber")


def test_resolve_require_independence_continues_past_solo_auto(tmp_path):
    """Lyrics at 0.76 clears the auto bar alone; pseudo-safe must still query
    a second independent probe (stem_hubert) so G2 can pass."""
    spans = [
        SpanCtx("1", "r1", "acappella", _timeline_span("1", "r1", stem="acappella"))
    ]
    calls: list[str] = []

    def lyrics(_):
        calls.append("lyrics")
        return _obs("lyrics", 30.0, prec=0.76)

    def hubert(_):
        calls.append("stem_hubert")
        return _obs("stem_hubert", 31.0, prec=0.75)

    runners = {"lyrics": lyrics, "stem_hubert": hubert}
    ladder = Ladder(auto=0.75, combine=True)

    resolve(
        spans,
        runners,
        EventLog(tmp_path / "indep.jsonl"),
        ladder=ladder,
        require_independence=True,
    )
    assert calls == ["lyrics", "stem_hubert"]

    # Without the flag, early-stop after lyrics alone.
    calls.clear()
    resolve(
        spans,
        runners,
        EventLog(tmp_path / "early.jsonl"),
        ladder=ladder,
        require_independence=False,
    )
    assert calls == ["lyrics"]


def test_refine_agentic_spans_demotes_unsafe_pseudo_commit(tmp_path):
    timeline = {
        "set_id": "pool",
        "spans": [
            _timeline_span("1", "r1"),
            _timeline_span("2", "r2", stem="acappella"),
        ],
    }
    spans = [
        SpanCtx("1", "r1", "regular", timeline["spans"][0]),
        SpanCtx("2", "r2", "acappella", timeline["spans"][1]),
    ]
    runners = {
        "fp": lambda _: _obs("fp", 25.0, prec=0.80),
        "lyrics": lambda _: _obs("lyrics", 30.0, prec=0.90),
    }
    ladder = Ladder(auto=0.75, combine=True)

    general, _ = refine_agentic_spans(
        timeline,
        spans,
        runners,
        EventLog(tmp_path / "general.jsonl"),
        ladder=ladder,
    )
    pseudo, _ = refine_agentic_spans(
        timeline,
        spans,
        runners,
        EventLog(tmp_path / "pseudo.jsonl"),
        ladder=ladder,
        pseudo_safe=True,
    )

    assert [span["driver_mode"] for span in general] == [
        "auto_commit",
        "auto_commit",
    ]
    assert pseudo[0]["driver_mode"] == "review"
    assert pseudo[0]["pseudo_gate_rejection"] == "g2_independence"
    assert pseudo[1]["driver_mode"] == "auto_commit"
    assert "pseudo_gate_rejection" not in pseudo[1]
    assert pseudo[0]["ref_segments"] == timeline["spans"][0]["ref_segments"]


def test_live_context_from_set_accepts_explicit_fiber_gate(monkeypatch):
    from core.result import Ok
    from workspaces.alignment_prototype import fp_placement_refine, mert_store

    mix = type("Mix", (), {"end_s": [60.0]})()
    monkeypatch.setattr(
        mert_store,
        "load_bb12_mert",
        lambda set_id: Ok((set_id, mix, {})),
    )
    monkeypatch.setattr(fp_placement_refine, "find_aligning_dir", lambda _: None)

    ctx = LiveContext.from_set("pool", [], apply_fiber_gate=True)

    assert ctx is not None
    assert ctx.apply_fiber_gate is True


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


def test_regular_stem_keeps_heaviest_cluster_over_fp():
    """bb12_42w5: regular stem — do NOT prefer fp over mert+surprise.

    fp@3519 is near audible onset but regresses official GT set_start (~3477);
    place-median SOTA requires heaviest-cluster (surprise/mert ~3466).
    """
    b = _belief(
        _obs("fp", 3519.5, conf=0.8, prec=0.53),
        _obs("mert_decode", 3470.4, conf=0.7, prec=0.55),
        _obs("surprise", 3466.08, conf=1.0, prec=0.45),
        stem="regular",
    )
    top = b.best()
    assert top is not None
    assert "fp" not in top.probes
    assert abs(top.set_start_s - 3466.08) < 5.0


def test_weak_fp_cluster_does_not_steal_strong_mert_pileup():
    """A low-weight stray fp diagonal must not beat a much heavier mert cluster."""
    b = _belief(
        _obs("fp", 100.0, conf=0.2, prec=0.53),  # weight ≈ 0.106
        _obs("mert_decode", 400.0, conf=1.0, prec=0.55),  # 0.55
        _obs("surprise", 402.0, conf=1.0, prec=0.45),  # 0.45 → pileup ≈ 1.0
        stem="instrumental",
    )
    top = b.best()
    assert top is not None
    assert "fp" not in top.probes
    assert 400.0 <= top.set_start_s <= 402.0


def test_fp_preferred_over_cue_prior_surprise_overwrite():
    """bb12_39 instrumental: classical fp @ GT; cue+surprise piled up ~46s off."""
    b = _belief(
        _obs("fp", 3210.7, conf=0.8, prec=0.53),
        _obs("mert_decode", 3256.14, conf=0.7, prec=0.55),
        _obs("cue_prior", 3256.8, conf=0.6, prec=0.50),
        _obs("surprise", 3256.8, conf=1.0, prec=0.45),
        stem="instrumental",
    )
    top = b.best()
    assert top is not None
    assert "fp" in top.probes
    assert abs(top.set_start_s - 3210.7) < 1.0


def test_instr_fp_preferred_over_mert_surprise_pileup():
    """Instrumental decoder_wall: competitive fp beats mert+surprise pile-up."""
    b = _belief(
        _obs("fp", 3399.8, conf=0.8, prec=0.53),
        _obs("mert_decode", 3480.0, conf=0.7, prec=0.55),
        _obs("surprise", 3481.0, conf=1.0, prec=0.45),
        stem="instrumental",
    )
    top = b.best()
    assert top is not None
    assert "fp" in top.probes
    assert abs(top.set_start_s - 3399.8) < 1.0


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
    # before the tail, plans are information-ordered; regulars now include
    # stem_hubert (after fp) so E1 can form fp+hubert G2 on the non-acap tail
    assert vocal[-2] == "stem_hubert"
    assert host.index("fp") < host.index("stem_hubert") < host.index("chroma_refine")
    assert host[-2] == "chroma_refine"


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


def test_ref_fp_for_span_prefers_claimed_stem_then_regular(monkeypatch):
    """Instrumental landmarks are often the only cached entry (bb11_34)."""
    from workspaces.alignment_prototype.agentic import live_runners as lr
    from workspaces.alignment_prototype.fp_index import FpKey

    calls: list[FpKey] = []

    def fake_load(key, **_kw):
        calls.append(key)
        if key.stem == "instrumental":
            return object()  # truthy stand-in
        return None

    monkeypatch.setattr("workspaces.alignment_prototype.fp_index.load", fake_load)
    # Also patch the late import path used inside ref_fp_for_span
    import workspaces.alignment_prototype.fp_index as fp_index

    monkeypatch.setattr(fp_index, "load", fake_load)

    hit = lr.ref_fp_for_span(
        {"recording_id": "g99m0t5", "claimed_stem": "instrumental"}
    )
    assert hit is not None
    assert calls[0] == FpKey("g99m0t5", "instrumental")
    assert all(c.stem != "regular" for c in calls)  # no fallback needed


def test_ref_fp_for_span_falls_back_to_regular(monkeypatch):
    from workspaces.alignment_prototype.agentic import live_runners as lr
    from workspaces.alignment_prototype.fp_index import FpKey

    calls: list[FpKey] = []

    def fake_load(key, **_kw):
        calls.append(key)
        return object() if key.stem == "regular" else None

    import workspaces.alignment_prototype.fp_index as fp_index

    monkeypatch.setattr(fp_index, "load", fake_load)

    hit = lr.ref_fp_for_span({"recording_id": "abc", "claimed_stem": "instrumental"})
    assert hit is not None
    assert calls == [FpKey("abc", "instrumental"), FpKey("abc", "regular")]


def test_ref_fp_for_span_abstains_acappella_allows_regular(monkeypatch):
    from workspaces.alignment_prototype.agentic import live_runners as lr
    from workspaces.alignment_prototype.fp_index import FpKey

    calls: list[FpKey] = []

    def fake_load(key, **_kw):
        calls.append(key)
        return object() if key.stem == "instrumental" else None

    import workspaces.alignment_prototype.fp_index as fp_index

    monkeypatch.setattr(fp_index, "load", fake_load)
    assert (
        lr.ref_fp_for_span({"recording_id": "abc", "claimed_stem": "acappella"}) is None
    )
    assert calls == []  # acappella must not touch the index
    hit = lr.ref_fp_for_span({"recording_id": "abc", "claimed_stem": "regular"})
    assert hit is not None
    assert calls[0] == FpKey("abc", "instrumental")


def test_live_hubert_can_be_skipped(monkeypatch):
    from workspaces.alignment_prototype.agentic.live_runners import LiveContext

    monkeypatch.setenv("AGENTIC_LIVE_SKIP_HUBERT", "1")
    ctx = LiveContext(set_id="set")
    LiveContext._load_hubert(ctx, Path("/unused"), [])
    assert "stem_hubert" not in ctx.loaded
    assert ctx.mix_hub is None
