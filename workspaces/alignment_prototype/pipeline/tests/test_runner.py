"""Runner logic: LOSO + synthetic guards, baseline auto-injection, the
learned-nothing flag, ledger append, and determinism — all with fake backends
so no audio/GT is needed (framework spec §5)."""

from __future__ import annotations

import json

import pytest

from workspaces.alignment_prototype.pipeline import runner as run
from workspaces.alignment_prototype.pipeline.stages import AblationSpec, Grain

BB11, BB12 = "2nvzlh2k", "1fsnxchk"


# --- fakes ----------------------------------------------------------------- #


class FakeBackend:
    """Records the pipelines it ran; returns an opaque marker artifact."""

    def __init__(self):
        self.calls = []

    def run(self, pipeline, ctx):
        self.calls.append((pipeline.name, ctx.set_id, ctx.source_set_id))
        return {"pipeline": pipeline.name}


class FakeScorer:
    """Deterministic metrics keyed by actor name, so tests can force a candidate
    below/above the control."""

    def __init__(self, strict_by_actor):
        self.strict_by_actor = strict_by_actor

    def score(self, artifact, ctx):
        actor = ctx.pipeline.bindings["actor"]
        return {"strict": self.strict_by_actor[actor], "fiber": 0.5}


# --- plan: guards ---------------------------------------------------------- #


def test_plan_resolves_cross_set_source_for_loso():
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",))
    plan = run.plan(spec)
    cells = [c for c in plan.cells if c.role is run.Role.CANDIDATE]
    assert cells and all(c.ctx.source_set_id != c.ctx.set_id for c in cells)
    assert cells[0].ctx.source_set_id == BB12  # the other GT set


def test_plan_rejects_train_equals_eval():
    with pytest.raises(ValueError, match="LOSO"):
        run.plan(
            AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",)),
            source_of={BB11: BB11},  # forced self-source
        )


def test_plan_rejects_synthetic_set_in_eval():
    spec = AblationSpec(grain=Grain.ISOLATE, sets=("synthv2_0007",), actors=("trm",))
    with pytest.raises(ValueError, match="synthetic"):
        run.plan(spec)


# --- plan: baseline injection --------------------------------------------- #


def test_isolate_grain_auto_injects_control_and_conv_baseline():
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",))
    plan = run.plan(spec)
    roles = {c.role: c.ctx.pipeline.bindings["actor"] for c in plan.cells}
    assert roles[run.Role.CONTROL] == "raw_argmax"
    assert roles[run.Role.BASELINE] == "conv_viterbi"
    assert "trm" in [
        c.ctx.pipeline.bindings["actor"]
        for c in plan.cells
        if c.role is run.Role.CANDIDATE
    ]


def test_baselines_not_duplicated_if_user_lists_them():
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("raw_argmax", "trm"))
    plan = run.plan(spec)
    argmax_cells = [
        c for c in plan.cells if c.ctx.pipeline.bindings["actor"] == "raw_argmax"
    ]
    assert len(argmax_cells) == 1  # control injected once, not twice


def test_compose_grain_includes_classical_baseline():
    spec = AblationSpec(grain=Grain.COMPOSE, sets=(BB11,), actors=("ml",))
    plan = run.plan(spec)
    baselines = [c for c in plan.cells if c.role is run.Role.BASELINE]
    assert any(c.ctx.pipeline.name == "classical" for c in baselines)


# --- run: execution, flags, ledger ---------------------------------------- #


def test_run_flags_learned_nothing_when_candidate_not_beating_control(tmp_path):
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",))
    scorer = FakeScorer(
        {"raw_argmax": 0.30, "conv_viterbi": 0.42, "trm": 0.25}
    )  # trm below control
    rows = run.run(
        spec,
        backend=FakeBackend(),
        scorer=scorer,
        ledger_path=tmp_path / "ledger.jsonl",
        git_sha="deadbee",
        now="2026-07-18T00:00:00Z",
    )
    trm = [r for r in rows if r.actor == "trm"][0]
    assert trm.learned_nothing is True


def test_run_does_not_flag_when_candidate_beats_control(tmp_path):
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",))
    scorer = FakeScorer({"raw_argmax": 0.30, "conv_viterbi": 0.42, "trm": 0.55})
    rows = run.run(
        spec,
        backend=FakeBackend(),
        scorer=scorer,
        ledger_path=tmp_path / "ledger.jsonl",
        git_sha="deadbee",
        now="2026-07-18T00:00:00Z",
    )
    trm = [r for r in rows if r.actor == "trm"][0]
    assert trm.learned_nothing is False


def test_run_appends_one_valid_jsonl_row_per_cell(tmp_path):
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",))
    scorer = FakeScorer({"raw_argmax": 0.30, "conv_viterbi": 0.42, "trm": 0.55})
    ledger = tmp_path / "ledger.jsonl"
    rows = run.run(
        spec,
        backend=FakeBackend(),
        scorer=scorer,
        ledger_path=ledger,
        git_sha="deadbee",
        now="2026-07-18T00:00:00Z",
    )
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == len(rows) == 3  # control + baseline + candidate
    for line in lines:
        rec = json.loads(line)
        assert rec["git_sha"] == "deadbee"
        assert rec["grain"] == "isolate"
        assert "config_hash" in rec and "metrics" in rec


def test_run_is_deterministic_same_seed_same_metrics(tmp_path):
    spec = AblationSpec(grain=Grain.ISOLATE, sets=(BB11,), actors=("trm",), seed=7)
    scorer = FakeScorer({"raw_argmax": 0.30, "conv_viterbi": 0.42, "trm": 0.55})

    def go(path):
        return run.run(
            spec,
            backend=FakeBackend(),
            scorer=scorer,
            ledger_path=path,
            git_sha="deadbee",
            now="2026-07-18T00:00:00Z",
        )

    a = go(tmp_path / "a.jsonl")
    b = go(tmp_path / "b.jsonl")
    assert [r.metrics for r in a] == [r.metrics for r in b]
    assert [r.config_hash for r in a] == [r.config_hash for r in b]
