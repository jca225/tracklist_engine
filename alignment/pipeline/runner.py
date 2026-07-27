"""The ablation runner: expand a spec into a guarded plan, execute it through
injected backends, flag learned-nothing, and append to the JSONL ledger.

The *logic* here is pure and unit-tested with fakes (see tests/test_runner.py):
LOSO + synthetic guards, baseline auto-injection, the learned-nothing flag,
deterministic hashing, ledger append. The *heavy* work (producing a timeline /
pred-segments, scoring against real GT) lives behind two injected protocols so
this module never imports audio:

    AlignBackend.run(pipeline, RunContext) -> artifact
    Scorer.score(artifact, RunContext)     -> metrics dict (must carry "strict")

Production wires the subprocess-backed impls in `adapters/scorers.py`; tests
inject fakes. The runner NEVER calls a score-variant — the canonical scorer is
the frozen referee (framework spec §3).
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from ..drivers.base import GT_BY_SET  # single source of truth for GT set ids
from .registry import canned_pipelines
from .stages import AblationSpec, Grain, Pipeline

# eval-set ids that begin with this are synthetic — train-only, never scored
SYNTHETIC_PREFIX = "synthv2"

# the two decode-stage baselines the bake-off mandates print beside every run
CONTROL_ACTOR = "raw_argmax"  # raw match-similarity argmax (no-model control)
CONV_BASELINE_ACTOR = "conv_viterbi"  # the current conv+Viterbi scaffold


# --------------------------------------------------------------------------- #
# injected backend protocols                                                  #
# --------------------------------------------------------------------------- #


class AlignBackend(Protocol):
    def run(self, pipeline: Pipeline, ctx: "RunContext") -> object: ...


class Scorer(Protocol):
    def score(self, artifact: object, ctx: "RunContext") -> Mapping[str, float]: ...


# --------------------------------------------------------------------------- #
# plan types                                                                  #
# --------------------------------------------------------------------------- #


class Role(enum.Enum):
    CONTROL = "control"  # the no-model null; a candidate must beat it
    BASELINE = "baseline"  # the incumbent to beat (conv_viterbi / classical)
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class RunContext:
    set_id: str  # eval set
    source_set_id: str  # train/transfer set (LOSO: != set_id)
    grain: Grain
    split: str
    seed: int
    pipeline: Pipeline


@dataclass(frozen=True)
class Cell:
    role: Role
    ctx: RunContext


@dataclass(frozen=True)
class RunPlan:
    spec_hash: str
    cells: tuple[Cell, ...]


@dataclass(frozen=True)
class RunRow:
    config_hash: str
    spec_hash: str
    git_sha: str
    now: str
    grain: str
    split: str
    seed: int
    set_id: str
    source_set_id: str
    pipeline: str
    actor: str
    role: str
    metrics: Mapping[str, float]
    learned_nothing: bool = False

    def to_json(self) -> dict:
        d = asdict(self)
        d["metrics"] = dict(self.metrics)
        return d


# --------------------------------------------------------------------------- #
# source resolution (LOSO)                                                     #
# --------------------------------------------------------------------------- #


def _default_source(set_id: str) -> str:
    """LOSO source = the other complete GT set. With exactly two GT sets this is
    unambiguous; with more it is the leave-one-out complement (caller may pass an
    explicit `source_of`)."""
    others = sorted(s for s in GT_BY_SET if s != set_id)
    if not others:
        raise ValueError(f"no cross-set LOSO source for {set_id!r}")
    return others[0]


# --------------------------------------------------------------------------- #
# planning                                                                    #
# --------------------------------------------------------------------------- #


def _pipeline_for_actor(spec: AblationSpec, actor: str) -> Pipeline:
    """Build the pipeline for one actor impl. Compose-grain baselines that are
    whole canned pipelines (classical) are looked up by name; every other actor
    varies the `base` pipeline's actor (and bridge)."""
    return Pipeline.of(f"{spec.base}+{actor}", bridge=spec.bridge, actor=actor)


def _canned(name: str) -> Pipeline:
    for p in canned_pipelines():
        if p.name == name:
            return p
    raise KeyError(f"no canned pipeline named {name!r}")


def plan(spec: AblationSpec, *, source_of: Mapping[str, str] | None = None) -> RunPlan:
    """Expand a spec into a guarded, baseline-injected plan. Pure — no backends.

    Guards (framework spec §3): rejects synthetic eval sets and any cell whose
    train set equals its eval set (LOSO). Auto-injects the control + baseline
    rows exactly once each."""
    source_of = dict(source_of or {})

    # guard: no synthetic set may be scored
    for sid in spec.sets:
        if sid.startswith(SYNTHETIC_PREFIX):
            raise ValueError(
                f"synthetic set {sid!r} in eval — synthetic is train-only "
                f"(framework spec §3)"
            )

    # candidate actors, with control + conv baseline injected once (isolate)
    if spec.grain is Grain.ISOLATE:
        injected = [CONTROL_ACTOR, CONV_BASELINE_ACTOR]
        actors = injected + [a for a in spec.actors if a not in injected]
    else:
        actors = list(spec.actors)

    def role_for(actor: str) -> Role:
        if actor == CONTROL_ACTOR:
            return Role.CONTROL
        if actor == CONV_BASELINE_ACTOR:
            return Role.BASELINE
        return Role.CANDIDATE

    cells: list[Cell] = []
    for sid in spec.sets:
        source = source_of.get(sid, _default_source(sid))
        if source == sid:
            raise ValueError(
                f"LOSO violation: train set == eval set == {sid!r} "
                f"(never train and eval on the same mix)"
            )
        # compose-grain: prepend the classical whole-pipeline baseline
        if spec.grain is Grain.COMPOSE:
            cells.append(
                Cell(
                    Role.BASELINE,
                    RunContext(
                        sid,
                        source,
                        spec.grain,
                        spec.split,
                        spec.seed,
                        _canned("classical"),
                    ),
                )
            )
        for actor in actors:
            cells.append(
                Cell(
                    role_for(actor),
                    RunContext(
                        sid,
                        source,
                        spec.grain,
                        spec.split,
                        spec.seed,
                        _pipeline_for_actor(spec, actor),
                    ),
                )
            )
    return RunPlan(spec_hash=spec.config_hash(), cells=tuple(cells))


# --------------------------------------------------------------------------- #
# execution                                                                   #
# --------------------------------------------------------------------------- #


def run(
    spec: AblationSpec,
    *,
    backend: AlignBackend,
    scorer: Scorer,
    ledger_path: Path,
    git_sha: str,
    now: str,
    source_of: Mapping[str, str] | None = None,
) -> list[RunRow]:
    """Execute a spec: run every plan cell through backend+scorer, flag any
    candidate that does not beat its set's control on `strict`, append one ledger
    row per cell, and return the rows. `git_sha` and `now` are injected (never
    read from the clock) so runs are reproducible and testable."""
    the_plan = plan(spec, source_of=source_of)

    # score every cell, remembering the control's strict per eval set
    scored: list[tuple[Cell, Mapping[str, float]]] = []
    control_strict: dict[str, float] = {}
    for cell in the_plan.cells:
        artifact = backend.run(cell.ctx.pipeline, cell.ctx)
        metrics = scorer.score(artifact, cell.ctx)
        scored.append((cell, metrics))
        if cell.role is Role.CONTROL:
            control_strict[cell.ctx.set_id] = float(metrics.get("strict", 0.0))

    rows: list[RunRow] = []
    for cell, metrics in scored:
        ctx = cell.ctx
        ctrl = control_strict.get(ctx.set_id)
        learned_nothing = (
            cell.role is Role.CANDIDATE
            and ctrl is not None
            and float(metrics.get("strict", 0.0)) <= ctrl
        )
        rows.append(
            RunRow(
                config_hash=ctx.pipeline.config_hash(),
                spec_hash=the_plan.spec_hash,
                git_sha=git_sha,
                now=now,
                grain=ctx.grain.value,
                split=ctx.split,
                seed=ctx.seed,
                set_id=ctx.set_id,
                source_set_id=ctx.source_set_id,
                pipeline=ctx.pipeline.name,
                actor=ctx.pipeline.bindings["actor"],
                role=cell.role.value,
                metrics=dict(metrics),
                learned_nothing=learned_nothing,
            )
        )

    _append_ledger(ledger_path, rows)
    return rows


def _append_ledger(path: Path, rows: list[RunRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_json(), sort_keys=True) + "\n")


__all__ = [
    "AlignBackend",
    "Scorer",
    "Role",
    "RunContext",
    "Cell",
    "RunPlan",
    "RunRow",
    "plan",
    "run",
    "SYNTHETIC_PREFIX",
    "CONTROL_ACTOR",
    "CONV_BASELINE_ACTOR",
]
