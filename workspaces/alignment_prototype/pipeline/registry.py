"""Impl registry + the four canned pipelines.

`Registry` maps `(stage, name) -> impl`. An impl is anything carrying `.stage`
and `.name` (behavior is attached by `adapters/`; the registry only indexes).
`default_registry()` is the process-wide set of impls the runner sees;
`canned_pipelines()` are the four named stage-bindings (`classical`/`agentic`/
`ml`/`trm`) that reproduce today's drivers as pipelines.

Frozen stages (perception/identity/placement/assemble) hold a single impl by
policy — the seam exists but is not multiplied (see the framework spec §0).
"""

from __future__ import annotations

from dataclasses import dataclass

from .stages import STAGES, Pipeline


@dataclass(frozen=True)
class ImplRef:
    """Minimal impl descriptor: which stage it plugs into and its name. Adapters
    subclass/attach callable behavior; the registry indexes on these two fields."""

    stage: str
    name: str


class Registry:
    """Indexes impls by `(stage, name)`. Duplicate names within a stage are a
    hard error — silent shadowing is exactly the kind of drift this framework
    exists to prevent."""

    def __init__(self) -> None:
        self._by_stage: dict[str, dict[str, object]] = {s: {} for s in STAGES}

    def register(self, impl: object) -> object:
        stage = getattr(impl, "stage", None)
        name = getattr(impl, "name", None)
        if stage not in self._by_stage:
            raise ValueError(f"unknown stage {stage!r}; valid: {STAGES}")
        if name in self._by_stage[stage]:
            raise ValueError(f"impl {name!r} already registered for stage {stage!r}")
        self._by_stage[stage][name] = impl
        return impl

    def get(self, stage: str, name: str) -> object:
        if stage not in self._by_stage:
            raise ValueError(f"unknown stage {stage!r}; valid: {STAGES}")
        bucket = self._by_stage[stage]
        if name not in bucket:
            raise KeyError(
                f"no impl {name!r} for stage {stage!r}; have {sorted(bucket)}"
            )
        return bucket[name]

    def names(self, stage: str) -> list[str]:
        if stage not in self._by_stage:
            raise ValueError(f"unknown stage {stage!r}; valid: {STAGES}")
        return sorted(self._by_stage[stage])


# --------------------------------------------------------------------------- #
# the default registry                                                         #
# --------------------------------------------------------------------------- #

# (stage, name) inventory. Actor + bridge are the live pluggable surface; the
# rest are single-impl frozen stages. Adapters (adapters/*.py) attach behavior
# to these names; until then they index as bare ImplRefs.
_DEFAULT_IMPLS: tuple[tuple[str, str], ...] = (
    ("perception", "std"),
    ("identity", "mert"),
    ("placement", "std"),
    ("placement", "agentic"),  # agentic is a placement intervention (see spec §1)
    ("bridge", "none"),
    ("bridge", "linear"),
    ("actor", "viterbi"),
    ("actor", "agentic"),
    ("actor", "trajectory_decoder"),
    ("actor", "trm"),
    ("assemble", "std"),
)


def default_registry() -> Registry:
    r = Registry()
    for stage, name in _DEFAULT_IMPLS:
        r.register(ImplRef(stage=stage, name=name))
    return r


def canned_pipelines() -> list[Pipeline]:
    """The four named pipelines that reproduce today's drivers as stage bindings.
    `agentic` overrides BOTH placement and actor — it is a placement champion,
    not merely a decode swap (framework spec §1)."""
    return [
        Pipeline.of("classical", bridge="none", actor="viterbi"),
        Pipeline.of("agentic", placement="agentic", actor="agentic"),
        Pipeline.of("ml", bridge="linear", actor="trajectory_decoder"),
        Pipeline.of("trm", bridge="linear", actor="trm"),
    ]


__all__ = ["ImplRef", "Registry", "default_registry", "canned_pipelines"]
