"""The staged alignment spine: typed stage seam + Pipeline / AblationSpec.

Six stages (`STAGES`), each a pluggable slot. A `Pipeline` binds one impl name
per stage; an `AblationSpec` describes a matrix of actor impls to sweep at one
grain (isolate vs compose). Both carry a deterministic `config_hash` so the
ablation ledger (`runner.py`) is reproducible.

This module owns *types only* — no impls, no I/O. Impls register in
`registry.py`; the runner (`runner.py`) executes matrices via injected backends.
The stage Protocols are the *target* contract; today only Actor + Bridge carry
multiple impls (see `docs/pipeline_ablation_framework.md` §0 honesty boundary).
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from ..harness.contract import AlignmentResult

# --------------------------------------------------------------------------- #
# the spine                                                                    #
# --------------------------------------------------------------------------- #

STAGES: tuple[str, ...] = (
    "perception",
    "identity",
    "placement",
    "bridge",
    "actor",
    "assemble",
)

# The default impl bound to a stage when a pipeline does not override it. Frozen
# stages have exactly one impl today; the seam exists for later multiplication.
_STAGE_DEFAULT: dict[str, str] = {
    "perception": "std",
    "identity": "mert",
    "placement": "std",
    "bridge": "none",
    "actor": "viterbi",
    "assemble": "std",
}


def _stable_hash(payload: object) -> str:
    """A short, stable content hash (git-sha-length) over JSON-able payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class Pipeline:
    """One impl bound per stage. `name` is a human label; the `config_hash` is
    computed over the *bindings only*, so two identically-bound pipelines with
    different names hash equal (rename must not perturb the ledger)."""

    name: str
    bindings: Mapping[str, str]

    @staticmethod
    def of(name: str, **overrides: str) -> "Pipeline":
        bad = set(overrides) - set(STAGES)
        if bad:
            raise ValueError(f"unknown stage(s) {sorted(bad)}; valid: {STAGES}")
        bindings = {**_STAGE_DEFAULT, **overrides}
        return Pipeline(name=name, bindings=bindings)

    def config_hash(self) -> str:
        return _stable_hash({s: self.bindings[s] for s in STAGES})


# --------------------------------------------------------------------------- #
# ablation spec                                                                #
# --------------------------------------------------------------------------- #


class Grain(enum.Enum):
    """Isolate = swap one stage, hold the rest at oracle/GT, local referee.
    Compose = swap the stage, run the whole spine, end-to-end referee."""

    ISOLATE = "isolate"
    COMPOSE = "compose"


@dataclass(frozen=True)
class AblationSpec:
    """A matrix: sweep `actors` (impls at the Actor stage) over `sets`, at one
    `grain`. `base` names the canned pipeline whose non-actor stages are held
    fixed. LOSO/synthetic guards live in the runner (they need set→source
    resolution); the spec only validates shape."""

    grain: Grain
    sets: tuple[str, ...]
    actors: tuple[str, ...]
    base: str = "classical"
    split: str = "set"
    seed: int = 0
    bridge: str = "none"

    def __post_init__(self) -> None:
        if not self.sets:
            raise ValueError("AblationSpec needs at least one set")
        if not self.actors:
            raise ValueError("AblationSpec needs at least one actor impl")

    def config_hash(self) -> str:
        return _stable_hash(
            {
                "grain": self.grain.value,
                "sets": sorted(self.sets),
                "actors": sorted(self.actors),
                "base": self.base,
                "split": self.split,
                "seed": self.seed,
                "bridge": self.bridge,
            }
        )


# --------------------------------------------------------------------------- #
# stage Protocols (target contract; runner/adapters implement against these)   #
# --------------------------------------------------------------------------- #


@runtime_checkable
class StageImpl(Protocol):
    """Every registered impl carries a `name` and declares its `stage`."""

    name: str
    stage: str


@runtime_checkable
class Actor(Protocol):
    """The pluggable lever. Consumes the bound set + upstream stage outputs and
    emits per-span `AlignmentResult`s (or, for a MonolithicActor wrapping a
    legacy driver, writes a full PredictedTimeline). See adapters/."""

    name: str
    stage: str  # == "actor"


__all__ = [
    "STAGES",
    "Pipeline",
    "Grain",
    "AblationSpec",
    "StageImpl",
    "Actor",
    "AlignmentResult",
]
