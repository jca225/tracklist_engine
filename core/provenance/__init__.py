"""Provenance-first substrate (§2/§4 of the engine contract).

Brick 1 of the migration in docs/provenance_engine_convergence_plan.md: the
content-addressed artifact store + run/derivation/observation repository that
Claims, Evidence, and AxisBeliefs (Brick 2) reference. Nothing else in the new
engine can be provenanced until these exist.
"""

from .decisions import decide, entropy
from .explain import (
    AxisView,
    HumanAssertionView,
    LineageNode,
    ObservationView,
    SubjectExplanation,
    explain_subject,
    format_lineage,
    format_subject,
    lineage,
)
from .laws import LawResult, LawVerdict, check_laws, format_law_results
from .repository import ProvenanceCycle, ProvenanceRepository
from .store import ArtifactStore, connect
from .types import (
    Artifact,
    ArtifactKind,
    Axis,
    AxisBelief,
    Claim,
    Decision,
    DecisionRule,
    Derivation,
    Evidence,
    EvidenceDirection,
    HumanLabelAssertion,
    HumanLabelBundle,
    Interval,
    Observation,
    ObservationStatus,
    Run,
    RunStatus,
    SubjectRef,
)

__all__ = [
    "ArtifactStore",
    "connect",
    "ProvenanceRepository",
    "ProvenanceCycle",
    "decide",
    "entropy",
    "check_laws",
    "format_law_results",
    "LawResult",
    "LawVerdict",
    "explain_subject",
    "lineage",
    "format_subject",
    "format_lineage",
    "SubjectExplanation",
    "AxisView",
    "HumanAssertionView",
    "ObservationView",
    "LineageNode",
    "HumanLabelAssertion",
    "HumanLabelBundle",
    "Artifact",
    "ArtifactKind",
    "Axis",
    "AxisBelief",
    "Claim",
    "Decision",
    "DecisionRule",
    "Derivation",
    "Evidence",
    "EvidenceDirection",
    "Interval",
    "Observation",
    "ObservationStatus",
    "Run",
    "RunStatus",
    "SubjectRef",
]
