"""Executable ledger for the provenance-first §16 system laws.

Each law from ``docs/engine/dj_engine_pseudocode.md`` §16 is tracked here as a
parametrized case. The verdicts mirror ``docs/engine/law_audit.md`` (Phase 0
audit, 2026-07-25):

  * PASS laws run a (currently placeholder) guard and stay green — as each is
    hardened, replace ``_guard_*`` bodies with a real regression assertion so a
    future change that reintroduces the violation FAILS ``make check``.
  * Non-PASS laws ``xfail`` with the phase that closes them, so the suite stays
    green today while the ledger shows exactly what remains. When a convergence
    phase satisfies a law, flip its ``verdict`` to ``PASS`` and write the guard.

This file is the mechanism the audit calls for: poison relocation (decision #19)
slipped past Crush precisely because there was no executable fence. See
``docs/provenance_engine_convergence_plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DOC = REPO_ROOT / "docs" / "engine" / "law_audit.md"


@dataclass(frozen=True)
class Law:
    number: int
    name: str
    verdict: str  # PASS | PARTIAL | VIOLATED | N/A-NOT-BUILT
    phase: str  # phase that closes it ("—" if already PASS)
    note: str


# Mirror of docs/engine/law_audit.md — keep in sync when a phase closes a law.
LAWS: tuple[Law, ...] = (
    Law(
        1,
        "provenance_is_complete",
        "VIOLATED",
        "P1",
        "features/timelines/models written with no Run/Derivation record",
    ),
    Law(
        2,
        "provenance_is_acyclic",
        "N/A-NOT-BUILT",
        "P1->P3",
        "no derivation graph exists to be acyclic",
    ),
    Law(
        3,
        "all_artifacts_verify",
        "PARTIAL",
        "P1",
        "sha columns exist but everything fetched/joined by mutable path",
    ),
    Law(
        4,
        "no_source_key_is_canonical_recording_id",
        "PARTIAL",
        "P1",
        "COALESCE(recording_id, track_id) leaks the scrape key as identity",
    ),
    Law(
        5,
        "no_path_was_used_as_identity",
        "PARTIAL",
        "P1",
        "labeling PASS (Crush); aligner VIOLATED (size-1 spine pool)",
    ),
    Law(
        6,
        "every_unknown_parser_row_has_diagnostic",
        "PARTIAL",
        "P0->P1",
        "tokenizer silently continues past unkeyable rows",
    ),
    Law(
        7,
        "all_abstentions_are_persisted",
        "PASS",
        "—",
        "abstains stored as NULL-recording_id rows in set_ground_truth",
    ),
    Law(
        8,
        "human_history_is_append_only",
        "VIOLATED",
        "P1",
        "re-export DELETEs + INSERT OR REPLACEs GT in place",
    ),
    Law(
        9,
        "human_uncertainty_is_field_specific",
        "VIOLATED",
        "P1",
        "GT stored as bare point values, zero uncertainty",
    ),
    Law(
        10,
        "identity_placement_structure_are_separate",
        "PARTIAL",
        "P2",
        "collapse into one span + one confidence; no AxisBelief",
    ),
    Law(
        11,
        "no_unknown_coordinate_equals_zero",
        "VIOLATED",
        "P2",
        "unknown cue -> real 0.0; GT coords are NOT NULL",
    ),
    Law(
        12,
        "decoder_consumes_posteriors_not_raw_margins",
        "VIOLATED",
        "P2",
        "Viterbi over raw cosine margins; calibration deferred",
    ),
    Law(
        13,
        "every_model_has_training_snapshot",
        "VIOLATED",
        "P3",
        "heads train on live GT/MERT; no frozen TrainingSnapshot",
    ),
    Law(
        14,
        "every_model_has_code_and_environment",
        "VIOLATED",
        "P3",
        "checkpoints save weights+args, no code sha / env / dep-lock",
    ),
    Law(
        15,
        "no_round_consumes_its_own_outputs",
        "PARTIAL",
        "P3",
        "set-split proxy only; no round object / cycle guard",
    ),
    Law(
        16,
        "pseudo_label_graph_is_acyclic",
        "VIOLATED",
        "P3",
        "pseudo-labels are a flat boolean flag, no id/parent/round",
    ),
    Law(
        17,
        "rejected_rounds_are_reproducible",
        "VIOLATED",
        "P3",
        "no round entity; rejected attempts discarded",
    ),
    Law(
        18,
        "evaluation_is_per_set_and_per_axis",
        "PASS",
        "—",
        "scorer reports id/placement/structure per set, per stem",
    ),
    Law(
        19,
        "not_claims_corpus_calibration_from_two_sets",
        "PASS",
        "—",
        "docs+code label n=2 LOSO development-only",
    ),
    Law(
        20,
        "reconstruction_does_not_certify_identity",
        "PASS",
        "—",
        "recon is a placement feature + stopping signal only",
    ),
    Law(
        21,
        "every_published_value_is_explainable",
        "PARTIAL",
        "P3",
        "input-provenance stamp exists; no per-axis explain()",
    ),
)


def test_registry_matches_audit_count():
    """The ledger tracks all 21 laws and the audit doc it mirrors exists."""
    assert len(LAWS) == 21
    assert len({law.number for law in LAWS}) == 21
    assert AUDIT_DOC.exists(), f"audit doc missing: {AUDIT_DOC}"


# --- PASS-law guards -------------------------------------------------------
# Placeholders today. Harden each into a real regression assertion so a change
# that reintroduces the violation turns the suite red. Tracked as Phase 0 debt.


def _guard_law_7() -> None:
    pass  # TODO(P1): assert an abstaining GT bind persists a NULL-recording_id row


def _guard_law_18() -> None:
    pass  # TODO: assert scorer emits per-set AND per-axis blocks


def _guard_law_19() -> None:
    pass  # TODO: assert no code/doc asserts corpus calibration from n=2


def _guard_law_20() -> None:
    pass  # TODO: assert reconstruction score never enters the identity decode


_GUARDS = {7: _guard_law_7, 18: _guard_law_18, 19: _guard_law_19, 20: _guard_law_20}


@pytest.mark.parametrize("law", LAWS, ids=lambda law: f"law{law.number:02d}_{law.name}")
def test_system_law(law: Law):
    if law.verdict != "PASS":
        pytest.xfail(f"{law.verdict} -> closes in {law.phase}: {law.note}")
    guard = _GUARDS.get(law.number)
    assert guard is not None, f"PASS law {law.number} has no guard registered"
    guard()
