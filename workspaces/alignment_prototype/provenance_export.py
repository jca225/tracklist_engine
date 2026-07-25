"""Express the aligner's identity decisions through the provenance substrate.

Brick 2b of the migration: take the real tracklist spine (set_track_slots claims)
+ the aligner's per-slot decision and emit provenanced identity beliefs into
``core.provenance``. This is the bridge that makes the *shipped* identity path run
through Claim → Evidence → AxisBelief, which is what flips the law-audit verdicts
(4/5/7/10) from VIOLATED → PASS — not just the substrate's own unit demo.

Two honest facts it encodes:
- With today's **size-1 candidate pool**, the posterior is degenerate on the
  tokenizer claim, so the belief simply ACCEPTS the claim. That is decision #19
  made explicit: shipped "identity" == the tracklist claim. Nothing here fakes a
  capability the aligner does not have.
- A slot with no claimed recording (an ID) yields an **abstaining** belief
  (persisted), never a guessed recording.

When the parked audio EvidenceSource (open_set_identity) is wired, it contributes
CONTRADICTS/ SUPPORTS evidence and a non-degenerate posterior — the belief then
can override the claim, with the override fully provenanced. The substrate is
ready for that without being it.

Lives in the aligner package (imports ``core.provenance`` downward only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from core.provenance import (
    Artifact,
    ArtifactStore,
    Axis,
    AxisBelief,
    DecisionRule,
    EvidenceDirection,
    ObservationStatus,
    ProvenanceRepository,
    SubjectRef,
)

_PROCESS = "identity.export"
_VERSION = "1"


@dataclass(frozen=True)
class SlotClaim:
    """One tracklist-spine row: what the source *claimed* for a slot."""

    slot_label: str
    claimed_recording_id: Optional[str]
    claimed_stem: str = "regular"


def _slot_subject(set_id: str, slot_label: str) -> SubjectRef:
    return SubjectRef("set-slot", f"{set_id}::{slot_label}")


def export_identity_beliefs(
    set_id: str,
    claims: Sequence[SlotClaim],
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    source: Artifact,
    rule: DecisionRule,
    code_commit: str = "unknown",
    posterior_by_slot: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> list[AxisBelief]:
    """Emit a provenanced identity belief per slot.

    ``posterior_by_slot`` (slot_label → {recording_id: prob}) is the hook for a
    real audio EvidenceSource. When absent, the posterior is degenerate on the
    tokenizer claim (the honest current pipeline) — {claim: 1.0}, or empty (→
    abstain) for an ID slot.
    """
    run = repo.begin_run(
        process=_PROCESS,
        process_version=_VERSION,
        code_commit=code_commit,
        params_hash="0",
    )
    beliefs: list[AxisBelief] = []
    try:
        for sc in claims:
            slot = _slot_subject(set_id, sc.slot_label)

            # 1. Observation: the source SAID this recording (not truth). An ID
            #    slot (no claimed recording) is a persisted abstention, not a drop.
            repo.record_observation(
                subject=slot,
                predicate="claimed_recording",
                value=sc.claimed_recording_id,
                source=source,
                run=run,
                status=(
                    ObservationStatus.OBSERVED
                    if sc.claimed_recording_id
                    else ObservationStatus.ABSTAINED
                ),
                diagnostic_code=None
                if sc.claimed_recording_id
                else "unidentified_slot",
            )

            # 2. Claim: the hypothesis that this slot is the claimed recording.
            candidate = (
                SubjectRef("recording", sc.claimed_recording_id)
                if sc.claimed_recording_id
                else None
            )
            claim = repo.record_claim(
                axis=Axis.IDENTITY,
                subject=slot,
                predicate="is_recording",
                candidate=candidate,
                run=run,
                context={"claimed_stem": sc.claimed_stem, "source": "set_track_slots"},
            )

            # 3. Evidence: the tokenizer claim is a (weak) prior, nothing more.
            ev = repo.record_evidence(
                claim=claim,
                source_family="tokenizer_claim",
                direction=(
                    EvidenceDirection.SUPPORTS
                    if sc.claimed_recording_id
                    else EvidenceDirection.ABSTAINS
                ),
                run=run,
                native_score={"prior": 1.0},
            )

            # 4. Belief over candidates. Default = degenerate on the claim (the
            #    honest size-1 pool); a real audio source supplies a wider one.
            posterior: Mapping[str, float]
            if posterior_by_slot and sc.slot_label in posterior_by_slot:
                posterior = posterior_by_slot[sc.slot_label]
            elif sc.claimed_recording_id:
                posterior = {sc.claimed_recording_id: 1.0}
            else:
                posterior = {}

            beliefs.append(
                repo.record_belief(
                    subject=slot,
                    axis=Axis.IDENTITY,
                    posterior=posterior,
                    rule=rule,
                    run=run,
                    contributing_evidence_ids=[ev.evidence_id],
                )
            )
        repo.succeed(run)
    except Exception as exc:  # provenance must record the failure, not swallow it
        repo.fail(run, {"error": repr(exc)})
        raise
    return beliefs


# --- standalone runner over a real set's live spine -------------------------
#
# This is the "flip the law-audit verdict" step: the SHIPPED identity spine
# (`fetch_slot_rows` — the same rows the aligner decodes) is expressed through
# Claim → Evidence → AxisBelief and persisted to a queryable provenance DB. It
# does NOT touch the aligner hot path (the workspace phase policy forbids adding
# to the live kernel); it composes the substrate over the spine as its own stage.

# A `tlp*` recording_id is a sided-row placeholder (scrape row with no
# data-trackid — the "Rvmor gap"), NOT a canonical recording. Accepting it as
# identity would be the claim-as-truth error the substrate exists to expose, so
# it becomes an unresolved claim → abstaining belief.
_PLACEHOLDER_PREFIX = "tlp"

_DEFAULT_RULE = DecisionRule(
    decision_rule_id="identity-export-v1",
    axis=Axis.IDENTITY,
    version="1",
    minimum_posterior=0.6,
    maximum_entropy=0.9,
)


def _claims_from_rows(rows: Sequence[Mapping[str, object]]) -> list[SlotClaim]:
    claims: list[SlotClaim] = []
    for r in rows:
        rec = r.get("recording_id")
        rec_s = str(rec) if rec else None
        if rec_s and rec_s.startswith(_PLACEHOLDER_PREFIX):
            rec_s = None  # placeholder id == unresolved identity
        claims.append(
            SlotClaim(
                slot_label=str(r["slot_label"]),
                claimed_recording_id=rec_s,
                claimed_stem=str(r.get("claimed_stem", "regular")),
            )
        )
    return claims


def run_export_for_set(
    set_id: str, *, db_root: "Path", code_commit: str = "unknown"
) -> list[AxisBelief]:
    """Fetch a set's live tracklist spine and emit provenanced identity beliefs
    into a fresh provenance DB rooted at ``db_root``. Returns the beliefs."""
    from pathlib import Path as _Path  # local: keep module import cheap

    from core.provenance import ArtifactKind, ProvenanceRepository, connect
    from workspaces.alignment_prototype.infer import fetch_slot_rows

    db_root = _Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)
    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    repo = ProvenanceRepository(conn)

    rows = fetch_slot_rows(set_id)
    claims = _claims_from_rows(rows)

    # The tracklist spine itself is the provenance source artifact (content-
    # addressed): the beliefs are derived FROM this exact byte-sequence.
    import json as _json

    source = store.put_bytes(
        _json.dumps(rows, sort_keys=True, default=str).encode("utf-8"),
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="application/json",
        source_system="set_track_slots",
        source_uri=f"fetch_slot_rows://{set_id}",
    )
    return export_identity_beliefs(
        set_id,
        claims,
        store=store,
        repo=repo,
        source=source,
        rule=_DEFAULT_RULE,
        code_commit=code_commit,
    )


def main() -> int:
    import argparse
    import subprocess
    from pathlib import Path

    from core.provenance import Decision

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--db-root",
        default=None,
        help="provenance DB dir (default: workspaces/alignment_prototype/out/provenance/<set_id>)",
    )
    args = ap.parse_args()

    db_root = Path(
        args.db_root or Path(__file__).parent / "out" / "provenance" / args.set_id
    )
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    beliefs = run_export_for_set(args.set_id, db_root=db_root, code_commit=commit)
    accepted = [b for b in beliefs if b.decision == Decision.ACCEPTED]
    abstained = [b for b in beliefs if b.decision == Decision.UNRESOLVED]

    print(f"provenance identity export — set {args.set_id}")
    print(f"  spine slots      : {len(beliefs)}")
    print(f"  ACCEPTED (claim) : {len(accepted)}")
    print(f"  ABSTAINED (tlp)  : {len(abstained)}  <- placeholder / unresolved")
    print(f"  provenance DB    : {db_root}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
