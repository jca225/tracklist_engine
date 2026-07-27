"""Register a predicted timeline as a provenanced artifact (Brick 3).

Closes the read side of three law-audit gaps WITHOUT touching the aligner hot
path (the workspace phase policy forbids adding to the live kernel — this reads
``out/<set>_predicted_timeline.json`` after the fact, exactly as the scorer
does):

- **Law 1 (provenance_is_complete):** today the timeline is a bare
  ``out_path.write_text(json.dumps(...))`` with no RunOutput edge. Here it becomes
  a content-addressed ``Artifact`` with a ``Run`` + ``derivation`` edge from its
  run-config, so the file traces back to what produced it.
- **Law 3 (all_artifacts_verify):** the timeline is addressed by sha256, not by
  its mutable path.
- **Law 10 (identity/placement/structure are separate):** the shipped span
  collapses identity + set_start + ref_segments into one scalar ``confidence``.
  Here placement and structure are recorded as *separate per-axis observations*
  hanging off the SAME ``set-slot`` subject the identity belief uses (Brick 2b) —
  three axes on one subject, not one fused number.

**Law 12 honesty (decoder_consumes_posteriors_not_raw_margins) — NOT faked.**
``probe_proposals`` are raw disagreeing values (e.g. mert 34 s vs fp 338 s) chosen
by source-priority, not a calibrated posterior. Manufacturing a posterior here
would be the exact dishonesty the substrate exists to prevent, so placement is an
**Observation** (chosen value + competing proposals kept as the value payload,
decision-by-priority flagged ``uncalibrated_priority_decode``), never a fake
``AxisBelief``. The debt stays visible instead of being papered over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.provenance import (
    Artifact,
    ArtifactKind,
    ArtifactStore,
    ObservationStatus,
    ProvenanceRepository,
    SubjectRef,
)

_PROCESS = "timeline.register"
_VERSION = "1"


def _slot_subject(set_id: str, slot_label: str) -> SubjectRef:
    # MUST match Brick 2b (provenance_export._slot_subject) so the identity
    # belief and the placement/structure observations share one subject.
    return SubjectRef("set-slot", f"{set_id}::{slot_label}")


@dataclass(frozen=True)
class RegisteredTimeline:
    timeline_sha256: str
    n_spans: int
    n_placement: int
    n_structure: int
    n_uncalibrated_placement: int  # spans whose probes disagreed (Law 12 debt)
    n_no_structure: int


def _proposals_disagree(proposals: dict) -> bool:
    vals = {round(float(v), 1) for v in proposals.values() if v is not None}
    return len(vals) > 1


def register_timeline(
    set_id: str,
    timeline_path: str | Path,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str = "unknown",
) -> RegisteredTimeline:
    """Register ``timeline_path`` as a content-addressed artifact with a
    derivation edge, and record per-span placement + structure observations."""
    timeline_path = Path(timeline_path)
    raw = timeline_path.read_bytes()
    doc = json.loads(raw)
    spans = doc.get("spans", [])

    # The timeline itself — content-addressed (Law 3).
    timeline_art: Artifact = store.put_bytes(
        raw,
        kind=ArtifactKind.PREDICTED_TIMELINE,
        media_type="application/json",
        source_uri=str(timeline_path),
        source_system="alignment_prototype.infer",
    )
    # The run-config the timeline already embeds (set/train ids, bands, the
    # infer.py provenance stamp) — the parent the timeline derives FROM (Law 1).
    config_art: Artifact = store.put_bytes(
        json.dumps(
            {
                "set_id": set_id,
                "train_set_id": doc.get("train_set_id"),
                "band_s": doc.get("band_s"),
                "fp_placement": doc.get("fp_placement"),
                "embedded_provenance": doc.get("provenance"),
            },
            sort_keys=True,
        ).encode("utf-8"),
        kind=ArtifactKind.DIAGNOSTICS,
        media_type="application/json",
        source_system="alignment_prototype.infer",
    )

    run = repo.begin_run(
        process=_PROCESS,
        process_version=_VERSION,
        code_commit=code_commit,
        params_hash="0",
    )
    n_place = n_struct = n_uncal = n_nostruct = 0
    try:
        repo.add_input(run, config_art, role="run_config")
        repo.add_output(run, timeline_art, role="predicted_timeline")
        repo.derive(timeline_art, config_art, run, relation="registered_from")

        for s in spans:
            slot = str(s.get("slot_label"))
            subj = _slot_subject(set_id, slot)

            # --- PLACEMENT (Law 10 separation; Law 12 debt kept visible) -------
            proposals = s.get("probe_proposals") or {}
            disagree = _proposals_disagree(proposals)
            repo.record_observation(
                subject=subj,
                predicate="set_start_s",
                value={
                    "chosen_s": s.get("set_start_s"),
                    "source": s.get("start_source"),
                    "proposals": proposals,  # competing candidates preserved
                    "placement_gated": s.get("placement_gated"),
                },
                source=timeline_art,
                run=run,
                source_confidence=s.get("confidence"),
                diagnostic_code=("uncalibrated_priority_decode" if disagree else None),
            )
            n_place += 1
            n_uncal += int(disagree)

            # --- STRUCTURE (separate axis: the trajectory, not a scalar) --------
            segs = s.get("ref_segments") or []
            repo.record_observation(
                subject=subj,
                predicate="ref_trajectory",
                value={
                    "n_segments": len(segs),
                    "segments": segs,
                    "ref_decode_status": s.get("ref_decode_status"),
                    "fiber_ambiguity": s.get("fiber_ambiguity"),
                },
                source=timeline_art,
                run=run,
                source_confidence=s.get("ref_path_conf"),
                status=(
                    ObservationStatus.OBSERVED if segs else ObservationStatus.ABSTAINED
                ),
                diagnostic_code=None if segs else "no_structure_decoded",
            )
            n_struct += 1
            n_nostruct += int(not segs)

        repo.succeed(run)
    except Exception as exc:
        repo.fail(run, {"error": repr(exc)})
        raise

    return RegisteredTimeline(
        timeline_sha256=timeline_art.content_sha256,
        n_spans=len(spans),
        n_placement=n_place,
        n_structure=n_struct,
        n_uncalibrated_placement=n_uncal,
        n_no_structure=n_nostruct,
    )


def main() -> int:
    import argparse
    import subprocess

    from core.provenance import connect

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--timeline",
        default=None,
        help="timeline JSON (default: out/<set_id>_predicted_timeline_lt.json)",
    )
    ap.add_argument("--db-root", default=None)
    args = ap.parse_args()

    here = Path(__file__).parent
    timeline = Path(
        args.timeline or here / "out" / f"{args.set_id}_predicted_timeline_lt.json"
    )
    db_root = Path(args.db_root or here / "out" / "provenance" / args.set_id)
    db_root.mkdir(parents=True, exist_ok=True)

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    repo = ProvenanceRepository(conn)
    reg = register_timeline(
        args.set_id, timeline, store=store, repo=repo, code_commit=commit
    )

    print(f"registered timeline — set {args.set_id}")
    print(f"  timeline sha256        : {reg.timeline_sha256[:16]}…")
    print(f"  spans                  : {reg.n_spans}")
    print(
        f"  placement observations : {reg.n_placement}  "
        f"({reg.n_uncalibrated_placement} uncalibrated priority-decode)"
    )
    print(
        f"  structure observations : {reg.n_structure}  "
        f"({reg.n_no_structure} no structure decoded)"
    )
    print(f"  provenance DB          : {db_root}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
