"""Express the human GT through the provenance substrate (Brick 6, §7).

Reads the shipped ground-truth YAML (READ ONLY — `labeling/` is never touched),
registers the file as a content-addressed source Artifact, and emits one
``HumanLabelBundle`` + per-track, per-field ``HumanLabelAssertion``s. This
closes the substrate side of the two biggest law-audit gaps:

- **Law 8 (human_history_is_append_only):** the shipped GT writer
  ``DELETE``s + ``INSERT OR REPLACE``s in place (law_audit §8 — that verdict
  still stands); here every assertion is INSERT-only and a revision is a NEW
  row via ``supersedes_assertion_id``.
- **Law 9 (human_uncertainty_is_field_specific):** the shipped GT stores bare
  point values; here every field carries its OWN §7 uncertainty model —
  ``categorical_prior`` for identity-like fields, ``student_t`` for time
  coordinates, ``curve_error`` for curves — never one global scalar, never
  absent.

Subjects are ``SubjectRef("set-slot", f"{set_id}::{slot_label}")`` — identical
to Bricks 2b/3, so identity beliefs, placement/structure observations, and
human GT all unify on one subject.

Honesty notes:
- A GT row with ``id_source: abstain``, no ``track_id``, or a ``tlp*``
  placeholder id yields NO recording_id assertion — instead a persisted
  ABSTAINED observation (never a smuggled identity; Laws 4/6/7).
- A GT row with no ``slot_label`` (annotator helper rows like ``mix``) cannot
  address a set-slot subject; it is persisted as an ABSTAINED observation on a
  ``gt-row`` subject, not silently dropped (Law 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from core.provenance import (
    Artifact,
    ArtifactKind,
    ArtifactStore,
    HumanLabelAssertion,
    HumanLabelBundle,
    ObservationStatus,
    ProvenanceRepository,
    SubjectRef,
)
from core.provenance.types import Json

_PROCESS = "human_gt.import"
_VERSION = "1"

# A `tlp*` track_id is a sided-row placeholder (scrape row with no
# data-trackid), NOT a canonical recording — same rule as provenance_export.
_PLACEHOLDER_PREFIX = "tlp"

# Shipped GT fixtures (read-only ground truth per set).
_FIXTURES: dict[str, str] = {
    "2nvzlh2k": "labeling/fixtures/bb11_ground_truth.yaml",  # BB11
    "1fsnxchk": "labeling/fixtures/bb12_ground_truth.yaml",  # BB12
}

# --- §7 field-specific uncertainty models (Law 9 — the load-bearing honesty) --
# GT YAML key -> (assertion field, uncertainty model). Each field gets a
# DIFFERENT model appropriate to what a human can actually resolve in Ableton:
# identity picks are near-certain categorical choices; time edges carry
# fat-tailed ~0.25 s placement error; curves carry per-control-point error.
_FIELD_MODELS: dict[str, tuple[str, Json]] = {
    "track_id": (
        "recording_id",
        {"kind": "categorical_prior", "mean": 0.99, "effective_sample_size": 20},
    ),
    "claimed_stem": (
        "claimed_stem",
        {"kind": "categorical_prior", "mean": 0.97, "effective_sample_size": 12},
    ),
    "set_start_s": ("mix_start_s", {"kind": "student_t", "scale_s": 0.25, "df": 4}),
    "set_end_s": ("mix_end_s", {"kind": "student_t", "scale_s": 0.25, "df": 4}),
    "ref_start_s": ("ref_start_s", {"kind": "student_t", "scale_s": 0.25, "df": 4}),
    "ref_end_s": ("ref_end_s", {"kind": "student_t", "scale_s": 0.25, "df": 4}),
    "tempo_ratio": (
        "tempo_ratio",
        {"kind": "curve_error", "control_point_scale": 0.003},
    ),
    "pitch_shift_semi": (
        # Integer semitone read off the Ableton clip — a rounding-style
        # categorical, tighter than the identity prior.
        "pitch_shift_semi",
        {"kind": "categorical_prior", "mean": 0.995, "effective_sample_size": 50},
    ),
    "gain_curve": (
        "gain_curve",
        {"kind": "curve_error", "control_point_scale": 0.02},
    ),
}


def _repo_root() -> Path:
    """Walk up to the repo root (marker: labeling/fixtures) — no hardcoded
    parent depth, so a file move cannot silently break the fixture paths."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "labeling" / "fixtures").is_dir():
            return parent
    raise SystemExit("cannot locate repo root (no labeling/fixtures ancestor)")


def _slot_subject(set_id: str, slot_label: str) -> SubjectRef:
    # MUST match Bricks 2b/3 so all axes + human GT unify on one subject.
    return SubjectRef("set-slot", f"{set_id}::{slot_label}")


@dataclass(frozen=True)
class ImportedGt:
    bundle: HumanLabelBundle
    assertions: list[HumanLabelAssertion]
    n_rows: int
    n_identity_abstain: int  # rows persisted as abstained recording_id obs
    n_unlabeled_rows: int  # rows with no slot_label (persisted, not dropped)


def _canonical_track_id(row: Mapping[str, Any]) -> Optional[str]:
    """The row's recording id iff it is assertable: present, id_source not an
    abstain, and not a scrape placeholder."""
    tid = row.get("track_id")
    if not tid or row.get("id_source") == "abstain":
        return None
    tid_s = str(tid)
    if tid_s.startswith(_PLACEHOLDER_PREFIX):
        return None
    return tid_s


def import_gt_bundle(
    set_id: str,
    gt: Mapping[str, Any],
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    source: Artifact,
    code_commit: str = "unknown",
) -> ImportedGt:
    """Emit one HumanLabelBundle + per-track per-field assertions from a parsed
    GT document. Append-only: re-running mints a NEW bundle, never an overwrite."""
    run = repo.begin_run(
        process=_PROCESS,
        process_version=_VERSION,
        code_commit=code_commit,
        params_hash="0",
    )
    assertions: list[HumanLabelAssertion] = []
    n_id_abstain = n_unlabeled = 0
    try:
        repo.add_input(run, source, role="ground_truth_yaml")
        bundle = repo.record_human_bundle(
            set_id=set_id,
            source=source,
            annotator_id=str(gt.get("annotated_by", "unknown")),
            run=run,
        )
        rows = list(gt.get("tracks", []))
        for row in rows:
            slot_label = row.get("slot_label")
            if not slot_label:
                # Annotator helper rows ("mix", …) address no slot — persist
                # the drop as a diagnostic, never silently (Law 6). No numeric
                # coordinates in the payload (Law 11).
                repo.record_observation(
                    subject=SubjectRef(
                        "gt-row", f"{set_id}::{row.get('track', '(unnamed)')}"
                    ),
                    predicate="gt_row_without_slot",
                    value={"track": row.get("track")},
                    source=source,
                    run=run,
                    status=ObservationStatus.ABSTAINED,
                    diagnostic_code="no_slot_label",
                )
                n_unlabeled += 1
                continue
            subject = _slot_subject(set_id, str(slot_label))

            recording_id = _canonical_track_id(row)
            if recording_id is None:
                # An unidentified/placeholder row asserts NO identity — the
                # abstention is persisted, never a guessed recording (Law 7).
                repo.record_observation(
                    subject=subject,
                    predicate="gt_recording_id",
                    value=None,
                    source=source,
                    run=run,
                    status=ObservationStatus.ABSTAINED,
                    diagnostic_code=(
                        "placeholder_track_id"
                        if str(row.get("track_id", "")).startswith(_PLACEHOLDER_PREFIX)
                        else "gt_identity_abstain"
                    ),
                )
                n_id_abstain += 1

            for gt_key, (field, model) in _FIELD_MODELS.items():
                if gt_key == "track_id":
                    value: Any = recording_id
                    if value is None:
                        continue  # handled above as a persisted abstention
                else:
                    if gt_key not in row or row[gt_key] is None:
                        continue  # absent field -> no assertion (nothing faked)
                    value = row[gt_key]
                assertions.append(
                    repo.record_human_assertion(
                        bundle=bundle,
                        subject=subject,
                        field=field,
                        observed_value=value,
                        uncertainty_model=model,
                        run=run,
                    )
                )
        repo.succeed(run)
    except Exception as exc:  # provenance must record the failure, not swallow it
        repo.fail(run, {"error": repr(exc)})
        raise
    return ImportedGt(
        bundle=bundle,
        assertions=assertions,
        n_rows=len(rows),
        n_identity_abstain=n_id_abstain,
        n_unlabeled_rows=n_unlabeled,
    )


def run_gt_import_for_set(
    set_id: str,
    *,
    db_root: Path,
    gt_path: Optional[Path] = None,
    code_commit: str = "unknown",
) -> ImportedGt:
    """Read a set's shipped GT YAML (read-only), register it as a
    content-addressed Artifact, and import it as a human-label bundle into the
    provenance DB rooted at ``db_root`` (the same DB the other adapters use)."""
    import yaml  # local: keep module import cheap

    from core.provenance import connect

    if gt_path is None:
        rel = _FIXTURES.get(set_id)
        if rel is None:
            raise SystemExit(
                f"no shipped GT fixture known for set {set_id!r}; pass --gt"
            )
        gt_path = _repo_root() / rel
    gt_path = Path(gt_path)
    raw = gt_path.read_bytes()
    gt = yaml.safe_load(raw)
    if str(gt.get("set_id")) != set_id:
        raise SystemExit(
            f"GT file {gt_path} is for set {gt.get('set_id')!r}, not {set_id!r}"
        )

    db_root = Path(db_root)
    db_root.mkdir(parents=True, exist_ok=True)
    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    repo = ProvenanceRepository(conn)

    # The GT YAML itself is the provenance source artifact (content-addressed):
    # the assertions are derived FROM this exact byte-sequence.
    source = store.put_bytes(
        raw,
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="application/x-yaml",
        source_uri=str(gt_path),
        source_system="labeling.ground_truth",
    )
    return import_gt_bundle(
        set_id, gt, store=store, repo=repo, source=source, code_commit=code_commit
    )


def main() -> int:
    import argparse
    import subprocess
    from collections import Counter

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--gt", default=None, help="GT YAML path (default: shipped fixture for set)"
    )
    ap.add_argument(
        "--db-root",
        default=None,
        help="provenance DB dir (default: alignment/out/provenance/<set_id>)",
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

    imported = run_gt_import_for_set(
        args.set_id,
        db_root=db_root,
        gt_path=Path(args.gt) if args.gt else None,
        code_commit=commit,
    )
    by_field = Counter(a.field for a in imported.assertions)
    kinds = sorted({str(a.uncertainty_model.get("kind")) for a in imported.assertions})

    print(f"human GT provenance import — set {args.set_id}")
    print(f"  bundle             : {imported.bundle.bundle_id}")
    print(f"  annotator          : {imported.bundle.annotator_id}")
    print(f"  GT rows            : {imported.n_rows}")
    print(f"  assertions         : {len(imported.assertions)}")
    for field, n in sorted(by_field.items()):
        print(f"    {field:<18}: {n}")
    print(f"  uncertainty kinds  : {', '.join(kinds)}")
    print(
        f"  identity abstains  : {imported.n_identity_abstain}  <- persisted, not guessed"
    )
    print(
        f"  unlabeled GT rows  : {imported.n_unlabeled_rows}  <- persisted, not dropped"
    )
    print(f"  provenance DB      : {db_root}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
