"""Leakage-safe LOSO calibration for placement/structure axis beliefs.

With only BB11 and BB12 labeled, calibration claims are development-only.  For
each held set this module fits on the *other* set and freezes a deliberately
low-capacity model:

    P(axis correct | source family fired) ~ Beta(1 + hits, 1 + misses)

Placement correctness is the canonical placement gate; structure correctness is
the canonical strict trajectory coverage gate. Sparse source families fall back
to the train-set global posterior. The resulting model is serializable,
content-addressed, and names its train/held sets. Same-set fit/apply is refused.

This is an offline producer. It never imports into the inference hot path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    AxisBelief,
    DecisionRule,
    ProvenanceRepository,
    Run,
    SubjectRef,
    connect,
)
from workspaces.alignment_prototype.belief_timeline import write_belief_bundle
from workspaces.alignment_prototype.harness.contract import AlignmentResult, RefSegment
from workspaces.alignment_prototype.placement_structure_beliefs import (
    CalibratedAxisProbability,
    record_placement_structure_beliefs,
)
from workspaces.alignment_prototype.score_timeline_vs_gt import SpanScore, score_spans

_PROCESS_FIT = "axis.calibrate_loso"
_PROCESS_INFER = "axis.infer_calibrated_loso"
_VERSION = "1"
_MODEL_SCHEMA = "source-beta-axis-calibration-v1"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "labeling" / "fixtures").is_dir():
            return parent
    raise RuntimeError("cannot locate repo root")


@dataclass(frozen=True)
class BetaCount:
    hits: int
    misses: int
    alpha_prior: float = 1.0
    beta_prior: float = 1.0

    @property
    def n(self) -> int:
        return self.hits + self.misses

    @property
    def mean(self) -> float:
        return (self.alpha_prior + self.hits) / (
            self.alpha_prior + self.beta_prior + self.n
        )


@dataclass(frozen=True)
class SourceAxisCalibration:
    axis: Axis
    train_set_id: str
    criterion: str
    by_family: Mapping[str, BetaCount]
    global_count: BetaCount
    minimum_family_n: int = 5
    calibration_status: str = "development_only"

    def probability(self, family: str) -> tuple[float, str]:
        count = self.by_family.get(family)
        if count is not None and count.n >= self.minimum_family_n:
            return count.mean, family
        return self.global_count.mean, "__global__"


@dataclass(frozen=True)
class LosoAxisModel:
    train_set_id: str
    held_set_id: str
    placement: SourceAxisCalibration
    structure: SourceAxisCalibration
    placement_success_s: float = 15.0
    structure_success: float = 0.8
    schema: str = _MODEL_SCHEMA
    intended_use: str = "development"
    independent_set_count: int = 2


@dataclass(frozen=True)
class LosoBeliefExport:
    model: LosoAxisModel
    beliefs: tuple[AxisBelief, ...]
    bundle_path: Path
    provenance_root: Path


def _family(span: Mapping[str, Any], axis: Axis) -> str:
    if axis is Axis.PLACEMENT:
        return str(span.get("start_source") or "unknown")
    if axis is Axis.STRUCTURE:
        return str(span.get("ref_decode_status") or "unknown")
    raise ValueError(f"unsupported calibration axis: {axis}")


def _axis_outcome(
    score: SpanScore,
    axis: Axis,
    *,
    placement_success_s: float,
    structure_success: float,
) -> bool | None:
    if axis is Axis.PLACEMENT:
        return (
            score.place_err_s < placement_success_s
            if score.place_err_s is not None
            else None
        )
    if axis is Axis.STRUCTURE:
        return (
            score.strict >= structure_success if score.strict is not None else None
        )
    raise ValueError(f"unsupported calibration axis: {axis}")


def fit_source_axis_calibration(
    *,
    train_set_id: str,
    timeline_spans: Sequence[Mapping[str, Any]],
    scores: Sequence[SpanScore],
    axis: Axis,
    placement_success_s: float = 15.0,
    structure_success: float = 0.8,
    minimum_family_n: int = 5,
) -> SourceAxisCalibration:
    """Fit the deterministic Beta source-family calibration for one axis."""
    if len(timeline_spans) != len(scores):
        raise ValueError(
            f"span/score length mismatch: {len(timeline_spans)} != {len(scores)}"
        )
    counts: dict[str, list[int]] = {}
    global_hits = global_misses = 0
    for span, score in zip(timeline_spans, scores):
        outcome = _axis_outcome(
            score,
            axis,
            placement_success_s=placement_success_s,
            structure_success=structure_success,
        )
        if outcome is None:
            continue
        family = _family(span, axis)
        pair = counts.setdefault(family, [0, 0])
        pair[0 if outcome else 1] += 1
        if outcome:
            global_hits += 1
        else:
            global_misses += 1
    return SourceAxisCalibration(
        axis=axis,
        train_set_id=train_set_id,
        criterion=(
            f"place_err_s<{placement_success_s:g}"
            if axis is Axis.PLACEMENT
            else f"strict>={structure_success:g}"
        ),
        by_family={
            family: BetaCount(hits=pair[0], misses=pair[1])
            for family, pair in sorted(counts.items())
        },
        global_count=BetaCount(global_hits, global_misses),
        minimum_family_n=minimum_family_n,
    )


def fit_loso_axis_model(
    *,
    train_set_id: str,
    held_set_id: str,
    timeline_spans: Sequence[Mapping[str, Any]],
    scores: Sequence[SpanScore],
    placement_success_s: float = 15.0,
    structure_success: float = 0.8,
    minimum_family_n: int = 5,
) -> LosoAxisModel:
    if train_set_id == held_set_id:
        raise ValueError(f"LOSO violation: train set == held set == {train_set_id}")
    return LosoAxisModel(
        train_set_id=train_set_id,
        held_set_id=held_set_id,
        placement=fit_source_axis_calibration(
            train_set_id=train_set_id,
            timeline_spans=timeline_spans,
            scores=scores,
            axis=Axis.PLACEMENT,
            placement_success_s=placement_success_s,
            structure_success=structure_success,
            minimum_family_n=minimum_family_n,
        ),
        structure=fit_source_axis_calibration(
            train_set_id=train_set_id,
            timeline_spans=timeline_spans,
            scores=scores,
            axis=Axis.STRUCTURE,
            placement_success_s=placement_success_s,
            structure_success=structure_success,
            minimum_family_n=minimum_family_n,
        ),
        placement_success_s=placement_success_s,
        structure_success=structure_success,
    )


def _count_dict(count: BetaCount) -> dict[str, Any]:
    return {
        "hits": count.hits,
        "misses": count.misses,
        "alpha_prior": count.alpha_prior,
        "beta_prior": count.beta_prior,
        "posterior_mean": count.mean,
    }


def model_document(model: LosoAxisModel) -> dict[str, Any]:
    def axis_doc(calibration: SourceAxisCalibration) -> dict[str, Any]:
        return {
            "axis": calibration.axis.value,
            "train_set_id": calibration.train_set_id,
            "criterion": calibration.criterion,
            "minimum_family_n": calibration.minimum_family_n,
            "calibration_status": calibration.calibration_status,
            "global": _count_dict(calibration.global_count),
            "families": {
                family: _count_dict(count)
                for family, count in sorted(calibration.by_family.items())
            },
        }

    return {
        "schema": model.schema,
        "train_set_id": model.train_set_id,
        "held_set_id": model.held_set_id,
        "intended_use": model.intended_use,
        "independent_set_count": model.independent_set_count,
        "placement_success_s": model.placement_success_s,
        "structure_success": model.structure_success,
        "placement": axis_doc(model.placement),
        "structure": axis_doc(model.structure),
    }


def _params_hash(model: LosoAxisModel) -> str:
    raw = json.dumps(model_document(model), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _segments(span: Mapping[str, Any], set_start_s: float) -> tuple[RefSegment, ...]:
    return tuple(
        RefSegment(
            mix_start_s=float(segment["mix_start_s"]) - set_start_s,
            ref_start_s=float(segment["ref_start_s"]),
            ref_end_s=float(segment["ref_end_s"]),
        )
        for segment in (span.get("ref_segments") or [])
    )


def infer_held_set_beliefs(
    *,
    model: LosoAxisModel,
    held_timeline: Mapping[str, Any],
    repo: ProvenanceRepository,
    run: Run,
    minimum_posterior: float = 0.6,
) -> list[AxisBelief]:
    """Apply a frozen cross-set model and persist held-set axis beliefs."""
    held_set_id = str(held_timeline.get("set_id") or "")
    if held_set_id != model.held_set_id:
        raise ValueError(
            f"held timeline set {held_set_id!r} != model held set "
            f"{model.held_set_id!r}"
        )
    if held_set_id == model.train_set_id:
        raise ValueError("LOSO violation: model applied to its training set")

    placement_rule = DecisionRule(
        decision_rule_id=f"placement-loso-{model.train_set_id}-v1",
        axis=Axis.PLACEMENT,
        version="1",
        minimum_posterior=minimum_posterior,
        maximum_entropy=1.0,
        calibration_status="development_only",
    )
    structure_rule = DecisionRule(
        decision_rule_id=f"structure-loso-{model.train_set_id}-v1",
        axis=Axis.STRUCTURE,
        version="1",
        minimum_posterior=minimum_posterior,
        maximum_entropy=1.0,
        calibration_status="development_only",
    )

    beliefs: list[AxisBelief] = []
    for span in held_timeline.get("spans", []):
        set_start_s = float(span["set_start_s"])
        placement_family = _family(span, Axis.PLACEMENT)
        structure_family = _family(span, Axis.STRUCTURE)
        placement_p, placement_bucket = model.placement.probability(
            placement_family
        )
        structure_p, structure_bucket = model.structure.probability(
            structure_family
        )
        segments = _segments(span, set_start_s)
        native_confidence = max(
            0.0, min(1.0, float(span.get("confidence") or 0.0))
        )
        result = AlignmentResult(
            recording_id=(
                str(span["recording_id"])
                if span.get("recording_id") is not None
                else None
            ),
            offset_s=float(span.get("ref_start_s") or 0.0),
            ref_end_s=(
                float(span["ref_end_s"])
                if span.get("ref_end_s") is not None
                else None
            ),
            segments=segments,
            confidence=native_confidence,
            source=f"timeline:{placement_family}|{structure_family}",
        )
        calibrated = record_placement_structure_beliefs(
            repo=repo,
            run=run,
            subject=SubjectRef(
                "set-slot", f"{held_set_id}::{span['slot_label']}"
            ),
            result=result,
            placement_rule=placement_rule,
            structure_rule=structure_rule,
            set_start_s=set_start_s,
            segment_mix_time_origin_s=set_start_s if segments else None,
            placement_probability=CalibratedAxisProbability(
                Axis.PLACEMENT,
                placement_p,
                f"{_MODEL_SCHEMA}:{model.train_set_id}:"
                f"placement:{placement_bucket}",
            ),
            structure_probability=(
                CalibratedAxisProbability(
                    Axis.STRUCTURE,
                    structure_p,
                    f"{_MODEL_SCHEMA}:{model.train_set_id}:"
                    f"structure:{structure_bucket}",
                )
                if segments
                else None
            ),
        )
        beliefs.extend((calibrated.placement, calibrated.structure))
    return beliefs


def run_loso_export(
    *,
    train_set_id: str,
    held_set_id: str,
    train_timeline_path: str | Path,
    held_timeline_path: str | Path,
    bundle_path: str | Path,
    provenance_root: str | Path,
    code_commit: str = "unknown",
    minimum_posterior: float = 0.6,
) -> LosoBeliefExport:
    """Fit on one set, apply to the other, and persist model + belief bundle."""
    if train_set_id == held_set_id:
        raise ValueError(f"LOSO violation: train set == held set == {train_set_id}")
    train_timeline_path = Path(train_timeline_path)
    held_timeline_path = Path(held_timeline_path)
    bundle_path = Path(bundle_path)
    provenance_root = Path(provenance_root)
    train_timeline = json.loads(train_timeline_path.read_text())
    held_timeline = json.loads(held_timeline_path.read_text())
    if str(train_timeline.get("set_id")) != train_set_id:
        raise ValueError("train timeline set_id mismatch")
    if str(held_timeline.get("set_id")) != held_set_id:
        raise ValueError("held timeline set_id mismatch")

    fixtures = sorted(
        (_repo_root() / "labeling" / "fixtures").glob("*_ground_truth.yaml")
    )
    gt_matches = [
        path
        for path in fixtures
        if str(yaml.safe_load(path.read_text()).get("set_id") or "")
        == train_set_id
    ]
    if len(gt_matches) != 1:
        raise ValueError(
            f"expected exactly one GT fixture for {train_set_id}, "
            f"found {len(gt_matches)}"
        )
    train_gt_path = gt_matches[0]
    train_scores = score_spans(
        train_set_id, train_timeline_path, gt_path=train_gt_path
    )
    model = fit_loso_axis_model(
        train_set_id=train_set_id,
        held_set_id=held_set_id,
        timeline_spans=train_timeline["spans"],
        scores=train_scores,
    )
    model_raw = (
        json.dumps(model_document(model), indent=2, sort_keys=True).encode() + b"\n"
    )

    conn = connect(provenance_root)
    store = ArtifactStore(provenance_root, conn)
    repo = ProvenanceRepository(conn)
    train_artifact = store.put_bytes(
        train_timeline_path.read_bytes(),
        kind=ArtifactKind.PREDICTED_TIMELINE,
        media_type="application/json",
        source_uri=str(train_timeline_path),
    )
    held_artifact = store.put_bytes(
        held_timeline_path.read_bytes(),
        kind=ArtifactKind.PREDICTED_TIMELINE,
        media_type="application/json",
        source_uri=str(held_timeline_path),
    )
    gt_artifact = store.put_bytes(
        train_gt_path.read_bytes(),
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="application/yaml",
        source_uri=str(train_gt_path),
    )

    fit_run = repo.begin_run(
        process=_PROCESS_FIT,
        process_version=_VERSION,
        code_commit=code_commit,
        params_hash=_params_hash(model),
    )
    try:
        repo.add_input(fit_run, train_artifact, role="development_timeline")
        repo.add_input(fit_run, gt_artifact, role="development_gt", ordinal=1)
        model_artifact = store.put_bytes(
            model_raw,
            kind=ArtifactKind.MODEL_CHECKPOINT,
            media_type="application/json",
            metadata={
                "calibration_status": "development_only",
                "train_set_id": train_set_id,
                "held_set_id": held_set_id,
            },
        )
        repo.add_output(fit_run, model_artifact, role="axis_calibration")
        repo.derive(
            model_artifact, train_artifact, fit_run, relation="calibrated_from"
        )
        repo.derive(model_artifact, gt_artifact, fit_run, relation="calibrated_from")
        repo.succeed(fit_run)
    except Exception as exc:
        repo.fail(fit_run, {"error": repr(exc)})
        raise

    infer_run = repo.begin_run(
        process=_PROCESS_INFER,
        process_version=_VERSION,
        code_commit=code_commit,
        params_hash=_params_hash(model),
    )
    try:
        repo.add_input(infer_run, model_artifact, role="axis_calibration", ordinal=0)
        repo.add_input(infer_run, held_artifact, role="held_timeline", ordinal=1)
        beliefs = infer_held_set_beliefs(
            model=model,
            held_timeline=held_timeline,
            repo=repo,
            run=infer_run,
            minimum_posterior=minimum_posterior,
        )
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        write_belief_bundle(bundle_path, held_set_id, beliefs)
        bundle_artifact = store.put_bytes(
            bundle_path.read_bytes(),
            kind=ArtifactKind.DIAGNOSTICS,
            media_type="application/json",
            source_uri=str(bundle_path),
        )
        repo.add_output(infer_run, bundle_artifact, role="axis_belief_bundle")
        repo.derive(
            bundle_artifact, model_artifact, infer_run, relation="inferred_from"
        )
        repo.derive(
            bundle_artifact, held_artifact, infer_run, relation="inferred_from"
        )
        repo.succeed(infer_run)
    except Exception as exc:
        repo.fail(infer_run, {"error": repr(exc)})
        raise

    return LosoBeliefExport(
        model=model,
        beliefs=tuple(beliefs),
        bundle_path=bundle_path,
        provenance_root=provenance_root,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-set-id", required=True)
    parser.add_argument("--held-set-id", required=True)
    parser.add_argument("--train-timeline", type=Path, required=True)
    parser.add_argument("--held-timeline", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, required=True)
    parser.add_argument("--minimum-posterior", type=float, default=0.6)
    args = parser.parse_args(argv)
    try:
        code_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        code_commit = "unknown"

    exported = run_loso_export(
        train_set_id=args.train_set_id,
        held_set_id=args.held_set_id,
        train_timeline_path=args.train_timeline,
        held_timeline_path=args.held_timeline,
        bundle_path=args.bundle_output,
        provenance_root=args.provenance_root,
        code_commit=code_commit,
        minimum_posterior=args.minimum_posterior,
    )
    accepted = sum(
        belief.decision.value == "accepted" for belief in exported.beliefs
    )
    print(
        f"LOSO axis beliefs: train={args.train_set_id} "
        f"held={args.held_set_id} (development_only)"
    )
    print(f"  beliefs: {len(exported.beliefs)} ({accepted} accepted)")
    print(f"  bundle: {exported.bundle_path}")
    print(f"  provenance: {exported.provenance_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
