"""Lab / sandbox evaluation for the learned identity verifier.

Trains the learned verifier on the canonical DB minus a held-out test set of
DJ sets (default: BB11 and BB12), then evaluates same-song and same-version
prediction on the held-out recordings. Outputs a JSON report with
probabilistic metrics, ternary-verdict metrics, confusion matrices, and top
failure cases.

This is intentionally a sandbox script: it does not write acquisition cases,
mutate the DB, or promote any audio to production.

Example:
    venvs/audio/bin/python scripts/eval_identity_verifier.py \
        --db /mnt/storage/data/db/music_database.db \
        --test-set-ids 2nvzlh2k 1fsnxchk \
        --output reports/identity_verifier_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.identity_data import _load_recordings
from analysis.identity_learned import (
    PAIR_FEATURE_NAMES,
    RecordingFeatures,
    build_song_pairs,
    build_version_pairs,
    extract_pair_features,
    train_verifiers,
)
from analysis.identity_verify import AxisVerdict, VersionVerdict

_LOG = logging.getLogger("eval_identity_verifier")

DEFAULT_TEST_SETS = ["2nvzlh2k", "1fsnxchk"]


@dataclass(frozen=True)
class PairResult:
    """One test pair with its labels and model predictions."""

    ref_id: str
    ref_name: str
    cand_id: str
    cand_name: str
    task: str  # "song" or "version"
    label: int
    song_prob: float | None
    version_prob: float | None
    song_verdict: str
    version_verdict: str


@dataclass(frozen=True)
class TaskMetrics:
    """Aggregated metrics for one task."""

    n_pairs: int
    n_ambiguous: int
    n_usable: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    confusion_matrix: list[list[int]] | None
    feature_importance: dict[str, float] | None


def _recording_ids_for_set(db_path: Path, set_id: str) -> set[str]:
    """Return the distinct recording_ids used in a set's ground truth."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT recording_id FROM set_ground_truth WHERE set_id = ?",
        (set_id,),
    )
    ids = {row["recording_id"] for row in cur.fetchall()}
    conn.close()
    return ids


def _split_recordings(
    recordings: dict[str, RecordingFeatures],
    test_ids: set[str],
    max_train: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, RecordingFeatures], dict[str, RecordingFeatures]]:
    """Partition recordings into train/test by recording_id.

    Optionally down-sample the training set to ``max_train`` recordings for lab
    runs on memory-constrained machines. Sampling is deterministic given the
    seed.

    Note: when this function is called from ``_run`` the sampling has already
    been performed at the DB level by ``_sample_train_ids`` to keep memory
    bounded. ``max_train`` is kept here for standalone test callers.
    """
    train: dict[str, RecordingFeatures] = {}
    test: dict[str, RecordingFeatures] = {}
    for rid, rec in recordings.items():
        if rid in test_ids:
            test[rid] = rec
        else:
            train[rid] = rec

    if max_train is not None and len(train) > max_train:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(list(train.keys()), size=max_train, replace=False))
        train = {rid: rec for rid, rec in train.items() if rid in keep}
        _LOG.info("Down-sampled train set to %d recordings", len(train))

    return train, test


def _sample_train_ids(
    db_path: Path,
    test_ids: set[str],
    max_train: int | None = None,
    seed: int = 42,
) -> set[str]:
    """Return a deterministic sample of training recording IDs from the DB.

    Excludes ``test_ids`` from the candidate pool. If ``max_train`` is set and
    the candidate pool is larger, this function prefers works that have
    multiple recordings (which are the only source of same-version training
    pairs) before filling the rest with single-recording works. This keeps the
    MERT/feature loading memory-bounded while still producing usable training
    pairs.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT recording_id, work_id FROM recording")
    rows = [row for row in cur.fetchall() if row["recording_id"] not in test_ids]
    conn.close()

    by_work: dict[str, list[str]] = {}
    for row in rows:
        wid = row["work_id"] or ""
        by_work.setdefault(wid, []).append(row["recording_id"])

    multi_works = [wid for wid, rids in by_work.items() if len(rids) >= 2 and wid]
    single_works = [wid for wid, rids in by_work.items() if len(rids) == 1 and wid]
    orphan_works = [wid for wid, rids in by_work.items() if not wid]

    rng = np.random.default_rng(seed)
    rng.shuffle(multi_works)
    rng.shuffle(single_works)
    rng.shuffle(orphan_works)

    chosen: list[str] = []
    work_order = multi_works + single_works + orphan_works
    for wid in work_order:
        chosen.extend(by_work[wid])
        if max_train is not None and len(chosen) >= max_train:
            break

    if max_train is not None:
        chosen = chosen[:max_train]

    covered_multi = {
        wid for wid in multi_works if any(rid in chosen for rid in by_work[wid])
    }
    _LOG.info(
        "Selected %d train recordings covering %d/%d multi-recording works",
        len(chosen),
        len(covered_multi),
        len(multi_works),
    )
    if max_train is not None and len(chosen) < max_train:
        _LOG.warning(
            "Could not reach max_train=%d; only %d non-test recordings available",
            max_train,
            len(chosen),
        )

    return set(chosen)


def _build_test_pairs(
    recordings: Sequence[RecordingFeatures],
    task: str,
) -> tuple[list[RecordingFeatures], list[RecordingFeatures], list[int]]:
    """Build test pairs for a task.

    ``task`` is ``"song"`` (positive = same work_id) or ``"version"`` (positive
    = same work_id and same version, negatives = same work_id different version).
    """
    refs: list[RecordingFeatures] = []
    cands: list[RecordingFeatures] = []
    labels: list[int] = []
    n = len(recordings)
    for i in range(n):
        ref = recordings[i]
        for j in range(i + 1, n):
            cand = recordings[j]
            if not ref.work_id or not cand.work_id:
                continue
            features = extract_pair_features(ref, cand)
            if not features.is_usable():
                continue
            if task == "song":
                label = 1 if ref.work_id == cand.work_id else 0
            else:  # version
                if ref.work_id != cand.work_id:
                    continue
                label = 1 if ref.version == cand.version else 0
            refs.append(ref)
            cands.append(cand)
            labels.append(label)
    return refs, cands, labels


def _ternary_metrics(
    y_true: list[int],
    y_prob: list[float | None],
    threshold: float = 0.5,
    margin: float = 0.2,
) -> tuple[list[int | None], TaskMetrics]:
    """Return (ternary_predictions, metrics) where ambiguous pairs are None."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred: list[int | None] = []
    y_filt: list[int] = []
    p_filt: list[int] = []
    for t, p in zip(y_true, y_prob):
        if p is None:
            y_pred.append(None)
            continue
        if p >= threshold + margin:
            pred = 1
        elif p <= threshold - margin:
            pred = 0
        else:
            y_pred.append(None)
            continue
        y_pred.append(pred)
        y_filt.append(t)
        p_filt.append(pred)

    n_ambiguous = sum(1 for p in y_pred if p is None)
    if not y_filt:
        return y_pred, TaskMetrics(
            n_pairs=len(y_true),
            n_ambiguous=n_ambiguous,
            n_usable=0,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            roc_auc=None,
            confusion_matrix=None,
            feature_importance=None,
        )

    cm = confusion_matrix(y_filt, p_filt, labels=[0, 1]).tolist()
    return y_pred, TaskMetrics(
        n_pairs=len(y_true),
        n_ambiguous=n_ambiguous,
        n_usable=len(y_filt),
        accuracy=float(accuracy_score(y_filt, p_filt)),
        precision=float(precision_score(y_filt, p_filt, zero_division=0)),
        recall=float(recall_score(y_filt, p_filt, zero_division=0)),
        f1=float(f1_score(y_filt, p_filt, zero_division=0)),
        roc_auc=None,
        confusion_matrix=cm,
        feature_importance=None,
    )


def _probabilistic_roc_auc(
    y_true: list[int], y_prob: list[float | None]
) -> float | None:
    """ROC-AUC on pairs with usable probabilities."""
    from sklearn.metrics import roc_auc_score

    pairs = [(t, p) for t, p in zip(y_true, y_prob) if p is not None]
    if len(pairs) < 2 or len({t for t, _ in pairs}) < 2:
        return None
    y, p = zip(*pairs)
    return float(roc_auc_score(y, p))


def _verdict_from_prob(
    prob: float | None,
    threshold: float,
    negative: str,
    positive: str,
    ambiguous: str,
    margin: float = 0.2,
) -> str:
    """Return a ternary verdict string from a probability without re-extracting pair features."""
    if prob is None:
        return ambiguous
    if prob >= threshold + margin:
        return positive
    if prob <= threshold - margin:
        return negative
    return ambiguous


def _evaluate_task(
    song_verifier: Any,
    version_verifier: Any,
    refs: list[RecordingFeatures],
    cands: list[RecordingFeatures],
    labels: list[int],
    task: str,
) -> tuple[TaskMetrics, list[PairResult]]:
    """Evaluate one task and return metrics + per-pair results.

    Feature extraction is still per-pair (the features are heterogeneous), but
    the Random Forest ``predict_proba`` calls are batched to avoid the heavy
    per-sample sklearn overhead on 34k+ test pairs.
    """
    features_list = [extract_pair_features(ref, cand) for ref, cand in zip(refs, cands)]
    usable_indices = [i for i, f in enumerate(features_list) if f.is_usable()]

    song_probs: list[float | None] = [None] * len(refs)
    version_probs: list[float | None] = [None] * len(refs)

    if usable_indices:
        X = np.stack([features_list[i].to_vector() for i in usable_indices])
        song_batch = song_verifier.model.predict_proba(X)[:, 1]
        version_batch = version_verifier.model.predict_proba(X)[:, 1]
        for idx, sp, vp in zip(usable_indices, song_batch, version_batch):
            song_probs[idx] = float(sp)
            version_probs[idx] = float(vp)

    song_verdicts = [
        _verdict_from_prob(
            sp,
            song_verifier.threshold,
            AxisVerdict.DIFFERENT.value,
            AxisVerdict.SAME.value,
            AxisVerdict.AMBIGUOUS.value,
        )
        for sp in song_probs
    ]
    version_verdicts = [
        _verdict_from_prob(
            vp,
            version_verifier.threshold,
            VersionVerdict.DIFFERENT_VERSION.value,
            VersionVerdict.SAME_VERSION.value,
            VersionVerdict.AMBIGUOUS.value,
        )
        for vp in version_probs
    ]

    y_pred, metrics = _ternary_metrics(
        labels, version_probs if task == "version" else song_probs
    )
    metrics = replace(
        metrics,
        roc_auc=_probabilistic_roc_auc(
            labels, version_probs if task == "version" else song_probs
        ),
    )

    if task == "version":
        metrics = replace(
            metrics,
            feature_importance=dict(
                zip(
                    PAIR_FEATURE_NAMES,
                    version_verifier.model.feature_importances_.tolist(),
                )
            ),
        )

    results = [
        PairResult(
            ref_id=ref.recording_id,
            ref_name=f"{ref.artist} - {ref.title}",
            cand_id=cand.recording_id,
            cand_name=f"{cand.artist} - {cand.title}",
            task=task,
            label=label,
            song_prob=sp,
            version_prob=vp,
            song_verdict=sv,
            version_verdict=vv,
        )
        for ref, cand, label, sp, vp, sv, vv in zip(
            refs,
            cands,
            labels,
            song_probs,
            version_probs,
            song_verdicts,
            version_verdicts,
        )
    ]
    return metrics, results


def _worst_cases(
    results: list[PairResult], task: str, k: int = 20
) -> list[dict[str, Any]]:
    """Return top false positives and false negatives for inspection."""
    usable = [
        r
        for r in results
        if r.task == task
        and (r.version_prob if task == "version" else r.song_prob) is not None
    ]
    out: list[dict[str, Any]] = []
    for r in usable:
        prob = r.version_prob if task == "version" else r.song_prob
        assert prob is not None
        pred = 1 if prob >= 0.7 else 0 if prob <= 0.3 else None
        if pred is None:
            continue
        if pred != r.label:
            out.append(
                {
                    "ref": f"{r.ref_id} {r.ref_name}",
                    "cand": f"{r.cand_id} {r.cand_name}",
                    "label": r.label,
                    "predicted": pred,
                    "prob": round(prob, 3),
                }
            )
    # Sort by confidence of wrong prediction descending.
    out.sort(key=lambda x: abs(x["prob"] - 0.5), reverse=True)
    return out[:k]


def _run(
    db_path: Path,
    test_set_ids: Sequence[str],
    model_path: Path | None,
    output_path: Path | None,
    min_train_recordings: int,
    max_train_recordings: int | None,
    mert_stride: int = 1,
) -> int:
    test_ids: set[str] = set()
    for set_id in test_set_ids:
        ids = _recording_ids_for_set(db_path, set_id)
        _LOG.info("Set %s contributes %d recording ids to test set", set_id, len(ids))
        test_ids.update(ids)

    train_ids = _sample_train_ids(db_path, test_ids, max_train=max_train_recordings)
    load_ids = train_ids | test_ids
    _LOG.info(
        "Loading %d recordings from DB (train=%d, test=%d, mert_stride=%d)",
        len(load_ids),
        len(train_ids),
        len(test_ids),
        mert_stride,
    )
    recordings = _load_recordings(
        db_path, recording_ids=load_ids, mert_stride=mert_stride
    )
    if not recordings:
        _LOG.error("No recordings loaded from %s", db_path)
        return 1

    train, test = _split_recordings(recordings, test_ids)
    _LOG.info(
        "Train recordings: %d, test recordings: %d (from %d in DB)",
        len(train),
        len(test),
        len(recordings),
    )

    if len(train) < min_train_recordings:
        _LOG.error(
            "Too few train recordings (%d < %d)", len(train), min_train_recordings
        )
        return 1

    if len(test) < 2:
        _LOG.error("Need at least 2 test recordings to form pairs, got %d", len(test))
        return 1

    if model_path and model_path.exists():
        _LOG.info("Loading pre-trained model from %s", model_path)
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        song_verifier = bundle["song"]
        version_verifier = bundle["version"]
    else:
        _LOG.info("Training model on non-test recordings")
        song_verifier, version_verifier, _ = train_verifiers(list(train.values()))

    song_refs, song_cands, song_labels = _build_test_pairs(list(test.values()), "song")
    version_refs, version_cands, version_labels = _build_test_pairs(
        list(test.values()), "version"
    )

    _LOG.info("Same-song test pairs: %d", len(song_refs))
    _LOG.info("Same-version test pairs: %d", len(version_refs))

    song_metrics, song_results = _evaluate_task(
        song_verifier, version_verifier, song_refs, song_cands, song_labels, "song"
    )
    version_metrics, version_results = _evaluate_task(
        song_verifier,
        version_verifier,
        version_refs,
        version_cands,
        version_labels,
        "version",
    )

    report = {
        "db": str(db_path),
        "test_set_ids": list(test_set_ids),
        "model_path": str(model_path) if model_path else None,
        "train_recordings": len(train),
        "test_recordings": len(test),
        "same_song": {
            "metrics": {
                "n_pairs": song_metrics.n_pairs,
                "n_ambiguous": song_metrics.n_ambiguous,
                "n_usable": song_metrics.n_usable,
                "accuracy": song_metrics.accuracy,
                "precision": song_metrics.precision,
                "recall": song_metrics.recall,
                "f1": song_metrics.f1,
                "roc_auc": song_metrics.roc_auc,
                "confusion_matrix": song_metrics.confusion_matrix,
            },
            "worst_cases": _worst_cases(song_results, "song"),
        },
        "same_version": {
            "metrics": {
                "n_pairs": version_metrics.n_pairs,
                "n_ambiguous": version_metrics.n_ambiguous,
                "n_usable": version_metrics.n_usable,
                "accuracy": version_metrics.accuracy,
                "precision": version_metrics.precision,
                "recall": version_metrics.recall,
                "f1": version_metrics.f1,
                "roc_auc": version_metrics.roc_auc,
                "confusion_matrix": version_metrics.confusion_matrix,
                "feature_importance": version_metrics.feature_importance,
            },
            "worst_cases": _worst_cases(version_results, "version"),
        },
    }

    _LOG.info("Same-song  ROC-AUC: %s", song_metrics.roc_auc)
    _LOG.info("Same-song  F1: %s", song_metrics.f1)
    _LOG.info("Same-version ROC-AUC: %s", version_metrics.roc_auc)
    _LOG.info("Same-version F1: %s", version_metrics.f1)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        _LOG.info("Report written to %s", output_path)
    else:
        print(json.dumps(report, indent=2))

    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lab evaluation of learned identity verifier"
    )
    p.add_argument("--db", type=Path, required=True, help="Path to music_database.db")
    p.add_argument(
        "--test-set-ids",
        nargs="+",
        default=DEFAULT_TEST_SETS,
        help="Held-out set IDs for evaluation (default: BB11 + BB12)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional pre-trained model pickle to evaluate (otherwise train on the fly)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (if omitted, prints to stdout)",
    )
    p.add_argument(
        "--min-train-recordings",
        type=int,
        default=10,
        help="Minimum training recordings required",
    )
    p.add_argument(
        "--max-train-recordings",
        type=int,
        default=None,
        help="Down-sample training recordings to this many (lab / memory-limited runs)",
    )
    p.add_argument(
        "--mert-stride",
        type=int,
        default=1,
        help="Load every Nth MERT measure per track (default: 1 = all). Values > 1 reduce slow-storage I/O.",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _run(
        db_path=args.db,
        test_set_ids=args.test_set_ids,
        model_path=args.model,
        output_path=args.output,
        min_train_recordings=args.min_train_recordings,
        max_train_recordings=args.max_train_recordings,
        mert_stride=args.mert_stride,
    )


if __name__ == "__main__":
    raise SystemExit(main())
