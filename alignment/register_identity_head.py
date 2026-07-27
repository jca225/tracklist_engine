"""Brick 9 — register a saved MERT identity head as FittedModel(axis=IDENTITY).

Closes the gap Brick 7 surfaced and feature_store_cutover.md Decision 3
ratified: ``train.py --train-mert`` used to fit the identity head in memory
and never persist it, so no IDENTITY FittedModel could exist (Brick 7
registered the trajectory decoder, honestly labeled ``axis=STRUCTURE``).
``train.py --save-head-checkpoint`` (additive, opt-in) now persists the fitted
head; this module walks that real checkpoint onto the provenance substrate,
mirroring ``producers.register_trained_decoder``:

- checkpoint bytes -> content-addressed ``MODEL_CHECKPOINT`` artifact
- the REAL training inputs -> artifacts + ``TrainingSnapshot``: the GT fixture
  YAML the spans came from (``LABEL_MANIFEST``) and the cached MERT measure
  stack the features came from (``FEATURE_BLOB`` — the ``mert_store``
  ``.cache/mert/<set>_mert.npz`` bytes as-trained-on)
- a training ``ProcessSpec`` (code commit + params from the checkpoint's own
  saved cfg/meta + environment + MERT model_refs)
- ``FittedModel(axis=IDENTITY, ...)`` — grounding laws 13/14 with an identity
  model on the substrate store.

Honesty: registration is post-hoc — ``code_commit`` is the registering commit
(flagged ``registered_at_commit`` in the checkpoint metadata, like Brick 7),
and the train/eval slot split is recorded as parameters (``eval_fraction`` +
the deterministic ``split_targets`` protocol), not as a separate artifact.
Default store: the Brick-7 producers store
(``alignment/out/provenance/producers``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    FittedModel,
    ProvenanceRepository,
    capture_environment,
    make_fitted_model,
    make_parameter_set,
    make_process_spec,
    make_training_snapshot,
)
from alignment.mert_store import _PROBE_LAYER, _cache_path
from alignment.producers import (
    _GT_FIXTURES,
    _MERT_MODEL_REF,
    _OUT_ROOT,
    _REPO,
)


def _checkpoint_params(checkpoint: Path) -> dict[str, Any]:
    """Read the head checkpoint's own saved cfg/meta (the REAL training args
    ``external.checkpoint.save_head`` embedded). Falls back to the filename
    when torch cannot read the payload."""
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return {
            "cfg": dict(payload.get("cfg", {})),
            "meta": dict(payload.get("meta", {})),
            "n_heads": payload.get("n_heads"),
        }
    except Exception as exc:
        print(f"  head: could not read checkpoint payload ({exc!r}) — file only")
        return {"source_checkpoint": checkpoint.name}


def register_mert_identity_head(
    checkpoint: Path,
    *,
    set_id: str,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str,
    gt_path: Optional[Path] = None,
    mert_cache: Optional[Path] = None,
    eval_fraction: float = 0.2,
    requirements: Optional[Path] = None,
) -> Optional[FittedModel]:
    """Saved MERT identity head -> MODEL_CHECKPOINT + TrainingSnapshot +
    training ProcessSpec + FittedModel(axis=IDENTITY).

    The snapshot members are the REAL training inputs: the GT fixture the
    spans were built from and the MERT feature cache the head trained on.
    Returns None (with a message) when any input file is absent — never
    registers a model whose training set cannot be snapshotted.
    """
    if not checkpoint.is_file():
        print(f"  head: checkpoint {checkpoint} not found — skipped")
        return None
    gt_path = gt_path if gt_path is not None else _GT_FIXTURES.get(set_id)
    if gt_path is None or not gt_path.is_file():
        print(f"  head: GT fixture missing for {set_id} ({gt_path}) — skipped")
        return None
    mert_cache = mert_cache if mert_cache is not None else _cache_path(set_id)
    if not mert_cache.is_file():
        print(f"  head: MERT cache missing for {set_id} ({mert_cache}) — skipped")
        return None

    params = _checkpoint_params(checkpoint)
    params["set_id"] = set_id
    params["mert_probe_layer"] = _PROBE_LAYER
    params["split"] = {"protocol": "split_targets", "eval_fraction": eval_fraction}
    params["train_cli"] = "train.py --eval --train-mert --save-head-checkpoint"

    gt = store.put_file(
        gt_path,
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="application/yaml",
        metadata={"set_id": set_id, "role": "training_ground_truth"},
        source_uri=str(
            gt_path.relative_to(_REPO) if gt_path.is_relative_to(_REPO) else gt_path
        ),
    )
    features = store.put_file(
        mert_cache,
        kind=ArtifactKind.FEATURE_BLOB,
        media_type="application/x-npz",
        metadata={
            "set_id": set_id,
            "role": "training_features",
            "model": _MERT_MODEL_REF,
            "layer": _PROBE_LAYER,
            "note": "mert_store cache (mix + ref measure stacks) as trained on",
        },
        source_uri=str(mert_cache),
    )
    state = store.put_file(
        checkpoint,
        kind=ArtifactKind.MODEL_CHECKPOINT,
        media_type="application/octet-stream",
        metadata={
            "registered_at_commit": code_commit,
            "note": (
                "post-hoc registration: code_commit is the registering commit; "
                "the head was trained by train.py --save-head-checkpoint on the "
                "same working tree"
            ),
            "payload": "external/checkpoint.py save_head({'meta','cfg','n_heads','state_dicts'})",
        },
        source_uri=str(checkpoint),
    )

    parameter_set = repo.record_parameter_set(
        make_parameter_set("train.mert_identity_head", params)
    )
    environment = repo.record_environment_spec(
        capture_environment(requirements or _REPO / "requirements-audio.txt")
    )
    spec = repo.record_process_spec(
        make_process_spec(
            name="train.mert_identity_head",
            version="1",
            stage="training",
            code_commit=code_commit,
            parameter_set=parameter_set,
            environment=environment,
            model_refs=(_MERT_MODEL_REF, f"layer={_PROBE_LAYER}"),
            implementation_hash="",
        )
    )
    run = repo.begin_run_from_spec(spec)
    try:
        repo.add_input(run, gt, role="training_ground_truth", ordinal=0)
        repo.add_input(run, features, role="training_features", ordinal=1)
        repo.add_output(run, state, role="model_state")
        repo.derive(state, gt, run, relation="trained_on")
        repo.derive(state, features, run, relation="trained_on")
        repo.succeed(run)
    except Exception as exc:
        repo.fail(run, {"error": repr(exc)})
        raise

    snapshot = repo.record_training_snapshot(
        make_training_snapshot([gt.content_sha256, features.content_sha256])
    )
    return repo.record_fitted_model(
        make_fitted_model(
            axis=Axis.IDENTITY,  # the MERT head scores span↔recording identity
            process_spec=spec,
            serialized_state_sha256=state.content_sha256,
            training_snapshot=snapshot,
        )
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="head checkpoint written by train.py --save-head-checkpoint",
    )
    ap.add_argument(
        "--set-id",
        required=True,
        help="training set id (e.g. BB12 = 1fsnxchk, train.py's DEFAULT_YAML)",
    )
    ap.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="the split_targets eval_fraction the training run used",
    )
    ap.add_argument(
        "--db-root", default=None, help=f"provenance root (default {_OUT_ROOT})"
    )
    args = ap.parse_args()

    from core.provenance import connect

    db_root = Path(args.db_root or _OUT_ROOT)
    db_root.mkdir(parents=True, exist_ok=True)
    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    repo = ProvenanceRepository(conn)
    try:
        code_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO)
            .decode()
            .strip()
        )
    except Exception:
        code_commit = "unknown"

    model = register_mert_identity_head(
        args.checkpoint,
        set_id=args.set_id,
        store=store,
        repo=repo,
        code_commit=code_commit,
        eval_fraction=args.eval_fraction,
    )
    if model is None:
        return 2
    print(
        f"  fitted_model         : {model.fitted_model_id[:16]} "
        f"(axis={model.axis.value}, state={model.serialized_state_sha256[:12]}, "
        f"snapshot={model.training_snapshot_id[:12]})"
    )
    print(f"  provenance DB        : {db_root}")
    print(f"  check laws           : python -m core.provenance.laws --db {db_root}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
