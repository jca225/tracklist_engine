"""Deterministic builders for the §2/§8 versioning backbone (Brick 7).

Pure construction logic, no I/O beyond reading a requirements file / the
installed-distribution list. Every id here is DERIVED from content (sha256 over
the defining fields), so building the same parameter set / environment /
process twice yields the same id — which is what makes the repository writers
idempotent (INSERT OR IGNORE on a deterministic primary key).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

from .types import (
    Axis,
    EnvironmentSpec,
    FittedModel,
    Json,
    ParameterSet,
    ProcessSpec,
    TrainingSnapshot,
)


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")  # unambiguous field separator
    return h.hexdigest()


def canonical_values_hash(values: Json) -> str:
    """sha256 of the sorted-JSON serialization — key order never changes it."""
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def make_parameter_set(namespace: str, values: Json) -> ParameterSet:
    canonical = canonical_values_hash(values)
    return ParameterSet(
        parameter_set_id=_sha("parameter_set", namespace, canonical),
        namespace=namespace,
        values=dict(values),
        canonical_hash=canonical,
    )


def capture_environment(requirements_file: Optional[Path] = None) -> EnvironmentSpec:
    """Capture the live environment cheaply (§2 EnvironmentSpec).

    ``dependency_lock_hash`` = sha256 of the requirements file's bytes when one
    is given; otherwise sha256 over the sorted installed ``name==version``
    distribution list (stdlib ``importlib.metadata`` — a real lock of what is
    importable right now, no file needed).
    """
    if requirements_file is not None:
        lock_hash = hashlib.sha256(Path(requirements_file).read_bytes()).hexdigest()
    else:
        from importlib import metadata

        pins = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in metadata.distributions()
            if d.metadata["Name"] is not None
        )
        lock_hash = hashlib.sha256("\n".join(pins).encode("utf-8")).hexdigest()
    os_name = platform.system()
    architecture = platform.machine()
    runtime = f"python-{sys.version.split()[0]}"
    return EnvironmentSpec(
        environment_spec_id=_sha(
            "environment_spec", os_name, architecture, runtime, lock_hash
        ),
        os=os_name,
        architecture=architecture,
        runtime=runtime,
        dependency_lock_hash=lock_hash,
    )


def implementation_hash_of(fn: Callable[..., object]) -> str:
    """sha256 of the function's source (falls back to its qualified name when
    source is unavailable, e.g. a C extension or REPL-defined fn)."""
    try:
        payload = inspect.getsource(fn)
    except (OSError, TypeError):
        payload = f"{fn.__module__}.{fn.__qualname__}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_process_spec(
    *,
    name: str,
    version: str,
    stage: str,
    code_commit: str,
    parameter_set: ParameterSet,
    environment: EnvironmentSpec,
    model_refs: tuple[str, ...] = (),
    implementation_hash: str = "",
) -> ProcessSpec:
    """Deterministic ProcessSpec: identical (name, version, code, params, env,
    model_refs) ⇒ identical ``process_spec_id`` (idempotent recording)."""
    return ProcessSpec(
        process_spec_id=_sha(
            "process_spec",
            name,
            version,
            code_commit,
            parameter_set.canonical_hash,
            environment.dependency_lock_hash,
            *model_refs,
        ),
        name=name,
        version=version,
        stage=stage,
        code_commit=code_commit,
        parameter_set_id=parameter_set.parameter_set_id,
        environment_spec_id=environment.environment_spec_id,
        model_refs=tuple(model_refs),
        implementation_hash=implementation_hash,
    )


def make_training_snapshot(member_artifact_shas: Iterable[str]) -> TrainingSnapshot:
    """Frozen training-set snapshot; hash over the SORTED members, so the same
    set of artifacts always snapshots identically regardless of order."""
    members = tuple(sorted(set(member_artifact_shas)))
    snapshot_hash = _sha("training_snapshot", *members)
    return TrainingSnapshot(
        training_snapshot_id=_sha("training_snapshot_id", snapshot_hash),
        member_artifact_shas=members,
        snapshot_hash=snapshot_hash,
    )


def make_fitted_model(
    *,
    axis: Axis,
    process_spec: ProcessSpec,
    serialized_state_sha256: str,
    training_snapshot: TrainingSnapshot,
    status: str = "candidate",
) -> FittedModel:
    """Deterministic FittedModel id over (axis, process, state, snapshot) —
    re-registering the identical model is idempotent, never a duplicate row."""
    return FittedModel(
        fitted_model_id=_sha(
            "fitted_model",
            axis.value,
            process_spec.process_spec_id,
            serialized_state_sha256,
            training_snapshot.training_snapshot_id,
        ),
        axis=axis,
        process_spec_id=process_spec.process_spec_id,
        serialized_state_sha256=serialized_state_sha256,
        training_snapshot_id=training_snapshot.training_snapshot_id,
        model_state_hash=serialized_state_sha256,
        status=status,
    )
