"""Metaprogramming registry for artifact types + transformations (Brick 7).

The declarative half of the §6 Analyzer contract: a feature producer is a PURE
function ``fn(inputs: Mapping[role, bytes]) -> Mapping[role, bytes]`` plus two
declarations — ``@artifact_type`` names a content-addressed kind, and
``@transformation`` names the producer, its declared input/output kinds, its
parameters, and the models it stands on. ``Registry.run`` then does ALL the
provenance plumbing: it persists a deterministic ProcessSpec (ParameterSet +
live-captured EnvironmentSpec), begins a versioned run, stores each output
content-addressed under its declared kind, and links input/output/derivation
edges. Adding a NEW data type to the engine is exactly two decorators — zero
extra plumbing (the proof test in tests/provenance/test_registry.py).

Substrate rules hold: stdlib only, mypy --strict, and the registry never
executes anything by itself — declarations are inert until ``run`` is called
with an explicit store/repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, TypeVar

from .repository import ProvenanceRepository
from .store import ArtifactStore, kind_str
from .types import Artifact, Json
from .versioning import (
    capture_environment,
    implementation_hash_of,
    make_parameter_set,
    make_process_spec,
)

_T = TypeVar("_T")

TransformFn = Callable[[Mapping[str, bytes]], Mapping[str, bytes]]


@dataclass(frozen=True)
class ArtifactTypeSpec:
    """One registered artifact kind: the ``kind`` string artifacts are stored
    under, its media type, and a schema version for the serialized payload."""

    kind: str
    media_type: str
    version: int = 1


@dataclass(frozen=True)
class TransformationSpec:
    """One declared producer: role→kind signatures + the pure compute fn."""

    name: str
    version: str
    inputs: Mapping[str, str]  # role -> artifact kind
    outputs: Mapping[str, str]  # role -> artifact kind
    stage: str
    params: Json = field(default_factory=dict)
    reproducibility: str = "exact"
    model_refs: tuple[str, ...] = ()
    fn: Optional[TransformFn] = None


class Registry:
    """Declared artifact types + transformations, and the one place that turns
    a declaration into a fully-provenanced execution."""

    def __init__(self) -> None:
        self._artifact_types: dict[str, ArtifactTypeSpec] = {}
        self._transformations: dict[str, TransformationSpec] = {}

    # --- declaration --------------------------------------------------------
    def register_artifact_type(self, spec: ArtifactTypeSpec) -> None:
        existing = self._artifact_types.get(spec.kind)
        if existing is not None and existing != spec:
            raise ValueError(
                f"artifact type {spec.kind!r} already registered with a "
                "different declaration"
            )
        self._artifact_types[spec.kind] = spec

    def register_transformation(self, spec: TransformationSpec) -> None:
        if spec.fn is None:
            raise ValueError(f"transformation {spec.name!r} declared without a fn")
        existing = self._transformations.get(spec.name)
        if existing is not None and existing.fn is not spec.fn:
            raise ValueError(f"transformation {spec.name!r} already registered")
        self._transformations[spec.name] = spec

    def artifact_type(self, kind: str) -> ArtifactTypeSpec:
        try:
            return self._artifact_types[kind]
        except KeyError:
            raise ValueError(f"artifact type {kind!r} is not registered") from None

    def transformation(self, name: str) -> TransformationSpec:
        try:
            return self._transformations[name]
        except KeyError:
            raise ValueError(f"transformation {name!r} is not registered") from None

    # --- dependency graph (from declared kinds) -----------------------------
    def producers_of(self, kind: str) -> list[str]:
        """Names of transformations declaring an output of this kind."""
        return sorted(
            name
            for name, spec in self._transformations.items()
            if kind in spec.outputs.values()
        )

    def graph(self) -> Mapping[str, Mapping[str, Mapping[str, str]]]:
        """The declared dependency graph: name -> {inputs, outputs} role→kind."""
        return {
            name: {"inputs": dict(spec.inputs), "outputs": dict(spec.outputs)}
            for name, spec in sorted(self._transformations.items())
        }

    # --- execution ----------------------------------------------------------
    def run(
        self,
        name: str,
        inputs: Mapping[str, Artifact],
        *,
        store: ArtifactStore,
        repo: ProvenanceRepository,
        code_commit: str,
        seed: int | None = None,
    ) -> dict[str, Artifact]:
        """Execute a declared transformation with full provenance.

        Persists the transformation's ProcessSpec (ParameterSet from the
        declared params + a live-captured EnvironmentSpec), begins a versioned
        run from it, records each input, calls the pure fn on the inputs'
        bytes, stores each output content-addressed under its declared kind,
        and records output + derivation edges (relation = transformation
        name). On failure the run is FAILED with the error, then re-raised.
        """
        spec = self.transformation(name)
        assert spec.fn is not None  # register_transformation guarantees this

        # Declared-signature checks fail BEFORE any provenance is written.
        if set(inputs) != set(spec.inputs):
            raise ValueError(
                f"{name}: input roles {sorted(inputs)} != declared "
                f"{sorted(spec.inputs)}"
            )
        for role, artifact in inputs.items():
            declared = spec.inputs[role]
            if kind_str(artifact.kind) != declared:
                raise ValueError(
                    f"{name}: input {role!r} is kind "
                    f"{kind_str(artifact.kind)!r}, declared {declared!r}"
                )
        out_types = {
            role: self.artifact_type(kind) for role, kind in spec.outputs.items()
        }

        parameter_set = repo.record_parameter_set(
            make_parameter_set(namespace=name, values=spec.params)
        )
        environment = repo.record_environment_spec(capture_environment())
        process_spec = repo.record_process_spec(
            make_process_spec(
                name=name,
                version=spec.version,
                stage=spec.stage,
                code_commit=code_commit,
                parameter_set=parameter_set,
                environment=environment,
                model_refs=spec.model_refs,
                implementation_hash=implementation_hash_of(spec.fn),
            )
        )
        run = repo.begin_run_from_spec(process_spec, seed=seed)
        try:
            input_bytes: dict[str, bytes] = {}
            for ordinal, (role, artifact) in enumerate(sorted(inputs.items())):
                repo.add_input(run, artifact, role=role, ordinal=ordinal)
                with store.open(artifact.content_sha256) as f:
                    input_bytes[role] = f.read()

            produced = spec.fn(input_bytes)
            if set(produced) != set(spec.outputs):
                raise ValueError(
                    f"{name}: fn returned roles {sorted(produced)} != declared "
                    f"{sorted(spec.outputs)}"
                )

            outputs: dict[str, Artifact] = {}
            for ordinal, (role, content) in enumerate(sorted(produced.items())):
                type_spec = out_types[role]
                out_art = store.put_bytes(
                    content,
                    kind=type_spec.kind,
                    media_type=type_spec.media_type,
                    metadata={
                        "artifact_type_version": type_spec.version,
                        "transformation": name,
                        "role": role,
                    },
                )
                repo.add_output(run, out_art, role=role, ordinal=ordinal)
                for in_art in inputs.values():
                    repo.derive(out_art, in_art, run, relation=name)
                outputs[role] = out_art
            repo.succeed(run)
            return outputs
        except Exception as exc:  # record the failure, never swallow it
            repo.fail(run, {"error": repr(exc)})
            raise

    # --- tests --------------------------------------------------------------
    def reset(self) -> None:
        """Drop all declarations (tests only)."""
        self._artifact_types.clear()
        self._transformations.clear()


#: Module-level singleton the decorators register into.
REGISTRY = Registry()


def artifact_type(kind: str, media_type: str, version: int = 1) -> Callable[[_T], _T]:
    """Register a content-addressed artifact kind; returns the target unchanged."""

    def deco(target: _T) -> _T:
        REGISTRY.register_artifact_type(ArtifactTypeSpec(kind, media_type, version))
        return target

    return deco


def transformation(
    name: str,
    version: str,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
    *,
    stage: str,
    params: Json | None = None,
    reproducibility: str = "exact",
    model_refs: tuple[str, ...] = (),
) -> Callable[[TransformFn], TransformFn]:
    """Register a pure compute fn as a declared transformation; the fn is
    returned unchanged (call ``REGISTRY.run(name, ...)`` to execute it with
    provenance)."""

    def deco(fn: TransformFn) -> TransformFn:
        REGISTRY.register_transformation(
            TransformationSpec(
                name=name,
                version=version,
                inputs=dict(inputs),
                outputs=dict(outputs),
                stage=stage,
                params=dict(params or {}),
                reproducibility=reproducibility,
                model_refs=tuple(model_refs),
                fn=fn,
            )
        )
        return fn

    return deco
