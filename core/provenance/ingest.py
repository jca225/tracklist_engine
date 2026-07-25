"""Register an already-computed feature as a canonical provenanced artifact.

The substrate primitive behind the forward-routing dual-write (cutover Decision 2
step 2): a stage that has *just computed* a feature (MERT, fingerprint, …) records
it here — content-addressed, versioned (a ProcessSpec naming the analyzer + its
model), with a producing run and, when the source audio is known, a derivation
edge from it. Unlike ``Registry.run`` this does NOT recompute from input bytes —
the feature already exists — so it mirrors ``producers.migrate_cached_mert``'s
"register existing bytes" shape, but lives in ``core`` so any stage can call it
without importing the aligner producers (analysis → core is the allowed
direction; analysis → workspaces is not).
"""

from __future__ import annotations

from typing import Optional

from .repository import ProvenanceRepository
from .store import ArtifactStore
from .types import Artifact, Json
from .versioning import capture_environment, make_parameter_set, make_process_spec


def register_computed_feature(
    kind: str,
    content: bytes,
    *,
    media_type: str,
    analyzer_name: str,
    analyzer_version: str,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    audio: Optional[Artifact] = None,
    model_refs: tuple[str, ...] = (),
    params: Optional[Json] = None,
    code_commit: str = "unknown",
    metadata: Optional[Json] = None,
) -> Artifact:
    """Store ``content`` as a content-addressed ``kind`` artifact produced by a
    versioned run of ``analyzer_name``@``analyzer_version``. If ``audio`` is
    given, the feature derives from it (``analyzed_from``). Returns the artifact.

    Idempotent: identical content ⇒ same sha; identical analyzer/params/env/
    commit ⇒ same ProcessSpec id — only a new run row (append-only history).
    """
    parameter_set = repo.record_parameter_set(
        make_parameter_set(analyzer_name, params or {})
    )
    environment = repo.record_environment_spec(capture_environment())
    spec = repo.record_process_spec(
        make_process_spec(
            name=analyzer_name,
            version=analyzer_version,
            stage="analysis",
            code_commit=code_commit,
            parameter_set=parameter_set,
            environment=environment,
            model_refs=model_refs,
            implementation_hash="",
        )
    )
    run = repo.begin_run_from_spec(spec)
    try:
        art = store.put_bytes(
            content,
            kind=kind,
            media_type=media_type,
            metadata=metadata or {},
        )
        repo.add_output(run, art, role="features")
        if audio is not None:
            repo.add_input(run, audio, role="audio")
            repo.derive(art, audio, run, relation="analyzed_from")
        repo.succeed(run)
        return art
    except Exception as exc:  # the run records its own failure, never swallowed
        repo.fail(run, {"error": repr(exc)})
        raise
