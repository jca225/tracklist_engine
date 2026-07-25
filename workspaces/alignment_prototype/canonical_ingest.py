"""Forward-routing seam — route a newly-computed feature through the registry
into the canonical central provenance store (cutover Decision 2, step 2).

This is the ONE call the analysis stage makes so that new features are canonical
**by construction** — content-addressed, versioned, provenanced — without the
caller knowing any provenance internals. Where ``provenance_backfill`` migrates a
slice of *existing* features, this is the primitive for *new* production:

    from workspaces.alignment_prototype import canonical_ingest as ci
    store, repo = ci.open_central_store()
    audio = ci.register_audio(store, audio_bytes, recording_id="12m8zb3x")
    feats = ci.ingest_feature("analyze.landmark_fp", audio, store=store, repo=repo)

Migration discipline (docs/engine/feature_store_cutover.md, Decision 2): during
cutover this is **additive** — a caller routes through here *in addition to* its
legacy write, never instead of it, until parity is demonstrated and a separate
gated step retires the legacy path. Nothing here mutates legacy storage or pi.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional

from core.provenance import Artifact, ArtifactStore, ProvenanceRepository, connect
from core.provenance.registry import REGISTRY

# Importing producers registers the analyze.* transformations into REGISTRY.
from workspaces.alignment_prototype import producers as _producers  # noqa: F401

#: The single central store the whole corpus's canonical features live in.
DEFAULT_CENTRAL = Path(__file__).resolve().parent / "out" / "provenance" / "central"


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def open_central_store(
    db_root: str | Path | None = None,
) -> tuple[ArtifactStore, ProvenanceRepository]:
    """Open (creating if needed) the central canonical provenance store."""
    root = Path(db_root) if db_root is not None else DEFAULT_CENTRAL
    root.mkdir(parents=True, exist_ok=True)
    conn = connect(root)
    return ArtifactStore(root, conn), ProvenanceRepository(conn)


def register_audio(
    store: ArtifactStore,
    data: bytes,
    *,
    recording_id: str,
    media_type: str = "audio/mp4",
    source_uri: Optional[str] = None,
) -> Artifact:
    """Content-address a recording's audio bytes as an ``audio`` artifact (the
    parent every feature derives from). Idempotent by content hash."""
    return store.put_bytes(
        data,
        kind="audio",
        media_type=media_type,
        metadata={"recording_id": recording_id},
        source_uri=source_uri,
        # Audio is an external rip — a legitimate provenance ROOT. Marking its
        # source_system keeps it from tripping law 1 (artifact with neither a
        # source nor a producing run).
        source_system="tracklist_engine.audio_rip",
    )


def ingest_feature(
    producer: str,
    audio: Artifact,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: Optional[str] = None,
) -> dict[str, Artifact]:
    """Forward-route: run a registered producer on ``audio`` into the store with
    full provenance (ProcessSpec + versioned run + content-addressed output +
    derivation edge). This is the call new analysis makes. Idempotent: identical
    input + producer ⇒ same output sha + same ProcessSpec, only a new run row."""
    return REGISTRY.run(
        producer,
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit=code_commit or _git_commit(),
    )


def ingest_features(
    producers: Iterable[str],
    audio: Artifact,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: Optional[str] = None,
) -> dict[str, Artifact]:
    """Route ``audio`` through several producers; returns role→Artifact merged
    across producers (roles are producer-declared and expected distinct)."""
    commit = code_commit or _git_commit()
    out: dict[str, Artifact] = {}
    for name in producers:
        out.update(
            ingest_feature(name, audio, store=store, repo=repo, code_commit=commit)
        )
    return out


def audio_producers() -> list[str]:
    """Registered transformations that consume an ``audio`` input — the set a
    caller can forward-route a recording through."""
    graph: Mapping[str, Mapping[str, Mapping[str, str]]] = REGISTRY.graph()
    return sorted(
        name for name, sig in graph.items() if "audio" in sig["inputs"].values()
    )
