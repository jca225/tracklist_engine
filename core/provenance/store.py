"""Content-addressed artifact store (§2) — bytes are identity, paths are not.

A ``put`` hashes the content (sha256), writes it once under ``objects/aa/<sha>``
(deduped), and records an ``Artifact`` row. ``verify`` re-hashes the stored bytes,
so a corrupted or path-swapped object is caught (Law 3). The object dir + SQLite
index live under one root; the same DB backs the ProvenanceRepository so lineage
edges reference real artifact rows.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from .types import Artifact, ArtifactKind, Json

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifact (
    content_sha256 TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    media_type     TEXT NOT NULL,
    byte_size      INTEGER NOT NULL,
    object_uri     TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    source_uri     TEXT,
    source_system  TEXT,
    metadata_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS run (
    run_id          TEXT PRIMARY KEY,
    process         TEXT NOT NULL,
    process_version TEXT NOT NULL,
    code_commit     TEXT NOT NULL,
    params_hash     TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    random_seed     INTEGER,
    error_json      TEXT
);
CREATE TABLE IF NOT EXISTS run_input (
    run_id         TEXT NOT NULL REFERENCES run(run_id),
    content_sha256 TEXT NOT NULL REFERENCES artifact(content_sha256),
    role           TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    PRIMARY KEY (run_id, content_sha256, role)
);
CREATE TABLE IF NOT EXISTS run_output (
    run_id         TEXT NOT NULL REFERENCES run(run_id),
    content_sha256 TEXT NOT NULL REFERENCES artifact(content_sha256),
    role           TEXT NOT NULL,
    ordinal        INTEGER NOT NULL,
    PRIMARY KEY (run_id, content_sha256, role)
);
CREATE TABLE IF NOT EXISTS derivation (
    child_sha256  TEXT NOT NULL REFERENCES artifact(content_sha256),
    parent_sha256 TEXT NOT NULL REFERENCES artifact(content_sha256),
    run_id        TEXT NOT NULL REFERENCES run(run_id),
    relation      TEXT NOT NULL,
    PRIMARY KEY (child_sha256, parent_sha256, run_id)
);
CREATE TABLE IF NOT EXISTS observation (
    observation_id   TEXT PRIMARY KEY,
    subject_type     TEXT NOT NULL,
    subject_id       TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    value_json       TEXT,
    source_sha256    TEXT NOT NULL REFERENCES artifact(content_sha256),
    producer_run_id  TEXT NOT NULL REFERENCES run(run_id),
    observed_at      TEXT NOT NULL,
    status           TEXT NOT NULL,
    source_confidence REAL,
    diagnostic_code  TEXT
);
CREATE TABLE IF NOT EXISTS claim (
    claim_id          TEXT PRIMARY KEY,
    axis              TEXT NOT NULL,
    subject_type      TEXT NOT NULL,
    subject_id        TEXT NOT NULL,
    predicate         TEXT NOT NULL,
    candidate_type    TEXT,
    candidate_id      TEXT,
    context_json      TEXT NOT NULL DEFAULT '{}',
    created_by_run_id TEXT NOT NULL REFERENCES run(run_id)
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id       TEXT PRIMARY KEY,
    claim_id          TEXT NOT NULL REFERENCES claim(claim_id),
    axis              TEXT NOT NULL,
    source_family     TEXT NOT NULL,
    producer_run_id   TEXT NOT NULL REFERENCES run(run_id),
    direction         TEXT NOT NULL,
    native_score_json TEXT NOT NULL DEFAULT '{}',
    uncertainty_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS belief (
    belief_id                     TEXT PRIMARY KEY,
    subject_type                  TEXT NOT NULL,
    subject_id                    TEXT NOT NULL,
    axis                          TEXT NOT NULL,
    inference_run_id              TEXT NOT NULL REFERENCES run(run_id),
    posterior_json                TEXT NOT NULL,
    entropy                       REAL NOT NULL,
    decision                      TEXT NOT NULL,
    chosen                        TEXT,
    contributing_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    supersedes_belief_id          TEXT
);
"""


def connect(root: Path) -> sqlite3.Connection:
    """Open (creating if needed) the provenance index DB under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "provenance.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactStore:
    """Content-addressed bytes + the ``artifact`` index."""

    def __init__(self, root: Path, conn: sqlite3.Connection) -> None:
        self._root = root
        self._objects = root / "objects"
        self._conn = conn

    def _object_path(self, sha: str) -> Path:
        return self._objects / sha[:2] / sha

    def put_bytes(
        self,
        content: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        metadata: Json | None = None,
        source_uri: str | None = None,
        source_system: str | None = None,
    ) -> Artifact:
        import json

        sha = hashlib.sha256(content).hexdigest()
        path = self._object_path(sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        art = Artifact(
            content_sha256=sha,
            kind=kind,
            media_type=media_type,
            byte_size=len(content),
            object_uri=str(path),
            created_at=_now(),
            source_uri=source_uri,
            source_system=source_system,
            metadata=metadata or {},
        )
        # Dedupe: identical bytes -> one row (idempotent put).
        self._conn.execute(
            "INSERT OR IGNORE INTO artifact(content_sha256, kind, media_type, "
            "byte_size, object_uri, created_at, source_uri, source_system, "
            "metadata_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sha,
                kind.value,
                media_type,
                art.byte_size,
                art.object_uri,
                art.created_at.isoformat(),
                source_uri,
                source_system,
                json.dumps(art.metadata),
            ),
        )
        self._conn.commit()
        return art

    def put_file(
        self,
        file_path: Path,
        *,
        kind: ArtifactKind,
        media_type: str,
        metadata: Json | None = None,
        source_uri: str | None = None,
    ) -> Artifact:
        return self.put_bytes(
            Path(file_path).read_bytes(),
            kind=kind,
            media_type=media_type,
            metadata=metadata,
            source_uri=source_uri,
        )

    def open(self, content_sha256: str) -> BinaryIO:
        return self._object_path(content_sha256).open("rb")

    def verify(self, content_sha256: str) -> bool:
        """Re-hash the stored bytes; a mismatch means corruption / wrong object."""
        path = self._object_path(content_sha256)
        if not path.exists():
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == content_sha256
