"""B4 — invalidation/tombstone events (spec v4.2).

A relink (audio was attached to the WRONG recording) or detach (abstain) must
invalidate the content_history generations recorded under the old recording, so
a future historical bind can never resurrect the mis-attached bytes with a
soundness label. The hash is kept (valid=0, audit) but chain(valid_only=True)
hides it — the shape the exporter consumes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.result import Ok
from ingest.content_history import Generation, append_generation, chain
from ingest.corrections import Correction, log_correction

_CORRECTION_SCHEMA = """
CREATE TABLE track_audio_correction (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id TEXT, position TEXT, track_id TEXT NOT NULL,
    axis TEXT NOT NULL, action TEXT NOT NULL,
    old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
    new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
    old_recording_id TEXT, new_recording_id TEXT,
    stem_value TEXT, reason TEXT, source TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem','recording')),
    CHECK (action IN ('replace','add','relink','detach'))
);
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(_CORRECTION_SCHEMA)
    return p


def _seed_gen(db: Path, recording_id="recOLD", stem="acappella", kind="separated"):
    append_generation(
        db,
        Generation(
            recording_id=recording_id,
            stem=stem,
            variant="regular",
            kind=kind,
            content_sha256="mis_attached_bytes",
        ),
    )


def test_relink_tombstones_old_generation(db: Path):
    _seed_gen(db)
    r = log_correction(
        db,
        Correction(
            track_id="recOLD",
            axis="recording",
            action="relink",
            old_recording_id="recOLD",
            new_recording_id="recNEW",
            stem_value="acappella",
            source="same_song_guard",
        ),
    )
    assert isinstance(r, Ok)
    # bind view: the mis-attached generation is gone
    assert (
        chain(db, "recOLD", "acappella", "regular", "separated", valid_only=True) == []
    )
    # audit view: the hash is preserved, marked valid=0 (never dropped)
    all_gens = chain(db, "recOLD", "acappella", "regular", "separated")
    assert len(all_gens) == 1
    assert all_gens[0]["valid"] == 0
    assert all_gens[0]["content_sha256"] == "mis_attached_bytes"


def test_detach_tombstones_old_generation(db: Path):
    _seed_gen(db)
    r = log_correction(
        db,
        Correction(
            track_id="recOLD",
            axis="recording",
            action="detach",
            old_recording_id="recOLD",
            new_recording_id=None,
            stem_value="acappella",
            source="same_song_guard",
        ),
    )
    assert isinstance(r, Ok)
    assert (
        chain(db, "recOLD", "acappella", "regular", "separated", valid_only=True) == []
    )


def test_replace_does_not_tombstone(db: Path):
    """Only relink/detach invalidate. A replace (new bytes, same identity) must
    leave the generation bindable — tombstoning it would drop a sound target."""
    _seed_gen(db)
    r = log_correction(
        db,
        Correction(
            track_id="recOLD",
            axis="version",
            action="replace",
            old_recording_id="recOLD",
            stem_value="acappella",
            source="replace_track_audio",
        ),
    )
    assert isinstance(r, Ok)
    live = chain(db, "recOLD", "acappella", "regular", "separated", valid_only=True)
    assert len(live) == 1 and live[0]["valid"] == 1


def test_relink_scopes_to_stem_axis(db: Path):
    """A relink stamped stem=acappella must not tombstone the recording's
    regular-stem generations (a different chain)."""
    _seed_gen(db, stem="acappella")
    _seed_gen(db, stem="regular", kind="master")
    log_correction(
        db,
        Correction(
            track_id="recOLD",
            axis="recording",
            action="relink",
            old_recording_id="recOLD",
            new_recording_id="recNEW",
            stem_value="acappella",
            source="guard",
        ),
    )
    assert (
        chain(db, "recOLD", "acappella", "regular", "separated", valid_only=True) == []
    )
    reg = chain(db, "recOLD", "regular", "regular", "master", valid_only=True)
    assert len(reg) == 1  # untouched


def test_relink_noop_when_history_table_absent(db: Path):
    """relink logs cleanly even if content_history was never created (auto-pull
    landed corrections before the ledger migration)."""
    r = log_correction(
        db,
        Correction(
            track_id="recOLD",
            axis="recording",
            action="relink",
            old_recording_id="recOLD",
            new_recording_id="recNEW",
            stem_value="acappella",
            source="guard",
        ),
    )
    assert isinstance(r, Ok)
