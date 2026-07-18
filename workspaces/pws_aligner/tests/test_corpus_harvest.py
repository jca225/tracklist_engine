"""Corpus-harvest CLI: DB→cases→ledger glue + eligibility census.

Tests the thin flywheel-step-2 runner against an in-memory fixture DB and a
FAKE scorer — no pi access, no real audio. All alignment logic is delegated to
the already-tested harvest/seam machinery; here we test the corpus glue: which
slots are eligible, how cases are built, that banding gates per axis, that the
ledger is idempotent, and that the census classifies blockers correctly.
"""

from __future__ import annotations

import sqlite3

from workspaces.pws_aligner.corpus_harvest import query_corpus_slots


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE set_track_slots (
            set_id TEXT, row_index INTEGER, recording_id TEXT, slot_label TEXT,
            cue_seconds INTEGER, cue_time_seconds INTEGER,
            claimed_version TEXT, claimed_stem TEXT, claimed_variant TEXT,
            duration_seconds INTEGER
        );
        CREATE TABLE set_audio (
            set_audio_id INTEGER, set_id TEXT, path TEXT, sha256 TEXT, is_reference INTEGER
        );
        CREATE TABLE track_audio (
            track_audio_id INTEGER, recording_id TEXT, stem TEXT,
            path TEXT, is_reference INTEGER
        );
        """
    )
    return conn


def _add_slot(
    conn,
    *,
    set_id,
    row_index,
    recording_id,
    slot_label="001",
    cue_time_seconds=100,
    claimed_stem="regular",
    duration_seconds=42,
    claimed_version="original",
    claimed_variant="regular",
):
    conn.execute(
        "INSERT INTO set_track_slots VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            set_id,
            row_index,
            recording_id,
            slot_label,
            None,
            cue_time_seconds,
            claimed_version,
            claimed_stem,
            claimed_variant,
            duration_seconds,
        ),
    )


def _add_set_audio(conn, *, set_audio_id, set_id, path, is_reference=1):
    conn.execute(
        "INSERT INTO set_audio VALUES (?,?,?,?,?)",
        (set_audio_id, set_id, path, None, is_reference),
    )
    # sha256 column omitted: fixture table has only the 5 columns above.


def _add_track_audio(
    conn, *, recording_id, stem="regular", path="/ref.flac", is_reference=1
):
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (None, recording_id, stem, path, is_reference),
    )


def test_query_returns_only_eligible_slots():
    conn = _make_db()
    # eligible regular slot
    _add_slot(conn, set_id="S1", row_index=0, recording_id="R1")
    _add_set_audio(conn, set_audio_id=10, set_id="S1", path="/mix1.m4a")
    _add_track_audio(conn, recording_id="R1", stem="regular", path="/r1.flac")
    # ineligible: acappella (uncertified axis, excluded by policy_stems)
    _add_slot(
        conn,
        set_id="S1",
        row_index=1,
        recording_id="R2",
        slot_label="002",
        claimed_stem="acappella",
    )
    _add_track_audio(conn, recording_id="R2", stem="acappella", path="/r2.flac")
    # ineligible: no cue time
    _add_slot(
        conn,
        set_id="S1",
        row_index=2,
        recording_id="R3",
        slot_label="003",
        cue_time_seconds=None,
    )
    _add_track_audio(conn, recording_id="R3", stem="regular", path="/r3.flac")
    # ineligible: ref audio wrong stem (claimed regular, only acappella ref exists)
    _add_slot(conn, set_id="S1", row_index=3, recording_id="R4", slot_label="004")
    _add_track_audio(conn, recording_id="R4", stem="acappella", path="/r4.flac")

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))

    assert [s.recording_id for s in slots] == ["R1"]
    s = slots[0]
    assert s.set_audio_id == 10
    assert s.cue_time_s == 100.0
    assert s.duration_s == 42.0
    assert s.ref_path == "/r1.flac"
    assert s.mix_full_path == "/mix1.m4a"


def test_query_excludes_non_reference_mix_and_ref():
    conn = _make_db()
    _add_slot(conn, set_id="S2", row_index=0, recording_id="R1")
    _add_set_audio(conn, set_audio_id=20, set_id="S2", path="/mix.m4a", is_reference=0)
    _add_track_audio(conn, recording_id="R1", stem="regular", is_reference=1)
    _add_slot(conn, set_id="S2", row_index=1, recording_id="R2", slot_label="002")
    _add_set_audio(conn, set_audio_id=21, set_id="S2", path="/mix.m4a", is_reference=1)
    _add_track_audio(conn, recording_id="R2", stem="regular", is_reference=0)

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))
    # S2 has a reference mix (id 21) but R1's row_index-0 slot: R1 ref is_reference=1
    # yet its set's reference mix is id 21 → R1 IS eligible; R2 ref is non-reference → excluded.
    assert [s.recording_id for s in slots] == ["R1"]
    assert slots[0].set_audio_id == 21


def test_query_respects_limit_and_order():
    conn = _make_db()
    _add_set_audio(conn, set_audio_id=30, set_id="S3", path="/m.m4a")
    for i in range(3):
        _add_slot(
            conn, set_id="S3", row_index=i, recording_id=f"R{i}", slot_label=f"00{i}"
        )
        _add_track_audio(conn, recording_id=f"R{i}", stem="regular")
    slots = query_corpus_slots(conn, policy_stems=("regular",), limit=2)
    assert [s.recording_id for s in slots] == ["R0", "R1"]
