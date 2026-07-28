from __future__ import annotations

"""Tests for ref-quality → AbstentionEvent → actuation queue wiring (B2).

Detect-then-correct: identity_gate verdicts and a duration/preview check
become typed AbstentionEvents (NO_DATA for missing, OUT_OF_DOMAIN for
wrong-version/preview); collect_actuations reduces them to a deduped,
budget-gated re-acquire queue written to JSON. No downloads here — the
ingest pipeline is the effector that drains the queue.
"""

import json
import sqlite3
from pathlib import Path

from ingest.identity_gate import Verdict, VerifyResult
from pws_aligner.quality.ref_quality import (
    event_from_verify,
    events_for_slots,
    preview_event,
    suspects_to_events,
    write_reacquire_queue,
)
from pws_aligner.core.votes import AbstainReason


def _mk_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE track_audio (
                track_audio_id INTEGER PRIMARY KEY,
                recording_id TEXT, track_id TEXT, path TEXT,
                stem TEXT, is_reference INTEGER, downloaded_at TEXT
            )"""
        )
    return db


def test_missing_verdicts_map_to_no_data():
    for verdict in (Verdict.NO_REFERENCE, Verdict.MISSING_FILE):
        ev = event_from_verify("set1", "3", "rec_a", VerifyResult(verdict, "gone"))
        assert ev is not None
        assert ev.reason == AbstainReason.NO_DATA


def test_wrong_version_verdicts_map_to_out_of_domain():
    for verdict in (
        Verdict.WRONG_SONG,
        Verdict.DURATION_MISMATCH,
        Verdict.LIVE_SUSPECT,
        Verdict.FALLBACK_TO_ORIGINAL,
    ):
        ev = event_from_verify("set1", "3", "rec_a", VerifyResult(verdict, "sus"))
        assert ev is not None
        assert ev.reason == AbstainReason.OUT_OF_DOMAIN


def test_ok_verdict_yields_no_event():
    assert event_from_verify("s", "1", "r", VerifyResult(Verdict.OK)) is None


def test_weak_signal_is_low_margin_never_actuates():
    """Ambiguous evidence → LOW_MARGIN event (traceable) that must not actuate."""
    ev = event_from_verify("s", "1", "r", VerifyResult(Verdict.WEAK_SIGNAL, "meh"))
    assert ev is not None
    assert ev.reason == AbstainReason.LOW_MARGIN


def test_preview_clip_detected():
    """A 46s file against a ~4min expected duration is a preview-clip suspect."""
    ev = preview_event("s", "1", "rec_p", actual_s=46.0, expected_s=240.0)
    assert ev is not None
    assert ev.reason == AbstainReason.OUT_OF_DOMAIN
    assert "preview" in ev.detail.lower()


def test_full_length_not_flagged():
    assert preview_event("s", "1", "r", actual_s=235.0, expected_s=240.0) is None


def test_short_but_unknown_expected_not_flagged():
    """No scraped duration → cannot distinguish short edit from preview → no event."""
    assert preview_event("s", "1", "r", actual_s=46.0, expected_s=None) is None


def test_suspects_to_events():
    """External suspect lists (corpus_integrity / preview-clip triage) become events."""
    suspects = [
        {
            "set_id": "s1",
            "slot_label": "2",
            "recording_id": "rec_x",
            "kind": "wrong_version",
            "detail": "rescue grabbed ORIGINAL from preview clip",
        },
        {
            "set_id": "s1",
            "slot_label": "4",
            "recording_id": "rec_y",
            "kind": "missing",
            "detail": "",
        },
    ]
    events = suspects_to_events(suspects)
    assert len(events) == 2
    assert events[0].reason == AbstainReason.OUT_OF_DOMAIN
    assert events[1].reason == AbstainReason.NO_DATA


def test_events_for_slots_against_db(tmp_path):
    db = _mk_db(tmp_path)
    present = tmp_path / "present.mp3"
    present.write_bytes(b"x")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO track_audio VALUES (1, 'rec_ok', 'rec_ok', ?, 'regular', 1, '2026-01-01')",
            (str(present),),
        )
        conn.execute(
            "INSERT INTO track_audio VALUES (2, 'rec_gone', 'rec_gone', ?, 'regular', 1, '2026-01-01')",
            (str(tmp_path / "nope.mp3"),),
        )
    slots = [
        {"set_id": "bb", "slot_label": "1", "recording_id": "rec_ok"},
        {"set_id": "bb", "slot_label": "2", "recording_id": "rec_gone"},
        {"set_id": "bb", "slot_label": "3", "recording_id": "rec_norow"},
    ]
    events = events_for_slots(db, slots)
    by_rid = {e.recording_id: e for e in events}
    assert "rec_ok" not in by_rid  # healthy reference → no event
    assert by_rid["rec_gone"].reason == AbstainReason.NO_DATA
    assert by_rid["rec_norow"].reason == AbstainReason.NO_DATA


def test_write_reacquire_queue_dedup_budget_kinds(tmp_path):
    db = _mk_db(tmp_path)
    slots = [
        {"set_id": "bb1", "slot_label": "1", "recording_id": "rec_m"},
        {"set_id": "bb2", "slot_label": "9", "recording_id": "rec_m"},  # dup rid
        {"set_id": "bb1", "slot_label": "2", "recording_id": "rec_spent"},
    ]
    events = events_for_slots(db, slots)
    events += suspects_to_events(
        [
            {
                "set_id": "bb1",
                "slot_label": "5",
                "recording_id": "rec_wv",
                "kind": "wrong_version",
                "detail": "preview-clip original",
            }
        ]
    )
    out = tmp_path / "queue.json"
    requests = write_reacquire_queue(events, attempts={"rec_spent": 2}, out_path=out)
    doc = json.loads(out.read_text())
    rids = [r["recording_id"] for r in doc]
    assert rids.count("rec_m") == 1  # deduped across sets
    assert "rec_spent" not in rids  # budget-gated out
    kinds = {r["recording_id"]: r["kind"] for r in doc}
    assert kinds["rec_m"] == "fetch_missing"
    assert kinds["rec_wv"] == "reacquire_wrong_version"
    assert len(requests) == len(doc)
