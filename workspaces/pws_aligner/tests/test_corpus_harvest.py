"""Corpus-harvest CLI: DB→cases→ledger glue + eligibility census.

Tests the thin flywheel-step-2 runner against an in-memory fixture DB and a
FAKE scorer — no pi access, no real audio. All alignment logic is delegated to
the already-tested harvest/seam machinery; here we test the corpus glue: which
slots are eligible, how cases are built, that banding gates per axis, that the
ledger is idempotent, and that the census classifies blockers correctly.
"""

from __future__ import annotations

import json
import sqlite3

from pathlib import Path

from workspaces.pws_aligner.corpus_harvest import query_corpus_slots
from workspaces.pws_aligner.corpus_harvest import (  # noqa: E402
    DEFAULT_SPAN_S,
    CorpusSlot,
    build_corpus_cases,
)
from workspaces.alignment_prototype.harness.contract import AlignmentResult  # noqa: E402
from workspaces.pws_aligner.corpus_harvest import run_corpus_harvest  # noqa: E402


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
        (set_audio_id, set_id, path, None, is_reference),  # sha256=None
    )


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


def _slot(**kw) -> CorpusSlot:
    base = dict(
        set_id="S1",
        set_audio_id=10,
        slot_label="001",
        recording_id="R1",
        ref_path="/ref/r1.flac",
        claimed_version="original",
        claimed_stem="regular",
        claimed_variant="regular",
        cue_time_s=120.0,
        duration_s=55.0,
        mix_full_path="/mix/s1.m4a",
    )
    base.update(kw)
    return CorpusSlot(**base)


def test_build_cases_maps_slot_to_candidate_span_axes():
    cases = build_corpus_cases([_slot()])
    assert len(cases) == 1
    cand, span, axes = cases[0]
    assert cand.recording_id == "R1"
    assert cand.source_path == "/ref/r1.flac"
    assert cand.stem == "regular"
    assert cand.version == "original"
    assert cand.variant == "regular"
    assert cand.source_url.startswith("corpus://")
    assert span.set_id == "S1"
    assert span.slot_label == "001"
    assert span.set_start_s == 120.0
    assert span.span_dur_s == 55.0
    assert axes == {"version": "original", "stem": "regular", "variant": "regular"}


def test_build_cases_defaults_span_when_no_duration():
    cases = build_corpus_cases([_slot(duration_s=None)])
    _, span, _ = cases[0]
    assert span.span_dur_s == DEFAULT_SPAN_S


def test_build_cases_applies_ref_audio_root_to_relative_paths():
    cases = build_corpus_cases(
        [_slot(ref_path="rel/r1.flac")], ref_audio_root=Path("/root")
    )
    cand, _, _ = cases[0]
    assert cand.source_path == "/root/rel/r1.flac"
    # absolute paths untouched
    cases2 = build_corpus_cases(
        [_slot(ref_path="/abs/r1.flac")], ref_audio_root=Path("/root")
    )
    assert cases2[0][0].source_path == "/abs/r1.flac"


def _agree(rec_id, sources, *, offset=12.0, conf=0.8):
    return [
        AlignmentResult(
            recording_id=rec_id, offset_s=offset, confidence=conf, source=src
        )
        for src in sources
    ]


def test_run_harvest_writes_only_accepts(tmp_path):
    # two regular slots in one set; scorer agrees (2 channels) → both ACCEPT.
    slots = [
        _slot(recording_id="R1", slot_label="001", cue_time_s=100.0),
        _slot(recording_id="R2", slot_label="002", cue_time_s=200.0),
    ]

    def factory(mix_full_path, mix_stem_dir):
        def scorer(cand, span):
            return _agree(cand.recording_id, ("fp", "chroma"))

        return scorer

    out = tmp_path / "ledger.jsonl"
    summary = run_corpus_harvest(
        slots, stems_root=tmp_path, out=out, scorer_factory=factory
    )
    assert summary.n_sets == 1
    assert summary.n_cases == 2
    assert summary.n_harvested == 2
    assert summary.n_written == 2
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert {r["recording_id"] for r in lines} == {"R1", "R2"}
    assert all(r["stem"] == "regular" for r in lines)


def test_run_harvest_instrumental_needs_three_channels(tmp_path):
    slots = [_slot(recording_id="RI", slot_label="003", claimed_stem="instrumental")]

    def two_channel(mix_full_path, mix_stem_dir):
        def scorer(cand, span):
            return _agree(cand.recording_id, ("fp", "chroma"))  # only 2 agree

        return scorer

    out = tmp_path / "ledger.jsonl"
    summary = run_corpus_harvest(
        slots, stems_root=tmp_path, out=out, scorer_factory=two_channel
    )
    assert summary.n_harvested == 0  # instrumental banded < ACCEPT at 2 channels
    assert not out.exists() or out.read_text().strip() == ""

    def three_channel(mix_full_path, mix_stem_dir):
        def scorer(cand, span):
            return _agree(cand.recording_id, ("fp", "chroma", "continuity"))

        return scorer

    out2 = tmp_path / "ledger2.jsonl"
    summary2 = run_corpus_harvest(
        slots, stems_root=tmp_path, out=out2, scorer_factory=three_channel
    )
    assert summary2.n_harvested == 1


def test_run_harvest_is_idempotent(tmp_path):
    slots = [_slot(recording_id="R1", slot_label="001", cue_time_s=100.0)]

    def factory(mix_full_path, mix_stem_dir):
        return lambda cand, span: _agree(cand.recording_id, ("fp", "chroma"))

    out = tmp_path / "ledger.jsonl"
    first = run_corpus_harvest(
        slots, stems_root=tmp_path, out=out, scorer_factory=factory
    )
    second = run_corpus_harvest(
        slots, stems_root=tmp_path, out=out, scorer_factory=factory
    )
    assert first.n_written == 1
    assert second.n_written == 0  # span_key already present
    assert len([x for x in out.read_text().splitlines() if x.strip()]) == 1


def test_run_harvest_builds_one_scorer_per_set(tmp_path):
    # two slots share set_audio_id → factory called once; stem dir routed by id.
    slots = [
        _slot(
            set_id="S1",
            set_audio_id=77,
            recording_id="R1",
            slot_label="001",
            cue_time_s=100.0,
        ),
        _slot(
            set_id="S1",
            set_audio_id=77,
            recording_id="R2",
            slot_label="002",
            cue_time_s=200.0,
        ),
    ]
    calls = []

    def factory(mix_full_path, mix_stem_dir):
        calls.append((Path(mix_full_path), Path(mix_stem_dir)))
        return lambda cand, span: _agree(cand.recording_id, ("fp", "chroma"))

    out = tmp_path / "ledger.jsonl"
    run_corpus_harvest(
        slots, stems_root=tmp_path / "stems", out=out, scorer_factory=factory
    )
    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "stems" / "77"
