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

from workspaces.alignment_prototype.harness.contract import AlignmentResult
from workspaces.pws_aligner.corpus_harvest import (
    DEFAULT_SPAN_S,
    CorpusSlot,
    build_corpus_cases,
    census,
    census_rows,
    main,
    query_corpus_slots,
    run_corpus_harvest,
)


_SCHEMA = """
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
    track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT,
    path TEXT, is_reference INTEGER
);
"""


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
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


def test_query_includes_slot_when_neither_mix_nor_ref_is_reference():
    """Fix 1: is_reference=0 on BOTH mix and ref must not block harvest.

    The real corpus has set_audio.is_reference=1 on only 2/1016 sets and
    track_audio.is_reference=1 on 0 instrumental refs; strict joins returned ~0.
    The fix uses deterministic MIN-id picks instead of is_reference filters.
    """
    conn = _make_db()
    _add_slot(conn, set_id="S2", row_index=0, recording_id="R1")
    # mix is_reference=0 — OLD code would have excluded this slot
    _add_set_audio(conn, set_audio_id=20, set_id="S2", path="/mix.m4a", is_reference=0)
    # ref is_reference=0 — OLD code would have excluded this slot
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (50, "R1", "regular", "/r1.flac", 0),
    )

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))
    # NEW: slot IS returned even though both is_reference=0
    assert [s.recording_id for s in slots] == ["R1"]
    assert slots[0].set_audio_id == 20
    assert slots[0].ref_path == "/r1.flac"


def test_query_deterministic_ref_pick_uses_min_track_audio_id():
    """Fix 1: two track_audio rows for same (recording_id, stem) → exactly one slot
    using the MIN(track_audio_id) row, not a fan-out.
    """
    conn = _make_db()
    _add_slot(conn, set_id="S3", row_index=0, recording_id="R1")
    _add_set_audio(conn, set_audio_id=30, set_id="S3", path="/mix.m4a")
    # Two track_audio rows — id 200 (later) and id 100 (earlier/lower)
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (200, "R1", "regular", "/r1_v2.flac", 0),
    )
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (100, "R1", "regular", "/r1_v1.flac", 0),
    )

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))
    # Exactly one slot (not a fan-out)
    assert len(slots) == 1
    # Picks the MIN(track_audio_id) = 100 row
    assert slots[0].ref_path == "/r1_v1.flac"


def test_query_excludes_zero_cue_seconds_without_cue_time_seconds():
    """Fix 2: cue_seconds=0 with cue_time_seconds=NULL → slot must be EXCLUDED.

    Old code used COALESCE(cue_time_seconds, cue_seconds) which resolved to 0,
    a truthy integer that passed the IS NOT NULL filter. Zero is not a valid
    placement anchor — cue_time_seconds>0 is the real gate.
    """
    conn = _make_db()
    _add_slot(
        conn,
        set_id="S4",
        row_index=0,
        recording_id="R1",
        cue_time_seconds=None,  # no real cue
    )
    # Manually insert with cue_seconds=0 (the sentinel)
    conn.execute("UPDATE set_track_slots SET cue_seconds=0 WHERE set_id='S4'")
    _add_set_audio(conn, set_audio_id=40, set_id="S4", path="/mix.m4a")
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (400, "R1", "regular", "/r1.flac", 0),
    )

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))
    assert slots == [], "cue_seconds=0 with no cue_time_seconds must be excluded"


def test_query_includes_slot_with_valid_cue_time_seconds():
    """Fix 2: cue_time_seconds=90, cue_seconds=0 → slot IS included, cue_time_s=90.0."""
    conn = _make_db()
    _add_slot(
        conn,
        set_id="S5",
        row_index=0,
        recording_id="R1",
        cue_time_seconds=90,
    )
    # Simulate cue_seconds=0 (old sentinel, should not poison the result)
    conn.execute("UPDATE set_track_slots SET cue_seconds=0 WHERE set_id='S5'")
    _add_set_audio(conn, set_audio_id=50, set_id="S5", path="/mix.m4a")
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (500, "R1", "regular", "/r1.flac", 0),
    )

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))
    assert len(slots) == 1
    assert slots[0].cue_time_s == 90.0


def test_census_rows_left_join_preserves_slot_with_no_ref_audio():
    """Fix 1 (census): slot with no matching track_audio still appears in census_rows
    so the classifier can label it no-ref-audio (LEFT JOIN preserved).
    """
    conn = _make_db()
    _add_slot(conn, set_id="S6", row_index=0, recording_id="RX", claimed_stem="regular")
    _add_set_audio(conn, set_audio_id=60, set_id="S6", path="/mix.m4a")
    # Intentionally NO track_audio for RX/regular — old LEFT JOIN already handled this,
    # but the new MIN-id subquery form must not accidentally turn it into an INNER JOIN.

    rows = census_rows(conn, policy_stems=("regular",))
    assert len(rows) == 1, "slot with no ref audio must still appear in census_rows"
    assert rows[0]["ref_path"] is None


def test_census_classifies_zero_cue_seconds_as_no_cue_time():
    """Fix 2 (census): cue_time_seconds=NULL, cue_seconds=0 → classified no-cue-time."""
    conn = _make_db()
    _add_slot(
        conn,
        set_id="S7",
        row_index=0,
        recording_id="R1",
        claimed_stem="regular",
        cue_time_seconds=None,
    )
    conn.execute("UPDATE set_track_slots SET cue_seconds=0 WHERE set_id='S7'")
    # Add mix audio so the mix-file check doesn't mask the cue check
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
        mix_path = f.name
        f.write(b"x")
    try:
        _add_set_audio(
            conn, set_audio_id=70, set_id="S7", path=mix_path, is_reference=0
        )
        conn.execute(
            "INSERT INTO track_audio VALUES (?,?,?,?,?)",
            (700, "R1", "regular", "/r1.flac", 0),
        )
        from pathlib import Path
        import tempfile as tf

        stems = Path(tf.mkdtemp())
        report = census(conn, stems_root=stems, policy_stems=("regular",))
        assert report.by_axis["regular"]["no-cue-time"] == 1
        assert report.by_axis["regular"].get("eligible-now", 0) == 0
    finally:
        os.unlink(mix_path)


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
    assert summary.n_written == 0  # gate test: nothing reaches the ledger
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


def test_census_classifies_blockers_per_axis(tmp_path):
    conn = _make_db()
    stems_root = tmp_path / "stems"
    # 1) eligible-now regular: mix file present, ref present, cue present
    mix1 = tmp_path / "mix1.m4a"
    mix1.write_bytes(b"x")
    _add_slot(conn, set_id="A", row_index=0, recording_id="R1", claimed_stem="regular")
    _add_set_audio(conn, set_audio_id=1, set_id="A", path=str(mix1))
    _add_track_audio(conn, recording_id="R1", stem="regular")
    # 2) no-cue-time regular
    _add_slot(
        conn,
        set_id="A",
        row_index=1,
        recording_id="R2",
        slot_label="002",
        claimed_stem="regular",
        cue_time_seconds=None,
    )
    _add_track_audio(conn, recording_id="R2", stem="regular")
    # 3) no-ref-audio regular (no track_audio row)
    _add_slot(
        conn,
        set_id="A",
        row_index=2,
        recording_id="R3",
        slot_label="003",
        claimed_stem="regular",
    )
    # 4) no-mix-audio regular (set B has no set_audio row)
    _add_slot(conn, set_id="B", row_index=0, recording_id="R4", claimed_stem="regular")
    _add_track_audio(conn, recording_id="R4", stem="regular")
    # 5) instrumental, mix present but no instrumental.flac on disk → no-mix-stem
    mix2 = tmp_path / "mix2.m4a"
    mix2.write_bytes(b"x")
    _add_slot(
        conn, set_id="C", row_index=0, recording_id="R5", claimed_stem="instrumental"
    )
    _add_set_audio(conn, set_audio_id=5, set_id="C", path=str(mix2))
    _add_track_audio(conn, recording_id="R5", stem="instrumental")
    # 6) instrumental eligible-now: instrumental.flac present
    mix3 = tmp_path / "mix3.m4a"
    mix3.write_bytes(b"x")
    (stems_root / "6").mkdir(parents=True)
    (stems_root / "6" / "instrumental.flac").write_bytes(b"x")
    _add_slot(
        conn, set_id="D", row_index=0, recording_id="R6", claimed_stem="instrumental"
    )
    _add_set_audio(conn, set_audio_id=6, set_id="D", path=str(mix3))
    _add_track_audio(conn, recording_id="R6", stem="instrumental")
    # excluded entirely: acappella (uncertified) — must not appear in any axis bucket
    _add_slot(
        conn,
        set_id="D",
        row_index=1,
        recording_id="R7",
        slot_label="002",
        claimed_stem="acappella",
    )
    _add_track_audio(conn, recording_id="R7", stem="acappella")

    report = census(
        conn, stems_root=stems_root, policy_stems=("regular", "instrumental")
    )

    assert set(report.by_axis) == {"regular", "instrumental"}
    assert report.by_axis["regular"]["eligible-now"] == 1
    assert report.by_axis["regular"]["no-cue-time"] == 1
    assert report.by_axis["regular"]["no-ref-audio"] == 1
    assert report.by_axis["regular"]["no-mix-audio"] == 1
    assert report.by_axis["instrumental"]["no-mix-stem"] == 1
    assert report.by_axis["instrumental"]["eligible-now"] == 1
    assert report.total() == 6
    # to_json + render are well-formed
    assert report.to_json()["total"] == 6
    assert "eligible-now" in report.render()


def test_census_rows_excludes_uncertified_axis(tmp_path):
    conn = _make_db()
    _add_slot(
        conn, set_id="A", row_index=0, recording_id="R1", claimed_stem="acappella"
    )
    rows = census_rows(conn, policy_stems=("regular", "instrumental"))
    assert rows == []


def _write_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.executescript(
        """
        INSERT INTO set_track_slots VALUES
            ('A',0,'R1','001',NULL,100,'original','regular','regular',42),
            ('A',1,'R2','002',NULL,NULL,'original','regular','regular',42);
        INSERT INTO set_audio VALUES (1,'A','/nope/mix.m4a',NULL,1);
        INSERT INTO track_audio VALUES (NULL,'R1','regular','/nope/r1.flac',1);
        INSERT INTO track_audio VALUES (NULL,'R2','regular','/nope/r2.flac',1);
        """
    )
    conn.commit()
    conn.close()


def test_main_census_mode_runs_on_fixture_db(tmp_path, capsys):
    db = tmp_path / "fix.db"
    _write_fixture_db(db)
    rc = main(["--db", str(db), "--stems-root", str(tmp_path / "stems"), "--census"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "eligibility census" in out
    # R1 has cue+ref but mix file absent → no-mix-audio; R2 no cue → no-cue-time
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["by_axis"]["regular"]["no-mix-audio"] == 1
    assert payload["by_axis"]["regular"]["no-cue-time"] == 1


def test_main_rejects_uncertified_stem(tmp_path):
    db = tmp_path / "fix.db"
    _write_fixture_db(db)
    try:
        main(["--db", str(db), "--stem", "acappella", "--census"])
        assert False, "expected SystemExit on uncertified stem"
    except SystemExit as e:
        assert e.code != 0


def test_main_harvest_mode_requires_out(tmp_path):
    db = tmp_path / "fix.db"
    _write_fixture_db(db)
    try:
        main(["--db", str(db)])  # no --out, no --census
        assert False, "expected SystemExit when --out missing"
    except SystemExit as e:
        assert e.code != 0


def test_census_first_missing_priority(tmp_path):
    """A slot missing both cue_time AND ref audio must be classified no-cue-time
    (the first-missing priority chain)."""
    conn = _make_db()
    stems_root = tmp_path / "stems"
    mix = tmp_path / "mix.m4a"
    mix.write_bytes(b"x")
    # Add slot with no cue time AND no track_audio row
    _add_slot(
        conn,
        set_id="X",
        row_index=0,
        recording_id="RX",
        claimed_stem="regular",
        cue_time_seconds=None,
    )
    _add_set_audio(conn, set_audio_id=99, set_id="X", path=str(mix))
    # Intentionally no _add_track_audio call → no ref audio

    report = census(
        conn, stems_root=stems_root, policy_stems=("regular", "instrumental")
    )

    assert report.by_axis["regular"]["no-cue-time"] == 1
    assert report.by_axis["regular"].get("no-ref-audio", 0) == 0


# ---------------------------------------------------------------------------
# NEW TESTS: three data-vs-assumption mismatches fixed in corpus_harvest
# ---------------------------------------------------------------------------


def test_query_relaxes_is_reference_joins():
    """Fix 1: is_reference=0 on both mix and ref does NOT block a slot.

    A set with only a non-reference set_audio row and a recording with only a
    non-reference track_audio row should still be returned by query_corpus_slots
    now that the is_reference=1 filter is replaced by deterministic MIN-id picks.
    """
    conn = _make_db()
    # mix is_reference=0
    _add_set_audio(conn, set_audio_id=50, set_id="S50", path="/mix50.m4a", is_reference=0)
    # ref is_reference=0 (explicit insert to set track_audio_id)
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (50, "R50", "regular", "/ref50.flac", 0),
    )
    _add_slot(conn, set_id="S50", row_index=0, recording_id="R50", cue_time_seconds=60)

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))

    assert len(slots) == 1
    assert slots[0].recording_id == "R50"
    assert slots[0].set_audio_id == 50
    assert slots[0].ref_path == "/ref50.flac"
    assert slots[0].mix_full_path == "/mix50.m4a"


def test_query_deterministic_min_id_ref_pick():
    """Fix 1b: two track_audio rows for the same (recording_id, stem) yield exactly
    one slot and the ref_path comes from the row with the smaller track_audio_id.
    """
    conn = _make_db()
    _add_set_audio(conn, set_audio_id=51, set_id="S51", path="/mix51.m4a", is_reference=1)
    _add_slot(conn, set_id="S51", row_index=0, recording_id="R51", cue_time_seconds=30)
    # Insert two track_audio rows: id=100 (low → chosen) and id=200 (high → ignored)
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (100, "R51", "regular", "/ref_low.flac", 1),
    )
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (200, "R51", "regular", "/ref_high.flac", 0),
    )

    slots = query_corpus_slots(conn, policy_stems=("regular",))

    assert len(slots) == 1, f"Expected 1 slot, got {len(slots)}"
    assert slots[0].ref_path == "/ref_low.flac"


def test_query_cue_time_seconds_anchor():
    """Fix 2: cue_time_seconds=NULL with cue_seconds=0 is EXCLUDED (no cue).
    A slot with cue_time_seconds=90 and cue_seconds=0 IS INCLUDED, and
    cue_time_s == 90.0 (not the coalesced cue_seconds fallback).
    """
    conn = _make_db()
    _add_set_audio(conn, set_audio_id=52, set_id="S52", path="/mix52.m4a")
    # Slot with cue_time_seconds=NULL, cue_seconds=0 → should be EXCLUDED
    conn.execute(
        "INSERT INTO set_track_slots VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("S52", 0, "R52a", "001", 0, None, "original", "regular", "regular", 40),
    )
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (None, "R52a", "regular", "/ref52a.flac", 1),
    )
    # Slot with cue_time_seconds=90, cue_seconds=0 → should be INCLUDED with cue_time_s=90.0
    conn.execute(
        "INSERT INTO set_track_slots VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("S52", 1, "R52b", "002", 0, 90, "original", "regular", "regular", 40),
    )
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (None, "R52b", "regular", "/ref52b.flac", 1),
    )

    slots = query_corpus_slots(conn, policy_stems=("regular",))

    assert len(slots) == 1, (
        f"Expected 1 slot, got {len(slots)}: {[s.recording_id for s in slots]}"
    )
    assert slots[0].recording_id == "R52b"
    assert slots[0].cue_time_s == 90.0


def test_census_rows_includes_blocked_slots_via_left_join():
    """Fix 4: census_rows uses LEFT JOIN so a slot with no set_audio row still
    appears in the result (mix_full_path = NULL), and cue_time_seconds=NULL with
    cue_seconds=0 yields cue_time_s=NULL (classified no-cue-time by _classify).
    """
    conn = _make_db()
    # Slot with no set_audio row → blocked (no-mix-audio); should still appear
    _add_slot(
        conn,
        set_id="S53",
        row_index=0,
        recording_id="R53",
        claimed_stem="regular",
        cue_time_seconds=50,
    )
    _add_track_audio(conn, recording_id="R53", stem="regular", path="/ref53.flac")
    # No _add_set_audio → set_audio LEFT JOIN yields NULL columns for this slot

    rows = census_rows(conn, policy_stems=("regular",))

    assert len(rows) == 1
    assert rows[0]["mix_full_path"] is None
    assert rows[0]["claimed_stem"] == "regular"

    # Also verify: cue_time_seconds=NULL, cue_seconds=0 → cue_time_s must be NULL
    conn.execute(
        "INSERT INTO set_track_slots VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("S54", 0, "R54", "001", 0, None, "original", "regular", "regular", 40),
    )
    _add_set_audio(conn, set_audio_id=54, set_id="S54", path="/mix54.m4a")
    _add_track_audio(conn, recording_id="R54", stem="regular", path="/ref54.flac")

    rows2 = census_rows(conn, policy_stems=("regular",))
    r54 = next(r for r in rows2 if r["set_id"] == "S54")
    assert r54["cue_time_s"] is None  # NULLIF(cue_time_seconds, 0) must return NULL
