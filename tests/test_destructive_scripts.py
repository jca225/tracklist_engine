"""Dry-run and apply tests for canonical-state mutators.

Uses real schema.sql fixtures on tmp paths — no pi-storage, no network.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core import db as db_adapter
from core.models import AudioAsset
from core.result import Ok
from scripts import reconcile_orphans as ro
from scripts import replace_track_audio as rta


_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"


@pytest.fixture
def canonical_env(tmp_path: Path) -> tuple[Path, Path]:
    """Fresh DB + audio_root under tmp_path."""
    db = tmp_path / "test.db"
    audio_root = tmp_path / "storage"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    conn.close()
    return db, audio_root


def _write_object(
    audio_root: Path,
    track_id: str,
    name: str,
    content: bytes = b"audio-payload",
) -> Path:
    d = audio_root / "objects" / track_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(content)
    return p


def _insert_track(
    db: Path,
    audio_root: Path,
    track_id: str,
    *,
    platform: str = "youtube",
    player_id: str = "vid1",
    content: bytes = b"registered-audio",
) -> tuple[int, Path]:
    path = _write_object(
        audio_root, track_id, f"{track_id}__{platform}__{player_id}.m4a", content,
    )
    asset = AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform=platform,
        source_url=f"https://example.com/{player_id}",
        player_id=player_id,
        path=str(path),
        sha256="abc",
        duration_s=200.0,
        sample_rate=44100,
        codec="m4a",
        bitrate_kbps=128,
    )
    r = db_adapter.insert_audio(db, asset)
    assert isinstance(r, Ok)
    return r.value, path


def _count_track_audio(db: Path) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM track_audio").fetchone()[0]


def _count_corrections(db: Path) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM track_audio_correction").fetchone()[0]


def _paths_on_disk(audio_root: Path) -> set[str]:
    root = audio_root / "objects"
    if not root.is_dir():
        return set()
    return {str(p) for p in root.rglob("*") if p.is_file()}


# ── reconcile_orphans: classification ─────────────────────────────────────


def test_classify_pure_orphan_is_register(canonical_env: tuple[Path, Path]) -> None:
    db, audio_root = canonical_env
    orphan = _write_object(audio_root, "PURE01", "PURE01__youtube__newvid.m4a")

    orphans, _, _ = ro.classify(db, audio_root / "objects")
    hit = [o for o in orphans if o.path == orphan]
    assert len(hit) == 1
    assert hit[0].disposition == "REGISTER"


def test_classify_intermediate_is_delete(canonical_env: tuple[Path, Path]) -> None:
    db, audio_root = canonical_env
    junk = _write_object(audio_root, "T1", "partial.webm", b"webm-chunk")

    orphans, _, _ = ro.classify(db, audio_root / "objects")
    hit = [o for o in orphans if o.path == junk]
    assert len(hit) == 1
    assert hit[0].disposition == "DELETE"


def test_classify_coexist_extra_final_is_review(canonical_env: tuple[Path, Path]) -> None:
    db, audio_root = canonical_env
    _insert_track(db, audio_root, "COEX01", player_id="refvid")
    extra = _write_object(
        audio_root, "COEX01", "COEX01__youtube__other.m4a", b"different-bytes",
    )

    orphans, _, _ = ro.classify(db, audio_root / "objects")
    hit = [o for o in orphans if o.path == extra]
    assert len(hit) == 1
    assert hit[0].disposition == "REVIEW"


# ── reconcile_orphans: dry-run vs apply ─────────────────────────────────────


def test_reconcile_dry_run_leaves_db_and_disk_unchanged(
    canonical_env: tuple[Path, Path], tmp_path: Path,
) -> None:
    db, audio_root = canonical_env
    _insert_track(db, audio_root, "REG01")
    pure = _write_object(audio_root, "PURE02", "PURE02__youtube__v2.m4a")
    junk = _write_object(audio_root, "REG01", "stale.part", b"partials")

    before_rows = _count_track_audio(db)
    before_files = _paths_on_disk(audio_root)
    review_tsv = tmp_path / "review.tsv"

    rc = ro.main([
        "--db", str(db),
        "--audio-root", str(audio_root),
        "--review-tsv", str(review_tsv),
    ])
    assert rc == 0
    assert _count_track_audio(db) == before_rows
    assert _paths_on_disk(audio_root) == before_files
    assert pure.exists()
    assert junk.exists()


def test_reconcile_apply_registers_pure_orphan(canonical_env: tuple[Path, Path]) -> None:
    db, audio_root = canonical_env
    orphan_path = _write_object(
        audio_root, "PURE03", "PURE03__youtube__regme.m4a", b"only-copy",
    )
    assert _count_track_audio(db) == 0

    rc = ro.main(["--db", str(db), "--audio-root", str(audio_root), "--apply"])
    assert rc == 0
    assert orphan_path.exists()
    assert _count_track_audio(db) == 1

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT path, platform, player_id FROM track_audio WHERE track_id='PURE03'",
        ).fetchone()
    assert row is not None
    assert row[0] == str(orphan_path)
    assert row[1] == "youtube"
    assert row[2] == "regme"
    assert _count_corrections(db) == 1


def test_reconcile_apply_deletes_intermediate_only(
    canonical_env: tuple[Path, Path],
) -> None:
    db, audio_root = canonical_env
    _, reg_path = _insert_track(db, audio_root, "REG02")
    junk = _write_object(audio_root, "REG02", "download.webm", b"webm")

    rc = ro.main(["--db", str(db), "--audio-root", str(audio_root), "--apply"])
    assert rc == 0
    assert not junk.exists()
    assert reg_path.exists()
    assert _count_track_audio(db) == 1


def test_reconcile_apply_does_not_promote_without_flag(
    canonical_env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PROMOTE dispositions are skipped unless --apply-promotions is passed."""
    db, audio_root = canonical_env
    _insert_track(db, audio_root, "PROM01", player_id="short", content=b"short-ref")
    long_orphan = _write_object(
        audio_root, "PROM01", "PROM01__youtube__long.m4a", b"long-orphan-audio",
    )

    # Force PROMOTE classification without relying on ffprobe/acappella heuristics.
    fake = ro.Orphan(
        path=long_orphan, track_id="PROM01", ext=".m4a",
        platform="youtube", player_id="long",
        disposition="PROMOTE", reason="test",
    )
    monkeypatch.setattr(ro, "classify", lambda _db, _root: ([fake], set(), {}))

    rc = ro.main(["--db", str(db), "--audio-root", str(audio_root), "--apply"])
    assert rc == 0
    assert _count_track_audio(db) == 1  # still only the seeded row
    assert long_orphan.exists()


# ── replace_track_audio: file mode (no network) ─────────────────────────────


def test_replace_via_file_replaces_row_and_cascades_analysis(
    canonical_env: tuple[Path, Path], tmp_path: Path,
) -> None:
    db, audio_root = canonical_env
    old_taid, old_path = _insert_track(db, audio_root, "REP01", player_id="old")

    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO track_analysis (track_audio_id, beat_times_json) VALUES (?, '[]')",
            (old_taid,),
        )
        conn.commit()

    new_src = tmp_path / "replacement.m4a"
    new_src.write_bytes(b"new-studio-master")

    rc = rta._replace_via_file(
        db, audio_root, "REP01", new_src, "manual_v2", old_taid,
        promote_reference=True, purge_siblings=False,
    )
    assert rc == 0
    assert not old_path.exists()

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT track_audio_id, path, platform, player_id, is_reference "
            "FROM track_audio WHERE track_id='REP01'",
        ).fetchall()
        analysis = conn.execute("SELECT COUNT(*) FROM track_analysis").fetchone()[0]

    assert len(rows) == 1
    new_taid, new_path, platform, player_id, is_ref = rows[0]
    assert new_taid != old_taid
    assert platform == "manual"
    assert player_id == "manual_v2"
    assert is_ref == 1
    assert Path(new_path).is_file()
    assert analysis == 0


def test_replace_via_file_respects_no_promote(
    canonical_env: tuple[Path, Path], tmp_path: Path,
) -> None:
    db, audio_root = canonical_env
    old_taid, _ = _insert_track(db, audio_root, "REP02", player_id="old")

    new_src = tmp_path / "replacement.m4a"
    new_src.write_bytes(b"alternate")

    rc = rta._replace_via_file(
        db, audio_root, "REP02", new_src, "manual_v3", old_taid,
        promote_reference=False, purge_siblings=False,
    )
    assert rc == 0

    with sqlite3.connect(db) as conn:
        is_ref = conn.execute(
            "SELECT is_reference FROM track_audio WHERE track_id='REP02'",
        ).fetchone()[0]
    assert is_ref == 0


def test_delete_old_row_if_exists_is_noop_when_missing(
    canonical_env: tuple[Path, Path],
) -> None:
    db, audio_root = canonical_env
    rta._delete_old_row_if_exists(db, audio_root, 99999)
    assert _count_track_audio(db) == 0
