"""Post-download identity verification for ingest skip/replace decisions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.result import Err, Ok


class Verdict(str, Enum):
    OK = "OK"
    MISSING_FILE = "MISSING_FILE"
    NO_REFERENCE = "NO_REFERENCE"
    FALLBACK_TO_ORIGINAL = "FALLBACK_TO_ORIGINAL"
    WRONG_SONG = "WRONG_SONG"
    DURATION_MISMATCH = "DURATION_MISMATCH"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    LIVE_SUSPECT = "LIVE_SUSPECT"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class VerifyResult:
    verdict: Verdict
    detail: str = ""
    track_audio_id: int | None = None
    path: str | None = None


def lookup_reference_row(db_path: Path, track_id: str) -> tuple[int, str, str] | None:
    """Return (track_audio_id, path, stem) for best reference row."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT track_audio_id, path, stem FROM track_audio
            WHERE recording_id = ? OR track_id = ?
            ORDER BY is_reference DESC, downloaded_at DESC
            LIMIT 1
            """,
            (track_id, track_id),
        ).fetchone()
    if not row:
        return None
    return int(row[0]), str(row[1]), str(row[2])


def verify_reference_exists(db_path: Path, track_id: str) -> VerifyResult:
    """Check reference row exists and file is on disk."""
    row = lookup_reference_row(db_path, track_id)
    if row is None:
        return VerifyResult(Verdict.NO_REFERENCE, "no track_audio row")
    taid, path, stem = row
    if not path or not Path(path).is_file():
        return VerifyResult(
            Verdict.MISSING_FILE,
            f"reference path missing: {path}",
            track_audio_id=taid,
            path=path,
        )
    return VerifyResult(
        Verdict.OK,
        "reference file present",
        track_audio_id=taid,
        path=path,
    )


def verify_variant_against_original(
    original_path: str,
    variant_path: str,
    stem_role: str,
) -> VerifyResult:
    """Chromaprint compare variant vs regular reference (advisory thresholds)."""
    from ingest.adapters import fingerprint as fp

    fa = fp.fingerprint_file(original_path)
    fb = fp.fingerprint_file(variant_path)
    match (fa, fb):
        case (Ok(a), Ok(b)):
            sim = fp.similarity(a.raw, b.raw)
            dur_ratio = (b.duration_s / a.duration_s) if a.duration_s else 0.0
            code, detail = fp.classify(stem_role, sim, dur_ratio)
            try:
                verdict = Verdict(code)
            except ValueError:
                verdict = Verdict.WEAK_SIGNAL
            return VerifyResult(verdict, detail, path=variant_path)
        case (Err(e), _) | (_, Err(e)):
            return VerifyResult(
                Verdict.SKIPPED,
                f"fingerprint failed: {e.kind} — {e.detail}",
                path=variant_path,
            )
    return VerifyResult(Verdict.SKIPPED, "unreachable")


def should_skip_existing(
    db_path: Path,
    track_id: str,
    *,
    reverify: bool = True,
) -> tuple[bool, VerifyResult]:
    """Whether ingest should skip download for this track_id.

    Default (``reverify=True``): skip only when the best reference row exists
    on disk. Legacy sticky skip (any ``track_audio`` row) remains available via
    ``reverify=False`` for one-off backfills.
    """
    if not reverify:
        with sqlite3.connect(db_path) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM track_audio WHERE recording_id = ? OR track_id = ?",
                (track_id, track_id),
            ).fetchone()[0]
        if n:
            return True, VerifyResult(Verdict.OK, "has_any_audio (legacy skip)")
        return False, VerifyResult(Verdict.NO_REFERENCE, "no rows")

    result = verify_reference_exists(db_path, track_id)
    if result.verdict == Verdict.OK:
        return True, result
    return False, result
