"""I/O wrappers around the pure `same_song_guard`: title probe, DB context,
the two runtime gates, and the recording/detach ledger write."""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.result import Ok
from ingest.adapters.fingerprint import fingerprint_file
from ingest.corrections import Correction, log_correction
from ingest.same_song_guard import GuardVerdict, same_song_guard


@dataclass(frozen=True)
class RecordingContext:
    title: str
    regular_path: str | None


def probe_url_title(url: str, yt_dlp: Path, *, timeout_s: float = 60.0) -> str | None:
    """Fetch the source title WITHOUT downloading (metadata-only yt-dlp call)."""
    try:
        out = subprocess.run(
            [
                str(yt_dlp),
                "--skip-download",
                "--no-playlist",
                "--print",
                "%(title)s",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    return lines[0] if lines else None


def recording_context(db_path: Path, recording_id: str) -> RecordingContext:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT COALESCE(rec.full_name, w.title, '') AS title "
            "FROM recording rec LEFT JOIN work w ON rec.work_id = w.work_id "
            "WHERE rec.recording_id = ?",
            (recording_id,),
        ).fetchone()
        ref = conn.execute(
            "SELECT path FROM track_audio WHERE recording_id = ? AND stem = 'regular' "
            "ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1",
            (recording_id,),
        ).fetchone()
    return RecordingContext(
        title=(r["title"] if r else ""),
        regular_path=(ref["path"] if ref else None),
    )


def title_gate(acquired_title: str, recording_title: str) -> GuardVerdict:
    """Pre-download decision: title channel only (no fingerprints)."""
    return same_song_guard(acquired_title, recording_title, "regular", None, None)


def content_gate(
    stem_axis: str, regular_path: str | None, candidate_path: str
) -> GuardVerdict:
    """Post-insert decision: content channel only (title left blank)."""
    if not regular_path:
        return GuardVerdict(
            True, None, "no regular reference — content channel skipped"
        )
    fa = fingerprint_file(regular_path)
    fb = fingerprint_file(candidate_path)
    if not (isinstance(fa, Ok) and isinstance(fb, Ok)):
        return GuardVerdict(
            True, None, "fingerprint unavailable — content channel skipped"
        )
    return same_song_guard("", "", stem_axis, fa.value, fb.value)


def log_detach(
    db_path: Path,
    *,
    recording_id: str,
    set_id: str | None,
    position: str | None,
    acquired_title: str,
    verdict: GuardVerdict,
) -> None:
    """Record a wrong-recording detach (abstain) in the correction ledger."""
    c = Correction(
        track_id=recording_id,  # see spec: track_id overloaded to recording_id for axis='recording'
        axis="recording",
        action="detach",
        set_id=set_id,
        position=position,
        old_recording_id=recording_id,
        new_recording_id=None,
        reason=f"[{verdict.channel}] {verdict.reason} (acquired={acquired_title!r})",
        source="same_song_guard",
    )
    log_correction(db_path, c)
