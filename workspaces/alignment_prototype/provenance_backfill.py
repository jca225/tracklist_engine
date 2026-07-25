"""Brick 8 — canonical backfill runner: a real slice of the corpus walks onto
the content-addressed, versioned, provenanced feature-store model.

First real step of the ratified forward-first cutover
(docs/engine/feature_store_cutover.md, Decision 2 step 1): take N real
recordings of one set that HAVE cached MERT and resolvable audio, pull each
audio READ-ONLY from pi-storage, and build ONE central provenance store
(a single provenance.db + objects/ — never per-set) holding, per recording:

- the ``audio`` artifact (content-addressed copy of the pi rip),
- a fresh ``landmark_fingerprint`` computed through ``Registry.run``,
- the cached MERT stack migrated to a ``mert_features`` artifact,

each feature carrying a derivation edge back to its audio artifact and a
deterministic ProcessSpec. Optionally (grounding the Brick 8 producer types)
it also runs ``analyze.hubert`` on the first pulled track's FULL audio and
``analyze.whisper`` on a SHORT slice of it — both through the registry, both
the REAL models.

Idempotent by construction: content addressing dedupes artifacts, deterministic
ids dedupe ParameterSet/EnvironmentSpec/ProcessSpec; re-running adds run rows
(append-only history), never duplicate artifacts or specs.

Safety contract (docs/engine/feature_store_cutover.md): pi-storage is
READ-ONLY here — one ``sqlite3 -readonly`` SELECT + one scp per recording;
nothing on the pi is written. This runner builds the store LOCALLY; the
additive deposit to ``/mnt/storage/provenance`` is a separate, parent-owned
step.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.provenance_backfill \
        --set-id 1fsnxchk --limit 5 \
        [--db-root workspaces/alignment_prototype/out/provenance/central] \
        [--skip-hubert] [--skip-whisper] [--whisper-slice-s 60]
    venvs/audio/bin/python -m core.provenance.laws --db <db-root>
"""

from __future__ import annotations

import io
import json
import subprocess
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.provenance import (
    REGISTRY,
    Artifact,
    ArtifactKind,
    ArtifactStore,
    ProvenanceRepository,
    connect,
)
from workspaces.alignment_prototype import producers
from workspaces.alignment_prototype.producers import (
    _MEDIA_TYPES,
    _print_lineage,
    migrate_cached_mert,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_CENTRAL_ROOT = Path(__file__).resolve().parent / "out" / "provenance" / "central"
DEFAULT_SET_ID = "1fsnxchk"  # BB12
DEFAULT_LIMIT = 5
DEFAULT_WHISPER_SLICE_S = 60.0

#: Pull signature: (recording_id, dest_dir) -> local audio path or None.
PullFn = Callable[[str, Path], Optional[Path]]


@dataclass
class TrackResult:
    """Per-recording shas after backfill (None = that feature was skipped)."""

    recording_id: str
    audio: Artifact
    fingerprint: Artifact
    mert: Optional[Artifact]


@dataclass
class BackfillSummary:
    set_id: str
    db_root: Path
    tracks: list[TrackResult] = field(default_factory=list)
    hubert: Optional[Artifact] = None
    whisper: Optional[Artifact] = None
    whisper_slice: Optional[Artifact] = None
    skipped: list[str] = field(default_factory=list)  # recording_ids w/o audio


def cached_recording_ids(set_id: str) -> list[str]:
    """recording_ids present in the local mert_store cache for this set
    (sorted — deterministic slice selection)."""
    cache = producers._cache_path(set_id)
    if not cache.is_file():
        return []
    with np.load(cache, allow_pickle=False) as z:
        return sorted(json.loads(str(z["ref_ids"])))


def _register_audio(
    audio_path: Path,
    recording_id: str,
    set_id: str,
    *,
    store: ArtifactStore,
) -> Artifact:
    return store.put_file(
        audio_path,
        kind=ArtifactKind.AUDIO,
        media_type=_MEDIA_TYPES.get(
            audio_path.suffix.lower(), "application/octet-stream"
        ),
        metadata={"recording_id": recording_id, "set_id": set_id},
        source_uri=(
            f"pi-storage:/mnt/storage/objects/{recording_id}/{audio_path.name}"
        ),
    )


def backfill_recording(
    recording_id: str,
    set_id: str,
    audio_path: Path,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str,
) -> TrackResult:
    """One recording onto the canonical model: audio artifact + fresh
    fingerprint via ``Registry.run`` + migrated cached MERT."""
    audio = _register_audio(audio_path, recording_id, set_id, store=store)
    fp = REGISTRY.run(
        "analyze.landmark_fp",
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit=code_commit,
    )["features"]
    mert = migrate_cached_mert(
        recording_id, set_id, audio, store=store, repo=repo, code_commit=code_commit
    )
    return TrackResult(recording_id, audio, fp, mert)


def _slice_wav_bytes(audio_path: Path, dur_s: float) -> bytes:
    """First ``dur_s`` seconds of an audio file as 16-bit PCM WAV bytes."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=dur_s)
    buf = io.BytesIO()
    sf.write(buf, y, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run_whisper_on_slice(
    track: TrackResult,
    audio_path: Path,
    slice_s: float,
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str,
) -> tuple[Artifact, Artifact]:
    """(slice audio artifact, transcript artifact): register a SHORT real
    slice of the pulled audio, then run the REAL Whisper producer on it."""
    wav = _slice_wav_bytes(audio_path, slice_s)
    slice_art = store.put_bytes(
        wav,
        kind=ArtifactKind.AUDIO,
        media_type="audio/wav",
        metadata={
            "recording_id": track.recording_id,
            "sliced_from": track.audio.content_sha256,
            "start_s": 0.0,
            "dur_s": slice_s,
        },
        source_uri=f"slice[0:{slice_s:g}s] of sha256:{track.audio.content_sha256}",
    )
    transcript = REGISTRY.run(
        "analyze.whisper",
        {"audio": slice_art},
        store=store,
        repo=repo,
        code_commit=code_commit,
    )["transcript"]
    return slice_art, transcript


def _totals(conn: sqlite3.Connection) -> dict[str, object]:
    by_kind = {
        r["kind"]: r["n"]
        for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM artifact GROUP BY kind ORDER BY kind"
        )
    }
    return {
        "artifacts": sum(by_kind.values()),
        "by_kind": by_kind,
        "runs": conn.execute("SELECT COUNT(*) FROM run").fetchone()[0],
        "process_specs": conn.execute("SELECT COUNT(*) FROM process_spec").fetchone()[
            0
        ],
        "derivations": conn.execute("SELECT COUNT(*) FROM derivation").fetchone()[0],
    }


def run_backfill(
    set_id: str,
    limit: int,
    db_root: Path,
    *,
    pull: PullFn | None = None,
    code_commit: str | None = None,
    run_hubert: bool = True,
    run_whisper: bool = True,
    whisper_slice_s: float = DEFAULT_WHISPER_SLICE_S,
) -> BackfillSummary:
    """Backfill up to ``limit`` recordings of ``set_id`` into the central
    store at ``db_root``. Returns the summary (also printed)."""
    if pull is None:
        pull = producers._pull_audio_from_pi
    db_root.mkdir(parents=True, exist_ok=True)
    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    repo = ProvenanceRepository(conn)
    if code_commit is None:
        try:
            code_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO
                )
                .decode()
                .strip()
            )
        except Exception:
            code_commit = "unknown"

    summary = BackfillSummary(set_id=set_id, db_root=db_root)
    candidates = cached_recording_ids(set_id)
    if not candidates:
        print(f"no local MERT cache for set {set_id} — nothing to backfill")
        return summary
    print(
        f"backfill — set {set_id}: {len(candidates)} recording(s) in MERT cache, "
        f"taking up to {limit}"
    )

    audio_paths: dict[str, Path] = {}
    for recording_id in candidates:
        if len(summary.tracks) >= limit:
            break
        print(f"[{len(summary.tracks) + 1}/{limit}] {recording_id}")
        audio_path = pull(recording_id, db_root / ".pulled")
        if audio_path is None:
            summary.skipped.append(recording_id)
            print("  no resolvable audio — skipped")
            continue
        track = backfill_recording(
            recording_id,
            set_id,
            audio_path,
            store=store,
            repo=repo,
            code_commit=code_commit,
        )
        audio_paths[recording_id] = audio_path
        summary.tracks.append(track)
        print(f"  audio                : {track.audio.content_sha256}")
        print(f"  landmark_fingerprint : {track.fingerprint.content_sha256}")
        _print_lineage(track.fingerprint, repo)
        if track.mert is not None:
            print(f"  mert_features        : {track.mert.content_sha256}")
            _print_lineage(track.mert, repo)

    # --- Brick 8 producer grounding on the first pulled track ---------------
    first = summary.tracks[0] if summary.tracks else None
    if first is not None and run_hubert:
        print(f"hubert — full audio of {first.recording_id} (real model)")
        summary.hubert = REGISTRY.run(
            "analyze.hubert",
            {"audio": first.audio},
            store=store,
            repo=repo,
            code_commit=code_commit,
        )["features"]
        print(f"  hubert_features      : {summary.hubert.content_sha256}")
        _print_lineage(summary.hubert, repo)
    if first is not None and run_whisper:
        print(
            f"whisper — first {whisper_slice_s:g}s of {first.recording_id} "
            "(real model, SHORT slice)"
        )
        summary.whisper_slice, summary.whisper = run_whisper_on_slice(
            first,
            audio_paths[first.recording_id],
            whisper_slice_s,
            store=store,
            repo=repo,
            code_commit=code_commit,
        )
        print(f"  audio slice          : {summary.whisper_slice.content_sha256}")
        print(f"  whisper_transcript   : {summary.whisper.content_sha256}")
        _print_lineage(summary.whisper, repo)

    totals = _totals(conn)
    print(
        f"totals — artifacts {totals['artifacts']} "
        f"({', '.join(f'{k}={v}' for k, v in totals['by_kind'].items())}), "
        f"runs {totals['runs']}, process_specs {totals['process_specs']}, "
        f"derivation edges {totals['derivations']}"
    )
    print(f"  central store        : {db_root}")
    print(f"  check laws           : python -m core.provenance.laws --db {db_root}")
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", default=DEFAULT_SET_ID)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument(
        "--db-root",
        default=str(_CENTRAL_ROOT),
        help=f"central provenance root (default {_CENTRAL_ROOT})",
    )
    ap.add_argument(
        "--skip-hubert",
        action="store_true",
        help="skip the analyze.hubert grounding run on the first track",
    )
    ap.add_argument(
        "--skip-whisper",
        action="store_true",
        help="skip the analyze.whisper grounding run on the first track's slice",
    )
    ap.add_argument(
        "--whisper-slice-s",
        type=float,
        default=DEFAULT_WHISPER_SLICE_S,
        help="length of the real audio slice fed to Whisper",
    )
    args = ap.parse_args()

    summary = run_backfill(
        args.set_id,
        args.limit,
        Path(args.db_root),
        run_hubert=not args.skip_hubert,
        run_whisper=not args.skip_whisper,
        whisper_slice_s=args.whisper_slice_s,
    )
    return 0 if summary.tracks else 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
