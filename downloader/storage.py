from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from downloader.commands import run_command
from downloader.constants import (
    AUDIO_EXTENSIONS,
    HARD_DRIVE_MOUNT,
    HARD_DRIVE_OUTPUT_ROOT,
    OBJECTS_DIRNAME,
    SOURCE_AUDIO_FILENAME,
    SOURCE_AUDIO_META_FILENAME,
    SOURCE_DIRNAME,
    STEMS_DIRNAME,
    TARGET_AUDIO_FORMAT,
    TARGET_BIT_DEPTH,
    TARGET_CHANNELS,
    TARGET_PCM_CODEC,
    TARGET_SAMPLE_RATE,
    TRACKS_DIRNAME,
)


def resolve_output_root(output_root: Path | None) -> Path:
    """
    Resolve where downloads should be written.
    Defaults to HARD_DRIVE_OUTPUT_ROOT when --output-root is not provided.
    """
    if output_root is None:
        if not HARD_DRIVE_MOUNT.exists():
            raise FileNotFoundError(
                f"Hard drive not found at {HARD_DRIVE_MOUNT}. "
                "Update HARD_DRIVE_MOUNT in downloader/constants.py "
                "or pass --output-root explicitly."
            )
        resolved = HARD_DRIVE_OUTPUT_ROOT
    else:
        resolved = output_root

    resolved = resolved.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_track_object_id(row: dict[str, str], canonical_url: str) -> str:
    seed = "|".join(
        [
            canonical_url,
            row.get("set_id", ""),
            row.get("track_id", ""),
            row.get("platform", ""),
            row.get("source", ""),
        ]
    )
    return f"trk_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:8]}"


def get_track_paths(output_root: Path, track_object_id: str) -> dict[str, Path]:
    track_dir = output_root / OBJECTS_DIRNAME / TRACKS_DIRNAME / track_object_id
    source_dir = track_dir / SOURCE_DIRNAME
    source_audio = source_dir / SOURCE_AUDIO_FILENAME
    source_meta = source_dir / SOURCE_AUDIO_META_FILENAME
    stems_root = track_dir / STEMS_DIRNAME
    source_dir.mkdir(parents=True, exist_ok=True)
    stems_root.mkdir(parents=True, exist_ok=True)
    return {
        "track_dir": track_dir,
        "source_dir": source_dir,
        "source_audio": source_audio,
        "source_meta": source_meta,
        "stems_root": stems_root,
    }


def transcode_to_pipeline_wav(input_audio: Path, output_wav: Path) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-vn",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        TARGET_PCM_CODEC,
        str(output_wav),
    ]
    ok, err = run_command(cmd)
    if not ok:
        raise RuntimeError(f"ffmpeg transcode failed: {err}")


def build_source_audio_metadata(audio_file: Path) -> dict[str, Any]:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_file), sr=None, mono=False)
    if isinstance(y, np.ndarray) and y.ndim == 1:
        channels = 1
        samples = int(y.shape[0])
    elif isinstance(y, np.ndarray) and y.ndim == 2:
        channels = int(y.shape[0])
        samples = int(y.shape[1])
    else:
        channels = TARGET_CHANNELS
        samples = 0

    duration = float(samples / sr) if sr and samples else 0.0
    return {
        "format": TARGET_AUDIO_FORMAT,
        "sample_rate_hz": int(sr),
        "channels": channels,
        "bit_depth": TARGET_BIT_DEPTH,
        "codec": TARGET_PCM_CODEC,
        "duration_sec": duration,
        "size_bytes": audio_file.stat().st_size,
        "sha256": sha256_file(audio_file),
    }


def write_source_audio_metadata(meta_path: Path, metadata: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")


def get_track_dir_for_source_audio(audio_file: Path) -> Path:
    if audio_file.name == SOURCE_AUDIO_FILENAME and audio_file.parent.name == SOURCE_DIRNAME:
        return audio_file.parent.parent
    return audio_file.parent


def list_audio_files_with_mtime(root: Path) -> dict[Path, float]:
    if not root.exists():
        return {}
    files: dict[Path, float] = {}
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        try:
            files[candidate] = candidate.stat().st_mtime
        except OSError:
            continue
    return files


def detect_downloaded_file(output_dir: Path, before_state: dict[Path, float], started_at: float) -> Path | None:
    after_state = list_audio_files_with_mtime(output_dir)
    changed = [
        path
        for path, mtime in after_state.items()
        if path not in before_state or mtime > before_state[path] or mtime >= started_at - 1.0
    ]
    if changed:
        return max(changed, key=lambda p: after_state.get(p, 0.0))
    if after_state:
        return max(after_state, key=lambda p: after_state.get(p, 0.0))
    return None
