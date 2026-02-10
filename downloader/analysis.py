from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from downloader.commands import run_command
from downloader.constants import (
    CAMELOT_MAJOR,
    CAMELOT_MINOR,
    KEY_NAMES_SHARP,
    KRUMHANSL_MAJOR,
    KRUMHANSL_MINOR,
    STEMS_DIRNAME,
    TARGET_AUDIO_FORMAT,
    TARGET_BIT_DEPTH,
    TARGET_CHANNELS,
    TARGET_PCM_CODEC,
    TARGET_SAMPLE_RATE,
)
from downloader.cue_detr import get_cue_points_with_cue_detr
from downloader.storage import get_track_dir_for_source_audio, sha256_file, transcode_to_pipeline_wav


def _best_key_from_chroma(chroma_vector: list[float]) -> dict[str, Any]:
    import numpy as np

    chroma = np.asarray(chroma_vector, dtype=float)
    if chroma.size != 12:
        raise ValueError("Expected 12 chroma bins.")

    chroma = chroma / (np.linalg.norm(chroma) + 1e-12)
    major_profile = np.asarray(KRUMHANSL_MAJOR, dtype=float)
    minor_profile = np.asarray(KRUMHANSL_MINOR, dtype=float)
    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)

    major_scores = [float(np.dot(chroma, np.roll(major_profile, i))) for i in range(12)]
    minor_scores = [float(np.dot(chroma, np.roll(minor_profile, i))) for i in range(12)]

    major_idx = int(max(range(12), key=lambda i: major_scores[i]))
    minor_idx = int(max(range(12), key=lambda i: minor_scores[i]))

    if major_scores[major_idx] >= minor_scores[minor_idx]:
        tonic = KEY_NAMES_SHARP[major_idx]
        return {
            "key": f"{tonic} major",
            "tonic": tonic,
            "mode": "major",
            "camelot": CAMELOT_MAJOR.get(tonic),
            "confidence": major_scores[major_idx],
        }

    tonic = KEY_NAMES_SHARP[minor_idx]
    return {
        "key": f"{tonic} minor",
        "tonic": tonic,
        "mode": "minor",
        "camelot": CAMELOT_MINOR.get(tonic),
        "confidence": minor_scores[minor_idx],
    }


def get_key_and_bpm(audio_file: Path) -> dict[str, Any]:
    """Estimate tempo (BPM) and musical key from an audio file."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_file), sr=TARGET_SAMPLE_RATE, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = float(np.atleast_1d(tempo)[0])

    y_harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()
    key_info = _best_key_from_chroma(chroma_mean)

    return {
        "bpm": tempo_value,
        "duration_sec": float(librosa.get_duration(y=y, sr=sr)),
        "sample_rate": int(sr),
        **key_info,
    }


def save_analysis_sidecar(audio_file: Path, analysis: dict[str, Any]) -> Path:
    sidecar = audio_file.parent / "analysis.json"
    sidecar.write_text(json.dumps(analysis, indent=2, ensure_ascii=True), encoding="utf-8")
    return sidecar


def create_stems_with_demucs(
    audio_file: Path,
    *,
    model: str = "htdemucs",
    device: str = "mps",
    segment: int = 7,
    overlap: float = 0.25,
) -> dict[str, Any]:
    """Create acapella (vocals) and instrumental stems using Demucs."""
    track_dir = get_track_dir_for_source_audio(audio_file)
    demucs_params = {
        "model": model,
        "device": device,
        "segment": segment,
        "overlap": overlap,
        "target_format": TARGET_AUDIO_FORMAT,
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "target_channels": TARGET_CHANNELS,
        "target_bit_depth": TARGET_BIT_DEPTH,
    }
    run_hash = hashlib.sha1(json.dumps(demucs_params, sort_keys=True).encode("utf-8")).hexdigest()[:4]
    stems_dir = track_dir / STEMS_DIRNAME / f"sep_demucs_v4__sha_{run_hash}"
    vocals_out = stems_dir / "vocals.wav"
    instrumental_out = stems_dir / "instrumental.wav"
    drums_out = stems_dir / "drums.wav"
    bass_out = stems_dir / "bass.wav"
    other_out = stems_dir / "other.wav"
    manifest_out = stems_dir / "manifest.json"

    if all(p.exists() for p in [vocals_out, instrumental_out, drums_out, bass_out, other_out, manifest_out]):
        return {
            "acapella_path": str(vocals_out),
            "instrumental_path": str(instrumental_out),
            "drums_path": str(drums_out),
            "bass_path": str(bass_out),
            "other_path": str(other_out),
            "model": model,
            "device": device,
            "run_id": stems_dir.name,
            "skipped_existing": True,
        }

    with tempfile.TemporaryDirectory(prefix="demucs_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        cmd = [
            "demucs",
            "-n",
            model,
            "--device",
            device,
            "--segment",
            str(segment),
            "--overlap",
            str(overlap),
            "-o",
            str(tmpdir),
            str(audio_file),
        ]
        ok, err = run_command(cmd)
        if not ok:
            raise RuntimeError(f"demucs failed: {err}")

        model_root = tmpdir / model
        stem_sources: dict[str, Path] = {}
        for stem_name in ["vocals", "drums", "bass", "other"]:
            candidates = list(model_root.rglob(f"{stem_name}.wav"))
            if not candidates:
                raise RuntimeError(f"demucs output missing {stem_name}.wav")
            stem_sources[stem_name] = max(candidates, key=lambda p: p.stat().st_mtime)

        stems_dir.mkdir(parents=True, exist_ok=True)
        transcode_to_pipeline_wav(stem_sources["vocals"], vocals_out)
        transcode_to_pipeline_wav(stem_sources["drums"], drums_out)
        transcode_to_pipeline_wav(stem_sources["bass"], bass_out)
        transcode_to_pipeline_wav(stem_sources["other"], other_out)

    mix_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(drums_out),
        "-i",
        str(bass_out),
        "-i",
        str(other_out),
        "-filter_complex",
        "amix=inputs=3:normalize=0",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        TARGET_PCM_CODEC,
        str(instrumental_out),
    ]
    ok, err = run_command(mix_cmd)
    if not ok:
        raise RuntimeError(f"Failed to build instrumental stem: {err}")

    manifest = {
        "pipeline": "demucs",
        "run_id": stems_dir.name,
        "params": demucs_params,
        "commit": None,
        "artifacts": {
            "vocals.wav": {"sha256": sha256_file(vocals_out), "size_bytes": vocals_out.stat().st_size},
            "instrumental.wav": {
                "sha256": sha256_file(instrumental_out),
                "size_bytes": instrumental_out.stat().st_size,
            },
            "drums.wav": {"sha256": sha256_file(drums_out), "size_bytes": drums_out.stat().st_size},
            "bass.wav": {"sha256": sha256_file(bass_out), "size_bytes": bass_out.stat().st_size},
            "other.wav": {"sha256": sha256_file(other_out), "size_bytes": other_out.stat().st_size},
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "acapella_path": str(vocals_out),
        "instrumental_path": str(instrumental_out),
        "drums_path": str(drums_out),
        "bass_path": str(bass_out),
        "other_path": str(other_out),
        "manifest_path": str(manifest_out),
        "model": model,
        "device": device,
        "run_id": stems_dir.name,
        "segment": segment,
        "overlap": overlap,
        "skipped_existing": False,
    }


def analyze_downloaded_audio(
    audio_file: Path,
    *,
    cue_detr_dir: Path,
    cue_detr_checkpoint: str,
    cue_detr_sensitivity: float,
    cue_detr_radius: int,
    run_demucs: bool,
    demucs_model: str,
    demucs_device: str,
    demucs_segment: int,
    demucs_overlap: float,
) -> tuple[bool, str, dict[str, Any]]:
    analysis: dict[str, Any] = {
        "audio_file": str(audio_file),
        "created_at_unix": time.time(),
    }
    summary: dict[str, Any] = {
        "stems_generated": False,
        "stems_skipped": False,
        "stems_error": None,
    }

    try:
        analysis["key_bpm"] = get_key_and_bpm(audio_file)
    except Exception as exc:
        analysis["key_bpm_error"] = str(exc)

    try:
        analysis["cue_points_sec"] = get_cue_points_with_cue_detr(
            audio_file,
            cue_detr_dir=cue_detr_dir,
            checkpoint=cue_detr_checkpoint,
            sensitivity=cue_detr_sensitivity,
            radius=cue_detr_radius,
        )
    except Exception as exc:
        analysis["cue_points_error"] = str(exc)

    if run_demucs:
        try:
            stems = create_stems_with_demucs(
                audio_file,
                model=demucs_model,
                device=demucs_device,
                segment=demucs_segment,
                overlap=demucs_overlap,
            )
            analysis["stems"] = stems
            if stems.get("skipped_existing"):
                summary["stems_skipped"] = True
            else:
                summary["stems_generated"] = True
        except Exception as exc:
            summary["stems_error"] = str(exc)
            analysis["stems_error"] = str(exc)

    save_analysis_sidecar(audio_file, analysis)
    ok = "key_bpm" in analysis or "cue_points_sec" in analysis or "stems" in analysis
    if ok:
        return True, "", summary
    return False, "analysis failed for key/bpm, cue points, and stems", summary
