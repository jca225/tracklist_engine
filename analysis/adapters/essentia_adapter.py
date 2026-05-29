"""Subprocess adapter to the Essentia (Python 3.13) worker.

Essentia has no Py 3.14 wheels, so it lives in venvs/essentia/ and we shell
out to it. The boundary is JSON over stdout — see essentia_worker.py.

The Homebrew Python 3.13 used for the sandbox links pyexpat against a
newer libexpat than ships with macOS, so we inject DYLD_LIBRARY_PATH at
spawn time. If that path moves on a different machine, override via the
`ESSENTIA_EXPAT_LIB` env var.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.result import Err, Ok, Result
from ..errors import EssentiaError
from ..models import EssentiaFeatures
from . import essentia_models as em

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_ESSENTIA_PYTHON: Path = _REPO_ROOT / "venvs" / "essentia" / "bin" / "python"
_WORKER_MODULE: str = "analysis.adapters.essentia_worker"
_DEFAULT_EXPAT_LIB: str = "/opt/homebrew/opt/expat/lib"


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    expat = env.get("ESSENTIA_EXPAT_LIB", _DEFAULT_EXPAT_LIB)
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{expat}:{existing}" if existing else expat
    env["PYTHONPATH"] = str(_REPO_ROOT)
    return env


def _opt_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _parse(payload: dict, track_audio_id: int) -> EssentiaFeatures:
    yamnet = payload.get("yamnet")
    yamnet_raw = (
        {k: float(v) for k, v in yamnet.items()}
        if isinstance(yamnet, dict)
        else None
    )
    models_present = tuple(payload.get("models_present", ()))
    return EssentiaFeatures(
        track_audio_id=track_audio_id,
        version=str(payload["version"]),
        models_present=models_present,
        key_tonic=str(payload["key"]["tonic"]),
        key_mode=str(payload["key"]["mode"]),
        key_strength=float(payload["key"]["strength"]),
        key_profile=str(payload["key"]["profile"]),
        bpm=float(payload["rhythm"]["bpm"]),
        n_beats=int(payload["rhythm"]["n_beats"]),
        danceability_sp=float(payload["danceability_sp"]),
        mood_happy=_opt_float(payload.get("mood_happy")),
        mood_acoustic=_opt_float(payload.get("mood_acoustic")),
        mood_aggressive=_opt_float(payload.get("mood_aggressive")),
        voice_prob=_opt_float(payload.get("voice_prob")),
        danceability_tf=_opt_float(payload.get("danceability_tf")),
        valence=_opt_float(payload.get("valence")),
        arousal=_opt_float(payload.get("arousal")),
        valence_raw=_opt_float(payload.get("valence_raw")),
        arousal_raw=_opt_float(payload.get("arousal_raw")),
        speechiness=_opt_float(payload.get("speechiness")),
        liveness=_opt_float(payload.get("liveness")),
        yamnet_raw=yamnet_raw,
    )


@dataclass(frozen=True)
class EnsureModelsResult:
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]   # (name, reason) pairs


def ensure_models(timeout_s: float = 600.0) -> Result[EnsureModelsResult, EssentiaError]:
    """Trigger the worker to download any missing .pb files.

    Calling Python is the cheap way to get a one-shot download command
    that runs under the same env as the analyze() worker — keeps proxies
    / DNS / SSL config in one place.
    """
    if not _ESSENTIA_PYTHON.exists():
        return Err(EssentiaError(kind="venv_missing", detail=str(_ESSENTIA_PYTHON)))
    try:
        proc = subprocess.run(
            [str(_ESSENTIA_PYTHON), "-m", _WORKER_MODULE, "--ensure-models"],
            cwd=str(_REPO_ROOT),
            env=_worker_env(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return Err(EssentiaError(kind="timeout", detail=f"after {timeout_s}s"))

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        return Err(EssentiaError(kind="bad_json", detail=str(e)))

    if proc.returncode != 0 and not payload.get("failed"):
        return Err(EssentiaError(kind="worker_failed", detail=proc.stderr or proc.stdout))

    return Ok(EnsureModelsResult(
        downloaded=tuple(payload.get("downloaded", [])),
        skipped=tuple(payload.get("skipped", [])),
        failed=tuple(
            (str(f["name"]), str(f["reason"]))
            for f in payload.get("failed", [])
        ),
    ))


def models_present() -> set[str]:
    """Names of models currently downloaded under data/essentia_models/."""
    return em.which_present()


def is_available() -> bool:
    """True if the Py 3.13 sandbox venv is installed on this machine."""
    return _ESSENTIA_PYTHON.exists()


def analyze(
    audio_path: Path,
    track_audio_id: int,
    timeout_s: float = 180.0,
) -> Result[EssentiaFeatures, EssentiaError]:
    if not _ESSENTIA_PYTHON.exists():
        return Err(EssentiaError(kind="venv_missing", detail=str(_ESSENTIA_PYTHON)))
    if not audio_path.exists():
        return Err(EssentiaError(kind="audio_missing", detail=str(audio_path)))

    try:
        proc = subprocess.run(
            [str(_ESSENTIA_PYTHON), "-m", _WORKER_MODULE, str(audio_path)],
            cwd=str(_REPO_ROOT),
            env=_worker_env(),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return Err(EssentiaError(kind="timeout", detail=f"after {timeout_s}s"))

    if proc.returncode != 0:
        try:
            payload = json.loads(proc.stdout or "{}")
            detail = str(payload.get("error", proc.stderr or proc.stdout))
        except json.JSONDecodeError:
            detail = proc.stderr or proc.stdout
        return Err(EssentiaError(kind="worker_failed", detail=detail))

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return Err(EssentiaError(kind="bad_json", detail=str(e)))

    return Ok(_parse(payload, track_audio_id))
