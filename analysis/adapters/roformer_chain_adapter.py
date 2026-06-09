"""MSST RoFormer chain — vocals + instrumental ensembles via MSSeparator.

Drop-in alternative to demucs_adapter / uvr_chain_adapter. Requires
workspaces/msst_webui + venvs/msst (see scripts/setup_roformer_separation.sh).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from core.result import Err, Ok, Result
from ..errors import StemError
from ..models import StemAsset, StemSet
from ..roformer_config import ModelSpec, RoformerChainConfig

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoformerChainHandle:
    config: RoformerChainConfig
    device: str
    version: str
    _msst_root: Path


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class _MsstCwd:
    """MSST resolves paths relative to its repo root — chdir for imports/inference."""

    def __init__(self, msst_root: Path) -> None:
        self._msst = msst_root.resolve()
        self._prev: str | None = None

    def __enter__(self) -> None:
        self._prev = os.getcwd()
        os.chdir(self._msst)
        root = str(self._msst)
        if root not in sys.path:
            sys.path.insert(0, root)
        for sub in ("configs", "data"):
            src = self._msst / f"{sub}_backup"
            dst = self._msst / sub
            if not dst.exists() and src.exists():
                shutil.copytree(src, dst)

    def __exit__(self, *_) -> None:
        if self._prev is not None:
            os.chdir(self._prev)


def _ensure_msst_on_path(msst_root: Path) -> _MsstCwd:
    return _MsstCwd(msst_root)


def load(
    config: RoformerChainConfig | None = None, device: str = "auto"
) -> Result[RoformerChainHandle, StemError]:
    cfg = config or RoformerChainConfig.default()
    dev = _resolve_device(device if device != "auto" else cfg.device)
    msst = cfg.msst_root
    if not msst.is_dir():
        return Err(StemError(
            kind="model_load",
            detail=f"msst_root not found: {msst} — run scripts/setup_roformer_separation.sh",
        ))
    try:
        with _ensure_msst_on_path(msst):
            from inference.msst_infer import MSSeparator  # noqa: F401
    except ImportError as e:
        return Err(StemError(kind="model_load", detail=f"msst import: {e}"))
    return Ok(RoformerChainHandle(
        config=cfg, device=dev, version=cfg.version, _msst_root=msst,
    ))


def _build_separator(
    h: RoformerChainHandle, spec: ModelSpec, cwd: _MsstCwd,
) -> Result[object, StemError]:
    weights = h._msst_root / "pretrain" / "vocal_models" / spec.ckpt
    if not weights.is_file():
        return Err(StemError(kind="model_load", detail=f"missing checkpoint: {weights}"))
    with cwd:
        from inference.msst_infer import MSSeparator
        from utils.logger import get_logger

        store = (
            {"vocals": "", "instrumental": ""}
            if spec.model_type == "bs_roformer"
            else {"vocals": "", "other": ""}
        )
        return Ok(MSSeparator(
            model_type=spec.model_type,
            config_path=str(Path("configs") / "vocal_models" / f"{spec.ckpt}.yaml"),
            model_path=str(Path("pretrain") / "vocal_models" / spec.ckpt),
            device=h.device,
            output_format="wav",
            store_dirs=store,
            logger=get_logger(),
            debug=False,
        ))


def _run_model(
    h: RoformerChainHandle, spec: ModelSpec, audio_path: Path, scratch: Path, cwd: _MsstCwd,
) -> Result[dict[str, Path], StemError]:
    import librosa

    br = _build_separator(h, spec, cwd)
    if not br.is_ok():
        return br
    sep = br.value
    try:
        mix, _sr = librosa.load(str(audio_path), mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        t0 = time.monotonic()
        with cwd:
            raw = sep.separate(mix)
            sep.del_cache()
        stems = dict(raw)
        if "instrumental" not in stems and "other" in stems:
            stems["instrumental"] = stems["other"]
        _log.info("roformer %s: %.1fs", spec.tag, time.monotonic() - t0)
    except (RuntimeError, OSError, ValueError) as e:
        return Err(StemError(kind="inference", detail=f"{spec.tag}: {e}"))

    out: dict[str, Path] = {}
    tag_dir = scratch / spec.tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("vocals", "instrumental"):
            p = tag_dir / f"{name}.wav"
            sf.write(str(p), stems[name], 44100)
            out[name] = p
    except OSError as e:
        return Err(StemError(kind="disk", detail=str(e)))
    return Ok(out)


def _ensemble(
    paths: list[Path], algorithm: str, cwd: _MsstCwd,
) -> Result[tuple[np.ndarray, int], StemError]:
    try:
        with cwd:
            from utils.ensemble import ensemble_audios
            audio, sr = ensemble_audios([str(p) for p in paths], algorithm, [1.0] * len(paths))
    except (OSError, ValueError, RuntimeError) as e:
        return Err(StemError(kind="inference", detail=f"ensemble: {e}"))
    return Ok((audio, sr))


def _write_stem(path: Path, audio: np.ndarray, sr: int, fmt: str, flac_depth: str) -> Result[None, StemError]:
    try:
        if fmt == "flac":
            sf.write(str(path), audio, sr, format="FLAC", subtype=flac_depth)
        else:
            sf.write(str(path), audio, sr)
    except OSError as e:
        return Err(StemError(kind="disk", detail=str(e)))
    return Ok(None)


def separate(
    h: RoformerChainHandle,
    audio_path: Path,
    out_dir: Path,
    track_audio_id: int,
) -> Result[StemSet, StemError]:
    """Run vocal + instrumental ensembles; write vocals.flac + instrumental.flac."""
    cfg = h.config
    dest = out_dir / str(track_audio_id)
    dest.mkdir(parents=True, exist_ok=True)
    ext = cfg.output_format.lower()
    vocals_dest = dest / f"vocals.{ext}"
    instrumental_dest = dest / f"instrumental.{ext}"

    cwd = _ensure_msst_on_path(h._msst_root)
    with tempfile.TemporaryDirectory(prefix="roformer_") as tmp:
        scratch = Path(tmp)
        cache: dict[str, dict[str, Path]] = {}

        def _cached(spec: ModelSpec) -> Result[dict[str, Path], StemError]:
            if spec.ckpt not in cache:
                r = _run_model(h, spec, audio_path, scratch, cwd)
                if not r.is_ok():
                    return r
                cache[spec.ckpt] = r.value
            return Ok(cache[spec.ckpt])

        vocal_wavs: list[Path] = []
        for spec in cfg.vocal_models:
            r = _cached(spec)
            if not r.is_ok():
                return r
            vocal_wavs.append(r.value["vocals"])

        inst_wavs: list[Path] = []
        for spec in cfg.instrumental_models:
            r = _cached(spec)
            if not r.is_ok():
                return r
            inst_wavs.append(r.value["instrumental"])

        ev = _ensemble(vocal_wavs, cfg.ensemble_algorithm, cwd)
        if not ev.is_ok():
            return ev
        ei = _ensemble(inst_wavs, cfg.ensemble_algorithm, cwd)
        if not ei.is_ok():
            return ei

        v_audio, sr = ev.value
        i_audio, _ = ei.value
        wr = _write_stem(vocals_dest, v_audio, sr, ext, cfg.flac_bit_depth)
        if not wr.is_ok():
            return wr
        wr = _write_stem(instrumental_dest, i_audio, sr, ext, cfg.flac_bit_depth)
        if not wr.is_ok():
            return wr

    return Ok(StemSet(track_audio_id=track_audio_id, stems=(
        StemAsset(track_audio_id=track_audio_id, stem_name="vocals",
                  path=str(vocals_dest), codec=ext),
        StemAsset(track_audio_id=track_audio_id, stem_name="instrumental",
                  path=str(instrumental_dest), codec=ext),
    )))
