"""UVR cleanup chain — a drop-in alternative to `demucs_adapter`.

Same interface (`load()` → handle, `separate()` → `StemSet(vocals,
instrumental)`) so the analysis pipeline can select it via
`load_analyzers(separator="uvr")` with zero downstream / schema change.

Internally runs the config-driven `audio-separator` chain
(`analysis/uvr_chain.yaml`): isolate → lead-vocal ensemble → dereverb →
de-echo → denoise, forwarding each stage's cleaned vocal to the next. The
persisted `vocals` is the final stage's output; `instrumental` is the isolate
stage's Instrumental.

This is much slower than Demucs (~5 sequential model passes vs 1), so it's an
opt-in quality backend, not the default. Per-stage timing is logged.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from core.result import Err, Ok, Result
from ..errors import StemError
from ..models import StemAsset, StemSet
from ..separation_config import ChainConfig
from . import audio_separator_adapter as asa
from . import demucs_adapter

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UvrChainHandle:
    _stages: tuple[asa.SeparatorStage, ...]
    config: ChainConfig
    device: str
    version: str                   # e.g. 'uvr_chain:isolate+lead+dereverb+deecho+denoise'
    _demucs: demucs_adapter.DemucsHandle | None = None   # instrumental cascade, if enabled


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


def load(
    config: ChainConfig | None = None, device: str = "auto"
) -> Result[UvrChainHandle, StemError]:
    """Construct + load every stage's model once. Fails fast on first error."""
    cfg = config or ChainConfig.default()
    dev = _resolve_device(device)
    model_dir = cfg.resolved_model_dir
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return Err(StemError(kind="model_load", detail=f"model_dir {model_dir}: {e}"))

    stages: list[asa.SeparatorStage] = []
    for spec in cfg.stages:
        r = asa.build(
            name=spec.name,
            models=cfg.effective_models(spec),
            arch=spec.arch,
            params=spec.params,
            ensemble_algorithm=spec.ensemble_algorithm,
            device=dev,
            model_dir=model_dir,
            output_format=cfg.output_format,
        )
        if not r.is_ok():
            return r
        stages.append(r.value)

    demucs_h = None
    version = "uvr_chain:" + "+".join(s.name for s in cfg.stages)
    if cfg.instrumental_cascade:
        d = demucs_adapter.load(device=device)
        if not d.is_ok():
            return d
        demucs_h = d.value
        version += f"|inst={demucs_h.version}"

    return Ok(UvrChainHandle(
        _stages=tuple(stages), config=cfg, device=dev, version=version, _demucs=demucs_h,
    ))


def separate(
    h: UvrChainHandle,
    audio_path: Path,
    out_dir: Path,
    track_audio_id: int,
    byproducts_dir: Path | None = None,
) -> Result[StemSet, StemError]:
    """Run the chain; write `vocals` + `instrumental` into `out_dir/<id>/`.

    Same output contract as `demucs_adapter.separate`. Intermediate stage
    outputs go to a scratch tempdir and are discarded — unless
    `byproducts_dir` is set (standalone/QA mode), in which case each stage's
    byproduct stem (chorus/reverb/echo/noise) is copied there as
    `<stage>_<label>.<ext>`.
    """
    cfg = h.config
    dest = out_dir / str(track_audio_id)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        if byproducts_dir is not None:
            byproducts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return Err(StemError(kind="disk", detail=str(e)))

    ext = cfg.output_format.lower()
    vocals_dest = dest / f"vocals.{ext}"
    instrumental_dest = dest / f"instrumental.{ext}"
    current = Path(audio_path)
    instrumental_src: Path | None = None

    with tempfile.TemporaryDirectory(prefix="uvr_") as tmp:
        for spec, stage in zip(cfg.stages, h._stages):
            stage_dir = Path(tmp) / spec.name
            t0 = time.monotonic()
            r = asa.run(stage, current, stage_dir)
            if not r.is_ok():
                return r
            labeled = r.value

            keep = asa.match_stem(labeled, spec.keep_match)
            if keep is None:
                return Err(StemError(
                    kind="inference",
                    detail=f"{spec.name}: kept stem {spec.keep_match!r} not in {list(labeled)}",
                ))
            if spec.name == cfg.instrumental_from:
                instrumental_src = asa.match_stem(labeled, cfg.instrumental_match)
                if instrumental_src is None:
                    return Err(StemError(
                        kind="inference",
                        detail=f"{spec.name}: instrumental {cfg.instrumental_match!r} "
                               f"not in {list(labeled)}",
                    ))
            if byproducts_dir is not None and spec.byproduct_match:
                bp = asa.match_stem(labeled, spec.byproduct_match)
                if bp is not None:
                    label = spec.byproduct_match.replace(" ", "_")
                    try:
                        shutil.copyfile(bp, byproducts_dir / f"{spec.name}_{label}.{ext}")
                    except OSError as e:
                        return Err(StemError(kind="disk", detail=str(e)))
            _log.info(
                "uvr stage %r (%s): %.1fs -> %s",
                spec.name, "+".join(stage.models), time.monotonic() - t0, keep.name,
            )
            current = keep

        if instrumental_src is None:
            return Err(StemError(
                kind="inference",
                detail=f"instrumental_from={cfg.instrumental_from!r} produced no instrumental",
            ))

        # Cascade: clean the branched-off instrumental through Demucs and keep
        # its drums+bass+other re-sum (demucs_adapter's 'instrumental' stem),
        # dropping the near-empty vocal — strips residual vocal bleed.
        if h._demucs is not None:
            t0 = time.monotonic()
            dr = demucs_adapter.separate(
                h._demucs, instrumental_src, Path(tmp) / "demucs_inst", track_audio_id,
            )
            if not dr.is_ok():
                return dr
            cascaded = next(
                (s.path for s in dr.value.stems if s.stem_name == "instrumental"), None,
            )
            if cascaded is None:
                return Err(StemError(kind="inference",
                                     detail="demucs cascade produced no instrumental stem"))
            _log.info("uvr instrumental cascade (demucs): %.1fs", time.monotonic() - t0)
            instrumental_src = Path(cascaded)

        # Copy out of the tempdir before it's cleaned up.
        try:
            shutil.copyfile(current, vocals_dest)
            shutil.copyfile(instrumental_src, instrumental_dest)
        except OSError as e:
            return Err(StemError(kind="disk", detail=str(e)))

    return Ok(StemSet(track_audio_id=track_audio_id, stems=(
        StemAsset(track_audio_id=track_audio_id, stem_name="vocals",
                  path=str(vocals_dest), codec=ext),
        StemAsset(track_audio_id=track_audio_id, stem_name="instrumental",
                  path=str(instrumental_dest), codec=ext),
    )))
