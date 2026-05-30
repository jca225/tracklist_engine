"""Boundary over the `audio-separator` Python API — one separation stage.

Used by `uvr_chain_adapter`. A stage is a single `Separator` with a loaded
model (or a list of models → audio-separator ensembles them natively with the
configured algorithm, e.g. 'avg_fft' = magnitude-spectrogram average).

Output-file tracking is done off the **returned file list** (never globbing):
audio-separator writes the stem label as a parenthesised token into each
filename, e.g. `song_(No Reverb)_UVR-De-Echo-Aggressive.flac`, which we parse
back out so the orchestrator can pick the stem to forward.

Device is auto-resolved by audio-separator; we read `torch_device` after load
to log the provider and **loud-fail when CUDA was demanded but it fell back to
CPU** (the spec's silent-CPU-fallback guard) — on `auto`/`mps`/`cpu` hosts a
CPU/CoreML provider is expected, so we only warn.
"""
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from core.result import Err, Ok, Result
from ..errors import StemError

_log = logging.getLogger(__name__)

# The parenthesised stem label audio-separator writes into output filenames.
_LABEL_RE = re.compile(r"\(([^)]+)\)")

# Stage arch → the audio-separator constructor param-dict it tunes.
_ARCH_PARAM_KEY = {"mdx": "mdx_params", "vr": "vr_params", "mdxc": "mdxc_params"}


@dataclass(frozen=True)
class SeparatorStage:
    _sep: object                   # audio_separator.separator.Separator (model loaded)
    name: str
    models: tuple[str, ...]
    device: str                    # resolved torch device: 'cuda' | 'mps' | 'cpu'


def build(
    *,
    name: str,
    models: tuple[str, ...],
    arch: str,
    params: dict,
    ensemble_algorithm: str,
    device: str,
    model_dir: Path,
    output_format: str,
) -> Result[SeparatorStage, StemError]:
    """Construct a Separator and load its model(s). One per chain stage."""
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        return Err(StemError(kind="model_load", detail=f"audio-separator import: {e}"))

    param_key = _ARCH_PARAM_KEY.get(arch)
    if param_key is None:
        return Err(StemError(kind="model_load", detail=f"{name}: unknown arch {arch!r}"))

    # Merge stage overrides onto the library's defaults for this arch dict so we
    # never drop required keys (window_size, hop_length, …) the model needs.
    defaults = inspect.signature(Separator.__init__).parameters[param_key].default
    merged = {**defaults, **params} if isinstance(defaults, dict) else dict(params)

    kwargs: dict = {
        "log_level": logging.WARNING,
        "model_file_dir": str(model_dir),
        "output_format": output_format,
        param_key: merged,
    }
    if len(models) > 1:
        kwargs["ensemble_algorithm"] = ensemble_algorithm

    # audio-separator wraps onnxruntime / torch / model downloads, whose failure
    # surface is wide and version-dependent — this adapter boundary converts any
    # of it into a StemError Result rather than leaking exceptions upward.
    try:
        sep = Separator(**kwargs)
        sep.load_model(model_filename=(list(models) if len(models) > 1 else models[0]))
    except Exception as e:  # noqa: BLE001 — third-party boundary, wrap as Result
        return Err(StemError(kind="model_load", detail=f"{name}: {e}"))

    resolved = str(getattr(sep, "torch_device", "") or "cpu").lower()
    if device == "cuda" and "cuda" not in resolved:
        return Err(StemError(
            kind="model_load",
            detail=(f"{name}: CUDA requested but audio-separator resolved device="
                    f"{resolved!r} — install onnxruntime-gpu and verify CUDA/cuDNN"),
        ))
    _log.info("uvr stage %r: models=%s device=%s", name, list(models), resolved)
    return Ok(SeparatorStage(_sep=sep, name=name, models=tuple(models), device=resolved))


def run(
    stage: SeparatorStage,
    input_path: Path,
    out_dir: Path,
    *,
    single_stem: str | None = None,
) -> Result[dict[str, Path], StemError]:
    """Separate `input_path` into `out_dir`; return {stem_label: path}.

    `single_stem` (a stem label) restricts output to just that stem to save I/O.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sep = stage._sep
    # The model's output_dir is captured at load_model time (in build()), so we
    # must redirect both the Separator and its loaded model_instance per-run —
    # setting only sep.output_dir leaks files into the cwd. The ensemble path
    # writes its final mix via sep.output_dir; single models via model_instance.
    sep.output_dir = str(out_dir)
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = str(out_dir)
    sep.output_single_stem = single_stem
    try:
        outputs = sep.separate(str(input_path))
    except Exception as e:  # noqa: BLE001 — third-party boundary, wrap as Result
        return Err(StemError(kind="inference", detail=f"{stage.name}: {e}"))

    labeled: dict[str, Path] = {}
    for out in outputs:
        p = Path(out)
        if not p.is_absolute():
            p = out_dir / p          # audio-separator returns basenames rel. to output_dir
        # audio-separator appends `_(Stem)_<Model>` to the *input* name each
        # pass, so a chained file accumulates tokens like
        # `clip_(Vocals)_..._(No Reverb)_Reverb_HQ`. The *last* parenthesised
        # token is this stage's stem (the model name carries no parens).
        labels = _LABEL_RE.findall(p.name)
        labeled[labels[-1] if labels else p.stem] = p
    if not labeled:
        return Err(StemError(kind="inference", detail=f"{stage.name}: produced no outputs"))
    return Ok(labeled)


def match_stem(labeled: dict[str, Path], needle: str) -> Path | None:
    """Pick the output stem matching `needle` (case-insensitive).

    Exact label match wins; otherwise the **shortest** label containing
    `needle`. The shortest-match rule disambiguates the De-Echo / De-Reverb /
    De-Noise stems, whose kept label ("No Reverb") is a superstring of its
    byproduct label ("Reverb") — so needle "Reverb" correctly picks "Reverb",
    not "No Reverb".
    """
    nl = needle.lower()
    for label, path in labeled.items():
        if label.lower() == nl:
            return path
    matches = [(label, path) for label, path in labeled.items() if nl in label.lower()]
    if not matches:
        return None
    return min(matches, key=lambda lp: len(lp[0]))[1]
