"""Config for the MSST RoFormer separation chain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MSST_ROOT = _REPO_ROOT / "workspaces" / "msst_webui"

_PRECISIONS = ("fp32", "bf16", "fp16")


def _validated_precision(value: str) -> str:
    if value not in _PRECISIONS:
        raise ValueError(f"precision must be one of {_PRECISIONS}, got {value!r}")
    return value


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    ckpt: str

    @property
    def tag(self) -> str:
        return self.ckpt.replace(".ckpt", "")


@dataclass(frozen=True)
class RoformerChainConfig:
    vocal_models: tuple[ModelSpec, ...]
    instrumental_models: tuple[ModelSpec, ...]
    msst_root: Path
    device: str = "auto"
    ensemble_algorithm: str = "avg_fft"
    output_format: str = "flac"
    flac_bit_depth: str = "PCM_16"
    # Chunks fed to the RoFormer per GPU forward pass. Default 1 = the pre-
    # batching corpus behavior. Raising it (e.g. 8) processes independent
    # audio chunks in parallel — mathematically identical output (chunked
    # overlap-add is deterministic), just higher GPU utilization. Overrides
    # the model YAML's inference.batch_size via MSSeparator(inference_params=).
    batch_size: int = 1
    # Inference compute dtype: fp32 | bf16 | fp16. Non-fp32 wraps the model
    # forward in torch.autocast, which also lets SDPA dispatch flash-attention
    # kernels (they are half-precision-only, so fp32 never engages them).
    # Output is near-identical, NOT bit-identical — any non-fp32 default must
    # first pass the real-music dB gate (streaming_mir precision A/B).
    precision: str = "fp32"

    @property
    def version(self) -> str:
        v = "+".join(m.tag for m in self.vocal_models)
        i = "+".join(m.tag for m in self.instrumental_models)
        return f"roformer:voc={v}|inst={i}@{self.ensemble_algorithm}"

    def with_batch_size(self, n: int) -> RoformerChainConfig:
        """Return a copy with batch_size overridden (runner applies
        --roformer-batch-size). batch_size is NOT in `version` — it changes
        only speed, not output, so cached stems stay valid across values."""
        return replace(self, batch_size=int(n))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoformerChainConfig:
        def _models(key: str) -> tuple[ModelSpec, ...]:
            return tuple(
                ModelSpec(model_type=m["model_type"], ckpt=m["ckpt"]) for m in data[key]
            )

        root = Path(data.get("msst_root", DEFAULT_MSST_ROOT))
        if not root.is_absolute():
            root = _REPO_ROOT / root
        return cls(
            vocal_models=_models("vocal_models"),
            instrumental_models=_models("instrumental_models"),
            msst_root=root,
            device=data.get("device", "auto"),
            ensemble_algorithm=data.get("ensemble_algorithm", "avg_fft"),
            output_format=data.get("output_format", "flac"),
            flac_bit_depth=data.get("flac_bit_depth", "PCM_16"),
            batch_size=int(data.get("batch_size", 1)),
            precision=_validated_precision(data.get("precision", "fp32")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> RoformerChainConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def default(cls) -> RoformerChainConfig:
        return cls.from_yaml(Path(__file__).with_name("roformer_chain.yaml"))
