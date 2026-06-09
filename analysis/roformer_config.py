"""Config for the MSST RoFormer separation chain."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MSST_ROOT = _REPO_ROOT / "workspaces" / "msst_webui"


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

    @property
    def version(self) -> str:
        v = "+".join(m.tag for m in self.vocal_models)
        i = "+".join(m.tag for m in self.instrumental_models)
        return f"roformer:voc={v}|inst={i}@{self.ensemble_algorithm}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoformerChainConfig:
        def _models(key: str) -> tuple[ModelSpec, ...]:
            return tuple(
                ModelSpec(model_type=m["model_type"], ckpt=m["ckpt"])
                for m in data[key]
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
        )

    @classmethod
    def from_yaml(cls, path: Path) -> RoformerChainConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def default(cls) -> RoformerChainConfig:
        return cls.from_yaml(Path(__file__).with_name("roformer_chain.yaml"))
