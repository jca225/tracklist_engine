"""Config for the UVR stem-separation chain (`uvr_chain_adapter`).

The chain is a sequence of `audio-separator` stages. Each stage runs one model
(or an ensemble of several, combined natively by audio-separator) on the
`Vocals` output of the previous stage, progressively cleaning the lead vocal:

    isolate → lead (ensemble) → dereverb → deecho → denoise

The persisted instrumental comes from the `isolate` stage's `Instrumental`
output (`instrumental_from`). Per-stage stem selection is done by matching the
parenthesised label audio-separator writes into each output filename
(`keep_match` / `byproduct_match`), so we never rely on file ordering.

Config is data, not code: reorder/retune/disable stages by editing
`uvr_chain.yaml` with no code changes. Malformed config fails fast at load
(this is an edge, not core — see the project style guide).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default persistent model cache. Override per-host via the yaml `model_dir`
# or the CLI. Keeps the ~6 UVR models downloaded once and reused across runs.
DEFAULT_MODEL_DIR = "~/uvr-models"


@dataclass(frozen=True)
class StageSpec:
    """One separation stage.

    `models`: a single filename, or several → audio-separator ensembles them
    with `ensemble_algorithm` (e.g. 'avg_fft' = magnitude-spectrogram average).
    `arch`: routes `params` into the right audio-separator param dict
    ('mdx'→mdx_params, 'vr'→vr_params, 'mdxc'→mdxc_params).
    `keep_match`: substring of the output stem label to forward to the next
    stage / treat as the final vocal (e.g. 'Vocals', 'No Reverb').
    `byproduct_match`: the complementary stem to keep as a loose artifact in
    standalone/QA mode (Chorus/Reverb/Echo/Noise); None to discard.
    """
    name: str
    models: tuple[str, ...]
    arch: str
    keep_match: str
    byproduct_match: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    ensemble_algorithm: str = "avg_fft"

    @property
    def is_ensemble(self) -> bool:
        return len(self.models) > 1


@dataclass(frozen=True)
class ChainConfig:
    stages: tuple[StageSpec, ...]
    instrumental_from: str          # stage name whose instrumental we branch off
    instrumental_match: str = "Instrumental"
    # Feed the branched-off instrumental through Demucs and persist its
    # drums+bass+other re-sum (dropping Demucs' near-empty vocal). Strips
    # residual vocal bleed for a cleaner instrumental. False → persist the
    # instrumental straight from `instrumental_from`.
    instrumental_cascade: bool = True
    model_dir: str = DEFAULT_MODEL_DIR
    output_format: str = "FLAC"
    enable_ensemble: bool = True     # False → ensemble stages collapse to models[0]

    def __post_init__(self) -> None:
        names = [s.name for s in self.stages]
        if not self.stages:
            raise ValueError("ChainConfig.stages is empty")
        if self.instrumental_from not in names:
            raise ValueError(
                f"instrumental_from={self.instrumental_from!r} not in stages {names}"
            )

    @property
    def resolved_model_dir(self) -> Path:
        return Path(self.model_dir).expanduser()

    def effective_models(self, stage: StageSpec) -> tuple[str, ...]:
        """The model list to actually load — honours `enable_ensemble`."""
        if stage.is_ensemble and not self.enable_ensemble:
            return (stage.models[0],)
        return stage.models

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChainConfig:
        try:
            stages = tuple(
                StageSpec(
                    name=s["name"],
                    models=tuple(s["models"]),
                    arch=s["arch"],
                    keep_match=s["keep_match"],
                    byproduct_match=s.get("byproduct_match"),
                    params=dict(s.get("params", {})),
                    ensemble_algorithm=s.get("ensemble_algorithm", "avg_fft"),
                )
                for s in data["stages"]
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed UVR chain config: {e}") from e
        return cls(
            stages=stages,
            instrumental_from=data["instrumental_from"],
            instrumental_match=data.get("instrumental_match", "Instrumental"),
            model_dir=data.get("model_dir", DEFAULT_MODEL_DIR),
            output_format=data.get("output_format", "FLAC"),
            enable_ensemble=data.get("enable_ensemble", True),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ChainConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def default(cls) -> ChainConfig:
        """The reference chain (matches uvr_chain.yaml), usable without a file."""
        return cls.from_yaml(Path(__file__).with_name("uvr_chain.yaml"))
