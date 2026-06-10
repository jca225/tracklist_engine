"""RoFormer adapter contract tests (no GPU / no MSST inference)."""
from __future__ import annotations

from pathlib import Path

from analysis.roformer_config import RoformerChainConfig


def test_roformer_config_default_loads() -> None:
    cfg = RoformerChainConfig.default()
    assert len(cfg.vocal_models) == 3
    assert len(cfg.instrumental_models) == 2
    assert cfg.ensemble_algorithm == "avg_fft"
    assert "roformer:" in cfg.version


def test_roformer_config_msst_root_resolves() -> None:
    cfg = RoformerChainConfig.default()
    assert cfg.msst_root.is_dir() or cfg.msst_root.name == "msst_webui"
