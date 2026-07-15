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


def test_roformer_config_batch_size_defaults_to_1() -> None:
    # Default MUST stay 1 — the corpus behavior before the batching knob.
    cfg = RoformerChainConfig.default()
    assert cfg.batch_size == 1


def test_roformer_config_batch_size_parsed_from_dict() -> None:
    cfg = RoformerChainConfig.from_dict(
        {
            "vocal_models": [{"model_type": "bs_roformer", "ckpt": "a.ckpt"}],
            "instrumental_models": [{"model_type": "bs_roformer", "ckpt": "b.ckpt"}],
            "batch_size": 8,
        }
    )
    assert cfg.batch_size == 8


def test_roformer_config_batch_size_absent_defaults_to_1() -> None:
    cfg = RoformerChainConfig.from_dict(
        {
            "vocal_models": [{"model_type": "bs_roformer", "ckpt": "a.ckpt"}],
            "instrumental_models": [{"model_type": "bs_roformer", "ckpt": "b.ckpt"}],
        }
    )
    assert cfg.batch_size == 1


def test_roformer_config_with_batch_size_override() -> None:
    # dataclasses.replace-based override the runner uses to apply --roformer-batch-size.
    cfg = RoformerChainConfig.default().with_batch_size(16)
    assert cfg.batch_size == 16
    # unrelated fields preserved
    assert len(cfg.vocal_models) == 3
