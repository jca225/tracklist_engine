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


def test_roformer_config_shipped_default_is_4() -> None:
    # roformer_chain.yaml ships batch_size: 4 — the validated sweet spot
    # (1.65x, plateaus there; output batch-invariant). See WS-batching result.
    cfg = RoformerChainConfig.default()
    assert cfg.batch_size == 4


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


def test_roformer_config_precision_parsed_from_dict() -> None:
    cfg = RoformerChainConfig.from_dict(
        {
            "vocal_models": [{"model_type": "bs_roformer", "ckpt": "a.ckpt"}],
            "instrumental_models": [{"model_type": "bs_roformer", "ckpt": "b.ckpt"}],
            "precision": "bf16",
        }
    )
    assert cfg.precision == "bf16"


def test_roformer_config_precision_defaults_to_fp32() -> None:
    # fp32 = shipped behavior; bf16/fp16 are experiment-only until they pass
    # the near-identical gate (see streaming_mir precision A/B).
    assert RoformerChainConfig.default().precision == "fp32"


def test_roformer_config_precision_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError):
        RoformerChainConfig.from_dict(
            {
                "vocal_models": [{"model_type": "bs_roformer", "ckpt": "a.ckpt"}],
                "instrumental_models": [
                    {"model_type": "bs_roformer", "ckpt": "b.ckpt"}
                ],
                "precision": "int8",
            }
        )
