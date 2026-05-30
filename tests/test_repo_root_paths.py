"""Structural tests: adapter modules resolve repo-root paths correctly."""
from __future__ import annotations

from pathlib import Path

from analysis.adapters import cue_detr_adapter, essentia_adapter, essentia_models
from ingest.adapters import spotdl_adapter, ytmusic_adapter

REPO = Path(__file__).resolve().parents[1]


def test_cue_detr_dir_under_repo() -> None:
    assert cue_detr_adapter._CUE_DETR_DIR == REPO / "cue-detr"
    assert cue_detr_adapter._CUE_DETR_DIR.is_dir()


def test_essentia_adapter_repo_root() -> None:
    assert essentia_adapter._REPO_ROOT == REPO


def test_essentia_models_repo_root() -> None:
    assert essentia_models._REPO_ROOT == REPO
    assert essentia_models.models_dir() == REPO / "data" / "essentia_models"


def test_ytmusic_adapter_repo_root() -> None:
    repo_root = Path(ytmusic_adapter.__file__).resolve().parents[2]
    assert repo_root == REPO


def test_spotdl_adapter_repo_root() -> None:
    repo_root = Path(spotdl_adapter.__file__).resolve().parents[2]
    assert repo_root == REPO
