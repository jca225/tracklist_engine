"""Structural tests: adapter modules resolve repo-root paths correctly."""
from __future__ import annotations

from pathlib import Path

import pytest

from analysis.adapters import cue_detr_adapter, essentia_adapter, essentia_models

REPO = Path(__file__).resolve().parents[1]

INGEST_ADAPTERS = ("spotdl_adapter.py", "ytmusic_adapter.py")


def test_cue_detr_dir_under_repo() -> None:
    assert cue_detr_adapter._CUE_DETR_DIR == REPO / "cue-detr"
    assert cue_detr_adapter._CUE_DETR_DIR.is_dir()


def test_essentia_adapter_repo_root() -> None:
    assert essentia_adapter._REPO_ROOT == REPO


def test_essentia_models_repo_root() -> None:
    assert essentia_models._REPO_ROOT == REPO
    assert essentia_models.models_dir() == REPO / "data" / "essentia_models"


@pytest.mark.parametrize("filename", INGEST_ADAPTERS)
def test_ingest_adapter_repo_depth(filename: str) -> None:
    """Ingest adapters pull yt-dlp at import — verify depth statically."""
    text = (REPO / "ingest" / "adapters" / filename).read_text(encoding="utf-8")
    assert "parents[2]" in text
    assert "parents[3]" not in text
