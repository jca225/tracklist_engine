"""Registry of Essentia pre-trained models we ship downloads for.

Source URLs are from https://essentia.upf.edu/models/. Files land in
`data/essentia_models/` (gitignored). Both the Py 3.14 adapter and the
Py 3.13 worker import this module, so it must use only stdlib.

Usage:
    from .essentia_models import MODELS, ensure_downloaded, models_dir
    ensure_downloaded(models_dir(), [m.name for m in MODELS])
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


@dataclass(frozen=True)
class Model:
    name: str            # short slug used as key
    url: str
    filename: str        # filename under data/essentia_models/
    kind: str            # 'embedding' | 'effnet_head' | 'musicnn_head' | 'yamnet'
    output_node: str     # TF graph output node to read


_BASE: Final = "https://essentia.upf.edu/models"

MODELS: Final[tuple[Model, ...]] = (
    Model(
        name="discogs_effnet",
        url=f"{_BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
        filename="discogs-effnet-bs64-1.pb",
        kind="embedding",
        output_node="PartitionedCall:1",
    ),
    Model(
        name="msd_musicnn",
        url=f"{_BASE}/feature-extractors/musicnn/msd-musicnn-1.pb",
        filename="msd-musicnn-1.pb",
        kind="embedding",
        output_node="model/dense/BiasAdd",
    ),
    Model(
        name="mood_acoustic",
        url=f"{_BASE}/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb",
        filename="mood_acoustic-discogs-effnet-1.pb",
        kind="effnet_head",
        output_node="model/Softmax",
    ),
    Model(
        name="mood_aggressive",
        url=f"{_BASE}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
        filename="mood_aggressive-discogs-effnet-1.pb",
        kind="effnet_head",
        output_node="model/Softmax",
    ),
    Model(
        name="mood_happy",
        url=f"{_BASE}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
        filename="mood_happy-discogs-effnet-1.pb",
        kind="effnet_head",
        output_node="model/Softmax",
    ),
    Model(
        name="voice_instrumental",
        url=f"{_BASE}/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
        filename="voice_instrumental-discogs-effnet-1.pb",
        kind="effnet_head",
        output_node="model/Softmax",
    ),
    Model(
        name="danceability_tf",
        url=f"{_BASE}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
        filename="danceability-discogs-effnet-1.pb",
        kind="effnet_head",
        output_node="model/Softmax",
    ),
    Model(
        name="emomusic",
        url=f"{_BASE}/classification-heads/emomusic/emomusic-msd-musicnn-2.pb",
        filename="emomusic-msd-musicnn-2.pb",
        kind="musicnn_head",
        output_node="model/Identity",
    ),
    Model(
        name="yamnet",
        url=f"{_BASE}/audio-event-recognition/yamnet/audioset-yamnet-1.pb",
        filename="audioset-yamnet-1.pb",
        kind="yamnet",
        output_node="activations",
    ),
)


_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR: Final[Path] = _REPO_ROOT / "data" / "essentia_models"


def models_dir() -> Path:
    return _DEFAULT_MODELS_DIR


def model_path(model: Model, root: Path | None = None) -> Path:
    return (root or models_dir()) / model.filename


def by_name() -> dict[str, Model]:
    return {m.name: m for m in MODELS}


@dataclass(frozen=True)
class DownloadError:
    name: str
    reason: str


@dataclass(frozen=True)
class DownloadReport:
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]      # already present
    failed: tuple[DownloadError, ...]


def ensure_downloaded(
    root: Path | None = None,
    names: tuple[str, ...] | None = None,
) -> DownloadReport:
    root = root or models_dir()
    root.mkdir(parents=True, exist_ok=True)
    selected = MODELS if names is None else tuple(m for m in MODELS if m.name in names)

    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[DownloadError] = []
    for m in selected:
        path = model_path(m, root)
        if path.exists() and path.stat().st_size > 0:
            skipped.append(m.name)
            continue
        try:
            urlretrieve(m.url, path)
            downloaded.append(m.name)
        except (HTTPError, URLError, OSError) as e:
            failed.append(DownloadError(name=m.name, reason=f"{type(e).__name__}: {e}"))
            if path.exists() and path.stat().st_size == 0:
                path.unlink()

    return DownloadReport(
        downloaded=tuple(downloaded),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )


def which_present(root: Path | None = None) -> set[str]:
    root = root or models_dir()
    return {m.name for m in MODELS if model_path(m, root).exists()}
