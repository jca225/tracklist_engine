"""Mix registry + paths for taste scraping."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "mixes.yaml"


@dataclass(frozen=True)
class MixTarget:
    mix_id: str
    set_id: str
    title: str
    soundcloud_url: str
    youtube_url: str = ""


@dataclass(frozen=True)
class TasteSettings:
    data_dir: Path
    db_path: Path
    soundcloud_rpm: int


def load_settings(
    *,
    data_dir: Path | None = None,
    db_path: Path | None = None,
) -> TasteSettings:
    root = data_dir or Path(os.environ.get("TASTE_DATA_DIR", REPO_ROOT / "data" / "taste"))
    return TasteSettings(
        data_dir=root,
        db_path=db_path or root / "taste_warehouse.db",
        soundcloud_rpm=int(os.environ.get("TASTE_SC_RPM", "45")),
    )


def load_mixes(config_path: Path = DEFAULT_CONFIG) -> tuple[MixTarget, ...]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mixes = raw.get("mixes") or {}
    out: list[MixTarget] = []
    for key, m in mixes.items():
        out.append(
            MixTarget(
                mix_id=str(key),
                set_id=str(m.get("set_id") or key),
                title=str(m.get("title") or key),
                soundcloud_url=str(m.get("soundcloud_url") or ""),
                youtube_url=str(m.get("youtube_url") or ""),
            )
        )
    return tuple(out)


def mix_by_id(mix_id: str, config_path: Path = DEFAULT_CONFIG) -> MixTarget:
    for m in load_mixes(config_path):
        if m.mix_id == mix_id:
            return m
    raise KeyError(f"mix {mix_id!r} not in {config_path}")
