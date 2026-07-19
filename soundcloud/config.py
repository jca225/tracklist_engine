"""SoundCloud data-lake settings + env resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/storage/data/soundcloud")


@dataclass(frozen=True)
class SoundCloudSettings:
    data_root: Path
    db_path: Path
    raw_root: Path
    rpm: int


def load_settings(
    *,
    data_root: Path | None = None,
    rpm: int | None = None,
) -> SoundCloudSettings:
    root = data_root or Path(os.environ.get("SC_LAKE_ROOT", str(DEFAULT_ROOT)))
    resolved_rpm = rpm if rpm is not None else int(os.environ.get("SC_LAKE_RPM", "60"))
    return SoundCloudSettings(
        data_root=root,
        db_path=root / "sc_lake.db",
        raw_root=root / "raw",
        rpm=resolved_rpm,
    )
