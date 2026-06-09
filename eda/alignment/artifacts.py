"""Local cache format for bar-synchronous mix MERT."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MixMertArtifact:
    set_id: str
    bar_start_s: np.ndarray   # (n_bars,)
    bar_end_s: np.ndarray     # (n_bars,)
    mert: np.ndarray          # (n_bars, dim) float32 — probe layer already pooled
    mert_layer: int
    mert_model: str

    @property
    def n_bars(self) -> int:
        return int(self.bar_start_s.shape[0])

    @property
    def dim(self) -> int:
        return int(self.mert.shape[1])


def save_mix_mert_artifact(
    path: Path,
    *,
    set_id: str,
    bar_start_s: np.ndarray,
    bar_end_s: np.ndarray,
    mert: np.ndarray,
    mert_layer: int,
    mert_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "set_id": set_id,
        "mert_layer": mert_layer,
        "mert_model": mert_model,
        "n_bars": int(len(bar_start_s)),
        "dim": int(mert.shape[1]),
    }
    np.savez_compressed(
        path,
        bar_start_s=bar_start_s.astype(np.float64),
        bar_end_s=bar_end_s.astype(np.float64),
        mert=mert.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )


def load_mix_mert_artifact(path: Path) -> MixMertArtifact:
    with np.load(path, allow_pickle=False) as z:
        bar_start_s = z["bar_start_s"]
        bar_end_s = z["bar_end_s"]
        mert = z["mert"]
        meta = json.loads(str(z["meta_json"]))
    return MixMertArtifact(
        set_id=str(meta["set_id"]),
        bar_start_s=bar_start_s,
        bar_end_s=bar_end_s,
        mert=mert,
        mert_layer=int(meta["mert_layer"]),
        mert_model=str(meta["mert_model"]),
    )
