"""Load mix + ref MERT measure stacks from pi-storage (cached locally)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.adapters.mert_adapter import MERT_DEFAULT_LAYER
from core.result import Err, Ok, Result
from eda.alignment.mert_vectors import probe_vector

PI_HOST = "pi-storage"
_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "mert"
_EXPORT_SCRIPT = Path(__file__).resolve().parent / "export_mert_from_pi.py"
_PROBE_LAYER = MERT_DEFAULT_LAYER


@dataclass(frozen=True)
class MertSeries:
    """Per-measure probe-layer vectors aligned to beat grid times."""

    start_s: np.ndarray   # (n,)
    end_s: np.ndarray     # (n,)
    vectors: np.ndarray   # (n, dim) float32

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def n_measures(self) -> int:
        return int(self.vectors.shape[0])

    def pool(self, start_s: float, end_s: float) -> np.ndarray | None:
        if self.n_measures == 0:
            return None
        mid = 0.5 * (self.start_s + self.end_s)
        mask = (mid >= start_s) & (mid <= end_s)
        if not mask.any():
            idx = int(np.argmin(np.abs(mid - start_s)))
            return self.vectors[idx]
        return self.vectors[mask].mean(axis=0).astype(np.float32)

    def track_mean(self) -> np.ndarray:
        if self.n_measures == 0:
            raise ValueError("empty MertSeries")
        return self.vectors.mean(axis=0).astype(np.float32)


def _cache_path(set_id: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{set_id}_mert.npz"


def _pull_from_pi(set_id: str) -> Result[Path, str]:
    remote_script = f"/tmp/export_mert_{set_id}.py"
    remote_out = f"/tmp/align_mert_{set_id}.npz"
    fd, local_s = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    local = Path(local_s)
    try:
        subprocess.run(
            ["scp", str(_EXPORT_SCRIPT), f"{PI_HOST}:{remote_script}"],
            check=True,
            capture_output=True,
            text=True,
        )
        run = (
            f"~/tracklist_engine/venvs/audio/bin/python {remote_script} "
            f"{set_id} {remote_out} {_PROBE_LAYER}"
        )
        subprocess.run(["ssh", PI_HOST, run], check=True, capture_output=True, text=True)
        subprocess.run(
            ["scp", f"{PI_HOST}:{remote_out}", str(local)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["ssh", PI_HOST, f"rm -f {remote_script} {remote_out}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        local.unlink(missing_ok=True)
        detail = (e.stderr or e.stdout or str(e)).strip()
        return Err(detail)
    return Ok(local)


_FP16_MAX = 65504.0


def _finite(vec: np.ndarray, what: str) -> np.ndarray:
    """Clamp non-finite values: fp16 overflow at embed time leaves ±inf in
    stored blobs (BB12 ref dkg1c995), and a single inf NaNs the whole
    training batch -> NaN head -> NaN decode curves (found 2026-06-11)."""
    bad = ~np.isfinite(vec)
    if bad.any():
        import logging

        logging.getLogger(__name__).warning(
            "mert_store: %d non-finite value(s) clamped in %s", int(bad.sum()), what
        )
        vec = np.nan_to_num(vec, nan=0.0, posinf=_FP16_MAX, neginf=-_FP16_MAX)
    return vec


def _parse_bundle(path: Path) -> tuple[int, MertSeries, dict[str, MertSeries]]:
    with np.load(path, allow_pickle=False) as z:
        mix = MertSeries(
            start_s=z["mix_start"], end_s=z["mix_end"],
            vectors=_finite(z["mix_vec"], "mix"),
        )
        ref_ids = json.loads(str(z["ref_ids"]))
        refs = {
            tid: MertSeries(
                start_s=z[f"ref_{tid}_start"],
                end_s=z[f"ref_{tid}_end"],
                vectors=_finite(z[f"ref_{tid}_vec"], f"ref {tid}"),
            )
            for tid in ref_ids
        }
        return int(z["set_audio_id"]), mix, refs


def load_bb12_mert(
    set_id: str = "1fsnxchk",
    *,
    refresh: bool = False,
) -> Result[tuple[int, MertSeries, dict[str, MertSeries]], str]:
    cache = _cache_path(set_id)
    if refresh or not cache.is_file():
        match _pull_from_pi(set_id):
            case Err(msg):
                return Err(msg)
            case Ok(tmp):
                tmp.replace(cache)
    try:
        return Ok(_parse_bundle(cache))
    except (OSError, KeyError, ValueError) as e:
        return Err(str(e))


def decode_probe_vector(embedding_bytes: bytes, dim: int, *, layer: int = _PROBE_LAYER) -> np.ndarray:
    return probe_vector(embedding_bytes, dim, layer=layer)
