"""Per-span fingerprint placement refinement (sharpness-gated argmax).

Coarse ``predict_sequence`` placement is ~37 s MAE — mostly the monotonic prior.
Landmark fingerprints can lock individual spans to sub-bar when the offset
histogram is peaked (see ``docs/fine_placement_plan.md``). Joint re-decode on
flat curves backfires, so this module applies **independent per-span argmax**
inside a corridor around the coarse start, gated on peak sharpness (z-score
within the band).

Usage from infer::

    ctx = FpPlacementContext.from_set(set_id, measure_mid_s=mix.start_s + ...)
    refined = refine_placements_fp(preds, ctx, band_s=45.0, gate_z=1.0)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .fp_index import DEFAULT_CACHE_DIR, FpKey, load
from .mix_fp_hits import load_mix_mono, placement_curve
from .records import SpanPrediction

log = logging.getLogger(__name__)

_ALIGNING_ROOT = Path.home() / "aligning"


def find_aligning_dir(set_id: str) -> Path | None:
    hits = sorted(_ALIGNING_ROOT.glob(f"{set_id}__*"))
    return hits[0] if hits else None


@dataclass
class FpPlacementContext:
    """Mix mono audio + ref paths + measure grid for curve argmax."""

    mix_y: np.ndarray
    measure_mid_s: np.ndarray
    rid2path: dict[str, str]
    cache_dir: Path
    db_path: Path | None = None

    @classmethod
    def from_set(
        cls,
        set_id: str,
        *,
        measure_mid_s: np.ndarray,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        db_path: Path | None = None,
        mix_path: Path | None = None,
    ) -> FpPlacementContext | None:
        d = find_aligning_dir(set_id)
        if d is None:
            log.warning("fp_placement: no aligning dir for %s", set_id)
            return None
        if mix_path is None:
            for name in ("mix.m4a", "mix_instrumental.flac", "mix.flac"):
                cand = d / name
                if cand.is_file():
                    mix_path = cand
                    break
        if mix_path is None or not mix_path.is_file():
            log.warning("fp_placement: no mix audio in %s", d)
            return None
        manifest = d / "manifest.json"
        if not manifest.is_file():
            log.warning("fp_placement: missing manifest in %s", d)
            return None
        man = json.loads(manifest.read_text())
        rid2path: dict[str, str] = {}
        for t in man["tracks"]:
            if t.get("recording_id") and t.get("local_path"):
                rid2path[str(t["recording_id"])] = str(t["local_path"])
            if t.get("track_id") and t.get("local_path"):
                rid2path.setdefault(str(t["track_id"]), str(t["local_path"]))
        mix_y = load_mix_mono(mix_path)
        return cls(
            mix_y=mix_y,
            measure_mid_s=np.asarray(measure_mid_s, dtype=np.float64),
            rid2path=rid2path,
            cache_dir=cache_dir,
            db_path=db_path,
        )


def _argmax_start(
    curve: np.ndarray,
    measure_mid_s: np.ndarray,
    coarse_s: float,
    *,
    band_s: float,
) -> tuple[float, float, float]:
    """Return (best_start_s, peak_score, sharpness_z) within the band."""
    from .sequence_decode import NEG

    lo = coarse_s - band_s
    hi = coarse_s + band_s
    mask = (measure_mid_s >= lo) & (measure_mid_s <= hi) & (curve > NEG / 2)
    if not mask.any():
        return coarse_s, float("-inf"), 0.0
    band_scores = curve[mask].astype(np.float64)
    mu, sig = band_scores.mean(), band_scores.std() + 1e-9
    z_all = (band_scores - mu) / sig
    j_local = int(np.argmax(band_scores))
    z_peak = float(z_all[j_local])
    idxs = np.flatnonzero(mask)
    best_i = int(idxs[j_local])
    return float(measure_mid_s[best_i]), float(band_scores[j_local]), z_peak


def refine_placements_fp(
    preds: tuple[SpanPrediction, ...],
    ctx: FpPlacementContext,
    *,
    band_s: float = 45.0,
    win_s: float = 12.0,
    gate_z: float = 1.0,
    weight: float = 1.0,
) -> tuple[SpanPrediction, ...]:
    """Snap each span's start when fp curve peak clears ``gate_z`` in-band."""
    out: list[SpanPrediction] = []
    for p in preds:
        rid = p.recording_id
        if not rid:
            out.append(p)
            continue
        key = FpKey(rid, p.claimed_stem or "regular")
        ref_fp = load(key, cache_dir=ctx.cache_dir, db_path=ctx.db_path)
        if ref_fp is None:
            out.append(p)
            continue
        curve = placement_curve(
            ctx.mix_y,
            ref_fp=ref_fp,
            ref_y=None,
            measure_mid_s=ctx.measure_mid_s,
            coarse_start_s=float(p.set_start_s),
            band_s=band_s,
            win_s=win_s,
        )
        if weight != 1.0:
            curve = curve * weight
        new_start, peak, z = _argmax_start(
            curve, ctx.measure_mid_s, float(p.set_start_s), band_s=band_s
        )
        if z < gate_z or peak <= 0:
            out.append(p)
            continue
        delta = new_start - p.set_start_s
        out.append(
            replace(
                p,
                set_start_s=new_start,
                set_end_s=p.set_end_s + delta,
                confidence=float(p.confidence) + 0.01 * z,
            )
        )
    return tuple(out)
