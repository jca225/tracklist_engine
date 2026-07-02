"""Scenario sampler: host bed + 1–2 acap overlays with curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import (
    BedEntry,
    PayloadEntry,
    StemCatalog,
    compatible,
    pitch_shift_semi,
    tempo_ratio,
)

CURRICULUM = {
    "easy": {"n_overlays": 1, "max_key_dist": 0, "max_bpm_fold": 0.02},
    "medium": {"n_overlays": 1, "max_key_dist": 1, "max_bpm_fold": 0.05},
    "hard": {"n_overlays": 2, "max_key_dist": 2, "max_bpm_fold": 0.08},
}


@dataclass(frozen=True)
class OverlaySpec:
    payload: PayloadEntry
    ref_start_s: float
    set_start_s: float
    set_end_s: float
    tempo_ratio: float
    pitch_shift_semi: int


@dataclass(frozen=True)
class MashupScenario:
    mix_id: str
    bed: BedEntry
    bed_ref_start_s: float
    mix_duration_s: float
    overlays: tuple[OverlaySpec, ...]
    curriculum: str


def _sample_overlay(
    rng: np.random.Generator,
    bed: BedEntry,
    payload: PayloadEntry,
    *,
    mix_duration_s: float,
    entry_lo: float,
    entry_hi: float,
    overlay_lo: float,
    overlay_hi: float,
) -> OverlaySpec | None:
    overlay_dur = float(rng.uniform(overlay_lo, overlay_hi))
    entry = float(
        rng.uniform(entry_lo, min(entry_hi, mix_duration_s - overlay_dur - 4.0))
    )
    set_end = min(mix_duration_s, entry + overlay_dur)
    if set_end - entry < 8.0:
        return None

    tr = tempo_ratio(bed, payload)
    ref_start = float(rng.uniform(10.0, 90.0))

    return OverlaySpec(
        payload=payload,
        ref_start_s=ref_start,
        set_start_s=entry,
        set_end_s=set_end,
        tempo_ratio=tr,
        pitch_shift_semi=pitch_shift_semi(bed, payload),
    )


def sample_scenario(
    catalog: StemCatalog,
    *,
    mix_id: str,
    curriculum: str,
    rng: np.random.Generator,
    mix_duration_s: float = 90.0,
) -> MashupScenario | None:
    if len(catalog.beds) < 1 or len(catalog.payloads) < 1:
        return None
    cfg = CURRICULUM.get(curriculum, CURRICULUM["medium"])
    n_overlays = cfg["n_overlays"]

    bed_indices = list(
        rng.choice(len(catalog.beds), size=min(len(catalog.beds), 30), replace=False)
    )
    for bi in bed_indices:
        bed = catalog.beds[bi]
        payloads = [
            p
            for p in catalog.payloads
            if compatible(
                bed,
                p,
                max_key_dist=cfg["max_key_dist"],
                max_bpm_fold=cfg["max_bpm_fold"],
            )
        ]
        if len(payloads) < n_overlays:
            continue
        chosen = list(rng.choice(len(payloads), size=n_overlays, replace=False))
        overlays: list[OverlaySpec] = []
        for ci in chosen:
            ov = _sample_overlay(
                rng,
                bed,
                payloads[ci],
                mix_duration_s=mix_duration_s,
                entry_lo=12.0,
                entry_hi=mix_duration_s * 0.45,
                overlay_lo=24.0,
                overlay_hi=48.0,
            )
            if ov is None:
                break
            overlays.append(ov)
        if len(overlays) != n_overlays:
            continue

        bed_ref_start = float(rng.uniform(0.0, 30.0))
        return MashupScenario(
            mix_id=mix_id,
            bed=bed,
            bed_ref_start_s=bed_ref_start,
            mix_duration_s=mix_duration_s,
            overlays=tuple(overlays),
            curriculum=curriculum,
        )
    return None
