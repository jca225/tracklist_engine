"""Training examples and inference helpers for MERT span alignment."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .mert_store import MertSeries
from .records import SlotCandidate, SpanTarget

log = logging.getLogger(__name__)


def candidate_list(
    slot: str,
    pools: dict[str, tuple[SlotCandidate, ...]],
    all_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cands = pools.get(slot, ())
    ids = tuple(c.recording_id for c in cands)
    stems = tuple(c.claimed_stem for c in cands)
    if ids:
        return ids, stems
    return all_ids, tuple("regular" for _ in all_ids)


@dataclass(frozen=True)
class MertSpanExample:
    target: SpanTarget
    candidate_ids: tuple[str, ...]
    candidate_stems: tuple[str, ...]
    positive_idx: int
    mix_window_vectors: np.ndarray   # (T, dim) measures in local search band
    mix_window_mid_s: np.ndarray   # (T,)
    mix_segment: np.ndarray        # (dim,) GT set span pool
    ref_segments: np.ndarray       # (C, dim) GT ref-span pools per candidate
    span_mask: np.ndarray          # (T,) 1 if measure mid in GT set span


def build_examples(
    targets: tuple[SpanTarget, ...],
    mix: MertSeries,
    refs: dict[str, MertSeries],
    slot_pools: dict[str, tuple[SlotCandidate, ...]],
    *,
    search_margin_s: float = 120.0,
    max_negatives: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[MertSpanExample, ...]:
    """Materialize supervised rows for spans with MERT + a candidate pool."""
    gen = rng or np.random.default_rng(0)
    all_ids = tuple(sorted(refs))
    out: list[MertSpanExample] = []
    missing_refs: set[str] = set()

    for t in targets:
        if not t.recording_id or t.recording_id not in refs:
            continue
        pos_ref = refs[t.recording_id]
        pos_seg = pos_ref.pool(t.ref_start_s, t.ref_end_s or t.ref_start_s + 30.0)
        mix_seg = mix.pool(t.set_start_s, t.set_end_s)
        if pos_seg is None or mix_seg is None:
            continue

        cand_ids, cand_stems = candidate_list(t.slot_label, slot_pools, all_ids)
        if t.recording_id not in cand_ids:
            cand_ids = (t.recording_id,) + cand_ids
            cand_stems = (t.claimed_stem,) + cand_stems

        # Cap negatives for stable batch shapes.
        if len(cand_ids) > max_negatives + 1:
            stem_map = dict(zip(cand_ids, cand_stems, strict=True))
            others = [cid for cid in cand_ids if cid != t.recording_id]
            pick = gen.choice(len(others), size=max_negatives, replace=False)
            keep_ids = [t.recording_id] + [others[i] for i in pick]
            order = {cid: i for i, cid in enumerate(cand_ids)}
            keep_ids.sort(key=lambda cid: order[cid])
            cand_ids = tuple(keep_ids)
            cand_stems = tuple(stem_map[cid] for cid in cand_ids)

        positive_idx = cand_ids.index(t.recording_id)
        ref_segments = []
        for cid in cand_ids:
            rs = refs.get(cid)
            if cid == t.recording_id:
                seg = pos_seg
            elif rs is None:
                # Zero vector can never win the identity argmax — a silent
                # handicap; surface it (see MISS slot=039 / 2qy7u05p).
                seg = np.zeros_like(pos_seg)
                missing_refs.add(cid)
            else:
                seg = rs.track_mean()
            ref_segments.append(seg)
        ref_segments_arr = np.stack(ref_segments, axis=0).astype(np.float32)

        lo = max(0.0, t.set_start_s - search_margin_s)
        hi = min(float(mix.end_s[-1]), t.set_end_s + search_margin_s)
        mid = 0.5 * (mix.start_s + mix.end_s)
        win = (mid >= lo) & (mid <= hi)
        if not win.any():
            continue
        mix_window_vectors = mix.vectors[win]
        mix_window_mid = mid[win]
        span_mask = ((mix_window_mid >= t.set_start_s) & (mix_window_mid <= t.set_end_s)).astype(
            np.float32
        )

        out.append(
            MertSpanExample(
                target=t,
                candidate_ids=cand_ids,
                candidate_stems=cand_stems,
                positive_idx=positive_idx,
                mix_window_vectors=mix_window_vectors,
                mix_window_mid_s=mix_window_mid,
                mix_segment=mix_seg,
                ref_segments=ref_segments_arr,
                span_mask=span_mask,
            )
        )
    if missing_refs:
        log.warning(
            "build_examples: %d candidate(s) have no MERT embedding and were "
            "zero-filled (cannot win identity): %s",
            len(missing_refs),
            ", ".join(sorted(missing_refs)),
        )
    return tuple(out)


def median_duration_by_slot(targets: tuple[SpanTarget, ...]) -> dict[str, float]:
    by_slot: dict[str, list[float]] = {}
    for t in targets:
        by_slot.setdefault(t.slot_label.split("w", 1)[0], []).append(t.set_end_s - t.set_start_s)
    return {slot: float(np.median(durs)) for slot, durs in by_slot.items()}


def slide_duration(target: SpanTarget, slot_medians: dict[str, float], fallback: float = 45.0) -> float:
    base = target.slot_label.split("w", 1)[0]
    return slot_medians.get(base, fallback)


def pool_mix_window(mix: MertSeries, start_s: float, end_s: float) -> np.ndarray | None:
    return mix.pool(start_s, end_s)
