"""Load exported ground truth into aligner training examples."""
from __future__ import annotations

from pathlib import Path

from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load
from core.result import Err, Ok, Result

from .records import SlotCandidate, SpanTarget


def track_to_target(t: GroundTruthTrack) -> SpanTarget:
    return SpanTarget(
        slot_label=t.slot_label or t.label,
        recording_id=t.track_id,
        claimed_stem=t.claimed_stem or "regular",
        set_start_s=t.set_start_s,
        set_end_s=t.set_end_s,
        ref_start_s=t.ref_start_s,
        ref_end_s=t.ref_end_s,
        tempo_ratio=t.tempo_ratio,
        pitch_shift_semi=t.pitch_shift_semi,
        label=t.label,
    )


def load_set(yaml_path: Path | str) -> Result[tuple[GroundTruthSet, tuple[SpanTarget, ...]], str]:
    match load(yaml_path):
        case Err(e):
            return Err(e.detail)
        case Ok(gt):
            targets = tuple(track_to_target(t) for t in gt.tracks)
            return Ok((gt, targets))


def slot_candidates_from_targets(
    targets: tuple[SpanTarget, ...],
) -> dict[str, tuple[SlotCandidate, ...]]:
    """Naive candidate pool: distinct (recording_id, stem) seen in GT for each slot."""
    by_slot: dict[str, list[SlotCandidate]] = {}
    for t in targets:
        if not t.recording_id:
            continue
        by_slot.setdefault(t.slot_label, [])
        cand = SlotCandidate(recording_id=t.recording_id, claimed_stem=t.claimed_stem)
        if cand not in by_slot[t.slot_label]:
            by_slot[t.slot_label].append(cand)
    return {k: tuple(v) for k, v in by_slot.items()}
