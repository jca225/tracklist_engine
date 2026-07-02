"""Emit GroundTruthSet with ref_segments, loops, gain curves."""

from __future__ import annotations

from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, RefSegment

from .timeline import (
    AcappellaSpan,
    InstrumentalBlock,
    MashupWindowV2,
    MixSlice,
    RegularSpan,
)


def _slice_to_ref_seg(sl: MixSlice) -> RefSegment:
    return RefSegment(
        mix_start_s=sl.mix_start_s,
        ref_start_s=sl.ref_start_s,
        ref_end_s=sl.ref_end_s,
    )


def _instr_track(block: InstrumentalBlock) -> GroundTruthTrack:
    slices = block.slices
    ref_start = slices[0].ref_start_s
    ref_end = slices[-1].ref_end_s
    segs = tuple(_slice_to_ref_seg(s) for s in slices) if len(slices) > 1 else ()
    return GroundTruthTrack(
        label=block.bed.label,
        track_id=block.bed.recording_id,
        claimed_stem="instrumental",
        set_start_s=block.mix_start_s,
        set_end_s=block.mix_end_s,
        ref_start_s=ref_start,
        ref_end_s=ref_end,
        slot_label=block.slot_label,
        ref_source="synthetic_stem",
        tempo_ratio=1.0,
        pitch_shift_semi=block.pitch_shift_semi,
        ref_segments=segs,
    )


def _acap_track(span: AcappellaSpan) -> GroundTruthTrack:
    segs = tuple(_slice_to_ref_seg(s) for s in span.slices) if span.slices else ()
    return GroundTruthTrack(
        label=span.payload.label,
        track_id=span.payload.recording_id,
        claimed_stem="acappella",
        set_start_s=span.mix_start_s,
        set_end_s=span.mix_end_s,
        ref_start_s=span.ref_start_s,
        ref_end_s=span.ref_end_s,
        slot_label=span.slot_label,
        ref_source="synthetic_stem",
        tempo_ratio=span.tempo_ratio,
        pitch_shift_semi=span.pitch_shift_semi,
        is_loop=span.is_loop,
        ref_segments=segs,
        gain_curve=span.gain_curve,
    )


def _regular_track(span: RegularSpan) -> GroundTruthTrack:
    return GroundTruthTrack(
        label=span.regular.label,
        track_id=span.regular.recording_id,
        claimed_stem="regular",
        set_start_s=span.mix_start_s,
        set_end_s=span.mix_end_s,
        ref_start_s=span.ref_start_s,
        ref_end_s=span.ref_end_s,
        slot_label=span.slot_label,
        ref_source="synthetic_stem",
        tempo_ratio=span.tempo_ratio,
        pitch_shift_semi=span.pitch_shift_semi,
        gain_curve=span.gain_curve,
    )


def window_to_gt(window: MashupWindowV2) -> GroundTruthSet:
    tracks: list[GroundTruthTrack] = []
    for block in window.instrumentals:
        tracks.append(_instr_track(block))
    for ac in window.acappellas:
        tracks.append(_acap_track(ac))
    for reg in window.regulars:
        tracks.append(_regular_track(reg))
    return GroundTruthSet(
        set_id=window.mix_id,
        tracks=tuple(tracks),
        source="synthetic_mix_generator_v2",
        annotated_by=f"curriculum:{window.curriculum}",
    )
