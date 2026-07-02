"""Emit GroundTruthSet labels for synthetic mashups."""

from __future__ import annotations

from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack

from .scenario import MashupScenario


def scenario_to_gt(scenario: MashupScenario) -> GroundTruthSet:
    tracks: list[GroundTruthTrack] = []
    slot = 100

    bed = scenario.bed
    tracks.append(
        GroundTruthTrack(
            label=bed.label,
            track_id=bed.recording_id,
            claimed_stem="instrumental",
            set_start_s=0.0,
            set_end_s=scenario.mix_duration_s,
            ref_start_s=scenario.bed_ref_start_s,
            ref_end_s=scenario.bed_ref_start_s + scenario.mix_duration_s,
            slot_label=str(slot),
            ref_source="synthetic_stem",
            tempo_ratio=1.0,
            pitch_shift_semi=0,
        )
    )

    for i, ov in enumerate(scenario.overlays):
        tracks.append(
            GroundTruthTrack(
                label=ov.payload.label,
                track_id=ov.payload.recording_id,
                claimed_stem="acappella",
                set_start_s=ov.set_start_s,
                set_end_s=ov.set_end_s,
                ref_start_s=ov.ref_start_s,
                ref_end_s=ov.ref_start_s
                + (ov.set_end_s - ov.set_start_s) * ov.tempo_ratio,
                slot_label=f"{slot}w{i + 1}",
                ref_source="synthetic_stem",
                tempo_ratio=ov.tempo_ratio,
                pitch_shift_semi=ov.pitch_shift_semi,
            )
        )

    return GroundTruthSet(
        set_id=scenario.mix_id,
        tracks=tuple(tracks),
        source="synthetic_mix_generator",
        annotated_by=f"curriculum:{scenario.curriculum}",
    )
