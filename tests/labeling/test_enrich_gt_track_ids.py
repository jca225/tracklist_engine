"""Tests for GT track_id enrichment."""
from __future__ import annotations

from dataclasses import replace

from labeling.als_io import ManifestIndex, ManifestSlot
from labeling.enrich_gt_track_ids import (
    SlotRow,
    enrich_track,
    lookup_db_label,
)
from labeling.export_als_to_gt import ClipRow
from labeling.als_io import ParsedClip, WarpMarkers
from labeling.ground_truth.schema import GroundTruthTrack


def _track(**kwargs) -> GroundTruthTrack:
    defaults = dict(
        label="Post Malone - Congratulations  Acapella",
        track_id=None,
        claimed_stem="acappella",
        set_start_s=64.4,
        set_end_s=94.4,
        ref_start_s=0.0,
        slot_label="002",
        ref_source="online_candidate",
    )
    defaults.update(kwargs)
    return GroundTruthTrack(**defaults)


def _clip(path: str) -> ClipRow:
    clip = ParsedClip(
        group_name="g",
        track_name="t",
        path=path,
        arr_start=0.0,
        arr_end=10.0,
        loop_start=0.0,
        loop_end=10.0,
        pitch_coarse=0,
        pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (100.0, 50.0))),
    )
    return ClipRow(
        clip=clip,
        set_start_s=64.4,
        set_end_s=94.4,
        ref_start_s=0.0,
        ref_end_s=30.0,
        recording_id=None,
        slot_label="002",
        display="Post Malone - Congratulations  Acapella",
        claimed_stem="acappella",
        ref_source="online_candidate",
        tempo_ratio=1.0,
        pitch_shift_semi=0,
    )


def test_lookup_db_label_unique():
    slots = (
        SlotRow("281u6p4x", "acappella", "Post Malone - Congratulations (Acappella)"),
        SlotRow("mtck04x", "regular", "Manse - Freeze Time"),
    )
    assert lookup_db_label(
        "Post Malone - Congratulations  Acapella",
        "acappella",
        slots,
    ) == "281u6p4x"


def test_enrich_track_db_label_when_no_manifest_path_match():
    manifest = ManifestIndex(by_slot={}, by_path={}, rows=())
    slots = (
        SlotRow("281u6p4x", "acappella", "Post Malone - Congratulations (Acappella)"),
    )
    path = "/set/stems/002__Post Malone/candidates/cand1__Post Malone - Congratulations.m4a"
    res = enrich_track(_track(), clip=_clip(path), manifest=manifest, slots=slots)
    assert res.source == "db_label"
    assert res.track_id == "281u6p4x"
    assert res.track.label.startswith("Post Malone")
