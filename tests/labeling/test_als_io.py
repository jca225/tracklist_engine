"""Tests for Ableton → GT helper functions."""
from __future__ import annotations

from pathlib import Path
from labeling.als_io import (
    ArrangementMapper,
    ManifestIndex,
    ManifestSlot,
    ParsedClip,
    WarpMarkers,
    build_manifest_index,
    classify_path,
    display_from_path,
    labels_overlap,
    match_manifest_for_path,
    resolve_identity,
    slot_from_path,
    split_clip_at_mix_span_edges,
    tempo_ratio,
)


def test_warp_interpolation():
    wm = WarpMarkers(points=((0.0, 0.0), (100.0, 50.0)))
    assert wm.beat_to_sec(0.0) == 0.0
    assert wm.beat_to_sec(100.0) == 50.0
    assert wm.beat_to_sec(50.0) == 25.0


def test_slot_from_path():
    p = "/Users/me/aligning/set/tracks/154w1__Artist - Title [100bpm 5B].m4a"
    assert slot_from_path(p) == "154w1"
    assert slot_from_path("stems/048__Foo/candidates/vocals/cand1__Bar.m4a") == "048"


def test_classify_path():
    assert classify_path("/aligning/set/tracks/001__A - B.m4a") == ("regular", "reference")
    assert classify_path("/aligning/set/stems/001__A/vocals.flac") == ("acappella", "demucs")
    assert classify_path("/aligning/set/stems/001__A/candidates/vocals/x.m4a") == (
        "acappella",
        "online_candidate",
    )
    assert classify_path(
        "/aligning/set/stems/002__Post Malone/candidates/cand1__Post Malone - X.m4a"
    ) == ("acappella", "online_candidate")


def test_display_from_path_and_labels_overlap():
    p = "/aligning/set/stems/002__Post Malone/candidates/cand1__Post Malone - Congratulations.m4a"
    assert display_from_path(p) == "Post Malone - Congratulations"
    assert labels_overlap("Manse - Freeze Time", "Post Malone - Congratulations") is False
    assert labels_overlap("Nelly Furtado - Say It Right", "Say It Right (Studio acapella)") is True


def test_resolve_identity_prefers_path_over_slot_collision(tmp_path: Path):
    set_dir = tmp_path / "set"
    tracks = set_dir / "tracks"
    stems = set_dir / "stems" / "002__Post Malone - Congratulations (Acappella)"
    cand = stems / "candidates"
    tracks.mkdir(parents=True)
    cand.mkdir(parents=True)
    manse = tracks / "002__Manse - Freeze Time.m4a"
    manse.write_bytes(b"x")
    post = cand / "cand1__Post Malone - Congratulations.m4a"
    post.write_bytes(b"x")
    manifest = {
        "set_id": "testset",
        "tracks": [
            {
                "label": "002",
                "track_id": "mtck04x",
                "artist": "Manse",
                "title": "Freeze Time",
                "local_path": str(manse),
            },
            {
                "label": "001w1",
                "track_id": "281u6p4x",
                "artist": "Post Malone",
                "title": "Congratulations",
                "version_tag": "Acappella",
                "local_path": str(tracks / "001w1__Post Malone - Congratulations.m4a"),
            },
        ],
    }
    (set_dir / "manifest.json").write_text(__import__("json").dumps(manifest))
    index = build_manifest_index(set_dir / "manifest.json")
    clip = ParsedClip(
        group_name="g",
        track_name="t",
        path=str(post),
        arr_start=0.0,
        arr_end=10.0,
        loop_start=0.0,
        loop_end=10.0,
        pitch_coarse=0,
        pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (100.0, 50.0))),
    )
    track_id, slot, display, stem = resolve_identity(clip, index)
    assert display == "Post Malone - Congratulations"
    assert track_id is None  # different stems folder than manifest pull path
    assert slot == "002"
    assert stem == "acappella"
    assert match_manifest_for_path(str(manse), index) is not None
    assert match_manifest_for_path(str(manse), index).track_id == "mtck04x"


def test_tempo_ratio():
    assert tempo_ratio(10.0, 12.0) == 1.2
    assert tempo_ratio(0.0, 1.0) is None


def test_split_clip_at_mix_span_edges():
    clip = ParsedClip(
        group_name="g",
        track_name="t",
        path="/stems/121__A/instrumental.flac",
        arr_start=5995.0,
        arr_end=6067.0,
        loop_start=0.0,
        loop_end=72.0,
        pitch_coarse=0,
        pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (100.0, 50.0))),
    )
    spans = (
        type("S", (), {
            "arr_start": 3350.0,
            "arr_end": 6000.0,
            "arr_to_set_sec": lambda self, arr: 2846.0 + (arr - 5995.0) * 0.02,
        })(),
        type("S", (), {
            "arr_start": 6000.0,
            "arr_end": 7735.0,
            "arr_to_set_sec": lambda self, arr: 2509.0 + (arr - 6000.0) * 0.4,
        })(),
    )
    mapper = ArrangementMapper(spans=spans, mix_duration_s=4000.0)  # type: ignore[arg-type]
    parts = split_clip_at_mix_span_edges(clip, mapper)
    assert len(parts) == 2
    assert parts[0].arr_end < 6000.02
    assert parts[1].arr_start >= 6000.0


def test_arrangement_mapper_gap_bridge():
    spans = (
        type("S", (), {
            "arr_start": 0.0,
            "arr_end": 100.0,
            "arr_to_set_sec": lambda self, arr: arr,
        })(),
        type("S", (), {
            "arr_start": 110.0,
            "arr_end": 200.0,
            "arr_to_set_sec": lambda self, arr: arr + 10.0,
        })(),
    )
    mapper = ArrangementMapper(spans=spans, mix_duration_s=200.0)  # type: ignore[arg-type]
    assert mapper.arr_to_set_sec(50.0) == 50.0
    assert mapper.arr_to_set_sec(105.0) == 110.0
