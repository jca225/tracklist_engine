"""Tests for Ableton → GT helper functions."""
from __future__ import annotations

from pathlib import Path

from lxml import etree

from labeling.als import (
    ArrangementMapper,
    ManifestIndex,
    ManifestSlot,
    ParsedClip,
    WarpMarkers,
    audible_span,
    build_manifest_index,
    classify_path,
    display_from_path,
    envelope_value,
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
    assert classify_path("/aligning/set/stems/001__A/instrumental.flac") == (
        "instrumental",
        "demucs",
    )
    assert classify_path("/aligning/set/stems/001__A/candidates/vocals/x.m4a") == (
        "acappella",
        "online_candidate",
    )
    assert classify_path(
        "/aligning/set/stems/002__Post Malone/candidates/cand1__Post Malone - X.m4a"
    ) == ("acappella", "online_candidate")
    assert classify_path(
        "/aligning/set/stems/002__X/candidates/instrumental/cand1__X (Instrumental).m4a"
    ) == ("instrumental", "online_candidate")


def test_classify_path_tracks_master_stem_marker():
    """REGRESSION: a master in tracks/ with an explicit stem qualifier must be
    classified by the qualifier, NOT blindly 'regular'. The old code returned
    'regular' for everything under /tracks/ before reading the filename, which
    dropped the stem of 45 BB12 GT rows (e.g. the real Bad Day (Acappella)).
    The .als clip's referenced FILE is the canonical stem oracle."""
    # the exact bug: a downloaded acappella master living in tracks/
    assert classify_path(
        "/aligning/set/tracks/127__Daniel Powter - Bad Day (Acappella).m4a"
    ) == ("acappella", "reference")
    assert classify_path(
        "/aligning/set/tracks/002__Manse ft. Alice Berg - Freeze Time (Instrumental Mix).m4a"
    ) == ("instrumental", "reference")
    # single-p spelling seen in candidate filenames
    assert classify_path("/aligning/set/tracks/x - y (Acapella).m4a") == (
        "acappella",
        "reference",
    )
    # version qualifiers must NOT flip the stem — these stay regular
    for ver in ("(Rework)", "(Remix)", "(AltVersion)", "(Two Friends Remix)"):
        assert classify_path(f"/aligning/set/tracks/145__Don Diablo - Momentum {ver}.m4a") == (
            "regular",
            "reference",
        ), ver
    # demucs stem inside an "(Acappella)"-named folder is decided by the FILE,
    # not the folder: instrumental.flac there is the instrumental stem
    assert classify_path(
        "/aligning/set/stems/104__Mako - Smoke Filled Room (Acappella)/instrumental.flac"
    ) == ("instrumental", "demucs")


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


def test_full_length_unwarped_sparse_mix_uses_master_tempo():
    """Coverage alone must not make sentinel warp markers beat a real tempo map."""
    from labeling.als.semantics import (
        TempoArrangementMapper,
        select_arrangement_mapper,
    )
    from tests.labeling.synth_session import session_root

    root = session_root(
        tempo_events=((0.0, 120.0), (64.0, 120.0), (64.0, 90.0)),
        mix_warp_points=((0.0, 0.0), (0.03125, 0.015625)),
    )
    mix_track = root.xpath(".//LiveSet/Tracks/AudioTrack")[0]
    mix_track.xpath(".//AudioClip")[0].append(etree.Element("IsWarped", Value="false"))
    mapper = select_arrangement_mapper(
        root, mix_track, mix_duration_s=300.0, label_arr_max=256.0
    )
    assert isinstance(mapper, TempoArrangementMapper)
    assert abs(mapper.arr_to_set_sec(64.0) - 32.0) < 1e-9


def test_envelope_value_interpolates():
    pts = ((0.0, 1.0), (10.0, 0.0))
    assert envelope_value(pts, 0.0) == 1.0
    assert envelope_value(pts, 10.0) == 0.0
    assert envelope_value(pts, 5.0) == 0.5


def test_audible_span_full_and_muted():
    full = audible_span((), 0.0, 10.0)
    assert full.fraction == 1.0
    assert full.arr_start == 0.0
    assert full.arr_end == 10.0

    muted = audible_span(((0.0, 0.0), (10.0, 0.0)), 0.0, 10.0)
    assert muted.fraction == 0.0

    fade = audible_span(((0.0, 0.0), (5.0, 0.0), (10.0, 1.0)), 0.0, 10.0)
    assert 0.0 < fade.fraction < 1.0
    assert fade.arr_start >= 0.0
    assert fade.arr_end <= 10.0


def test_beat_to_sec_extrapolates_past_duplicated_markers():
    """Clustered/duplicated warp markers (Aftershock: 2 pairs at beats 0 &
    0.03125) must EXTRAPOLATE on the real slope, not clamp to the last sec
    (which gave a zero-span ref). 0.022s over 0.03125 beats -> ~0.7 s/beat."""
    from labeling.als import WarpMarkers
    w = WarpMarkers(points=((0.0, 0.366), (0.0, 0.366), (0.03125, 0.388), (0.03125, 0.388)))
    assert w.beat_to_sec(3.475) > 2.0          # was clamped to 0.388
    assert w.beat_to_sec(93.5) > 60.0          # was clamped to 0.388
    assert w.beat_to_sec(93.5) > w.beat_to_sec(3.475)


def test_match_manifest_tag_insensitive(tmp_path):
    """REGRESSION: annotator [NNNbpm KK] renames must not break manifest
    matching. BB11's export produced track_id=None on all 127 rows because the
    clip paths carried tags while manifest.json keeps canonical names — the
    matcher compared them verbatim (found 2026-07-02)."""
    set_dir = tmp_path / "set"
    tracks = set_dir / "tracks"
    tracks.mkdir(parents=True)
    canonical = tracks / "030__Going Deeper - Little Big Adventure.m4a"
    manifest = {
        "set_id": "testset",
        "tracks": [
            {
                "label": "030",
                "track_id": "gd030x",
                "artist": "Going Deeper",
                "title": "Little Big Adventure",
                "local_path": str(canonical),
            },
        ],
    }
    (set_dir / "manifest.json").write_text(__import__("json").dumps(manifest))
    index = build_manifest_index(set_dir / "manifest.json")

    # tagged master file in tracks/
    tagged = str(tracks / "030__Going Deeper - Little Big Adventure [126bpm 8B].m4a")
    row = match_manifest_for_path(tagged, index)
    assert row is not None and row.track_id == "gd030x"

    # demucs stem inside a tagged stems/ subdir
    tagged_stem = str(
        set_dir / "stems" / "030__Going Deeper - Little Big Adventure [126bpm 8B]" / "vocals.flac"
    )
    row = match_manifest_for_path(tagged_stem, index)
    assert row is not None and row.track_id == "gd030x"

    # un-tagged exact match still works
    row = match_manifest_for_path(str(canonical), index)
    assert row is not None and row.track_id == "gd030x"

    # a genuinely different song must NOT match tag-stripped
    other = str(tracks / "031__Other Artist - Other Song [126bpm 8B].m4a")
    assert match_manifest_for_path(other, index) is None


def _clip(**overrides) -> ParsedClip:
    base = dict(
        group_name="g",
        track_name="t",
        path="/aligning/set/tracks/001__A - B.m4a",
        arr_start=5066.0,
        arr_end=5120.0,
        loop_start=59.053,
        loop_end=95.053,
        pitch_coarse=0,
        pitch_fine=0,
        # sentinel-pair markers as Live writes on never-hand-warped clips
        warp=WarpMarkers(points=((0.0, 0.229), (0.03125, 0.244))),
    )
    base.update(overrides)
    return ParsedClip(**base)


def test_unwarped_ref_offsets_are_loop_seconds():
    # BB12 Faded shape (found 2026-07-02): unwarped clips store Loop values in
    # SECONDS; pushing them through the sentinel warp map scaled them ~0.48x
    # and exported ref offsets 39-112 s wrong.
    clip = _clip(is_warped=False)
    assert clip.ref_start_s() == 59.053
    assert clip.ref_end_s() == 95.053


def test_warped_ref_offsets_still_use_warp_map():
    clip = _clip(
        is_warped=True,
        loop_start=16.0,
        warp=WarpMarkers(points=((0.0, 0.0), (100.0, 50.0))),
    )
    assert clip.ref_start_s() == 8.0  # 16 beats * 0.5 s/beat
    assert clip.ref_end_s() == 35.0  # (16 + 54 arr beats) * 0.5


def test_is_warped_defaults_true():
    assert _clip().is_warped is True


def test_unwarped_trim_uses_mix_seconds():
    # One mix span at 0.5 s/beat: trimming 10 arrangement beats off the left
    # edge must advance an unwarped clip's loop values by 5 SECONDS (elapsed
    # mix time), not 10 (beats).
    span = type(
        "S",
        (),
        {
            "arr_start": 5000.0,
            "arr_end": 5200.0,
            "arr_to_set_sec": lambda self, arr: 2000.0 + (arr - 5000.0) * 0.5,
        },
    )()
    mapper = ArrangementMapper(spans=(span,), mix_duration_s=4000.0)  # type: ignore[arg-type]
    clip = _clip(is_warped=False)
    from labeling.als.semantics import _trim_to_interval

    trimmed = _trim_to_interval(clip, mapper, 5076.0, 5120.0)
    assert trimmed.loop_start == 59.053 + 5.0
    assert trimmed.loop_end == 59.053 + 5.0 + 22.0  # 44 beats * 0.5 s
    # warped clip: same trim advances loop_start by BEATS
    warped = _clip(is_warped=True)
    trimmed_w = _trim_to_interval(warped, mapper, 5076.0, 5120.0)
    assert trimmed_w.loop_start == 59.053 + 10.0
    # untouched interval returns the clip unchanged
    assert _trim_to_interval(clip, mapper, 5066.0, 5120.0) is clip
