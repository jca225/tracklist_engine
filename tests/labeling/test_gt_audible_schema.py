"""GT schema round-trip for volume-aware export fields."""
from __future__ import annotations

from labeling.export_als_to_gt import _placeholder_note
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, dump, load
from core.result import Ok


def test_audible_fields_round_trip(tmp_path):
    gt_track = GroundTruthTrack(
        label="Artist - Title",
        track_id="abc123",
        claimed_stem="acappella",
        set_start_s=100.0,
        set_end_s=120.0,
        ref_start_s=5.0,
        ref_end_s=25.0,
        slot_label="130",
        audible_frac=0.65,
        audible_start_s=105.0,
        audible_end_s=118.0,
        skip_training=False,
    )
    text = dump(GroundTruthSet(set_id="testset", tracks=(gt_track,)))
    path = tmp_path / "gt.yaml"
    path.write_text(text)
    match load(path):
        case Ok(gt):
            t = gt.tracks[0]
            assert t.audible_frac == 0.65
            assert t.audible_start_s == 105.0
            assert t.audible_end_s == 118.0
            assert t.skip_training is False
        case _:
            raise AssertionError("expected Ok")


def test_unalignable_placeholder_detection():
    """The mix's own audio used as a clip = the human's UNALIGNABLE marker
    (too hard / source absent). Must be detected; real songs must NOT be."""
    assert _placeholder_note("/s/mix.m4a", "g")            # mix self-reference
    assert _placeholder_note("/s/mix.flac", "g")
    note = _placeholder_note("/s/mix_instrumental.flac", "Lux x Spaceman")
    assert note and "unavailable" in note                  # Lux Omega outsourced-host case
    # real placements are NOT placeholders — incl. an imported instrumental-N.flac,
    # which is a real instrumental the human dragged in (Mako at lane 202), not the
    # set's own mix audio
    assert _placeholder_note("/s/Imported/instrumental-2.flac", "g") is None
    assert _placeholder_note("/s/tracks/154__Honest (Virtu Remix).m4a", "g") is None
    assert _placeholder_note("/s/stems/053__Honest/vocals.flac", "g") is None
    assert _placeholder_note("/s/stems/072__X/instrumental.flac", "g") is None  # real demucs stem


def test_unalignable_round_trip(tmp_path):
    """unalignable + source_note survive save/load. It's a positive abstain
    LABEL, NOT skip_training — the row stays a training example."""
    t = GroundTruthTrack(
        label="mix_instrumental", track_id=None, claimed_stem="instrumental",
        set_start_s=2636.5, set_end_s=2688.8, ref_start_s=0.0,
        slot_label="", unalignable=True, skip_training=False,
        source_note="Lux Omega — original unavailable",
    )
    path = tmp_path / "gt.yaml"
    path.write_text(dump(GroundTruthSet(set_id="s", tracks=(t,))))
    match load(path):
        case Ok(gt):
            r = gt.tracks[0]
            assert r.unalignable is True
            assert r.skip_training is False            # label, not mask
            assert r.source_note == "Lux Omega — original unavailable"
        case _:
            raise AssertionError("expected Ok")
