"""Loop tempo_ratio = playback speed from PLAYED segments, not the ref envelope.

Regression for the Emily-instrumental bug: a 3-segment loop that played each
segment at 1.0x (jumping between non-contiguous song sections) was reported at
2.69x because the old code used last_ref_end - first_ref_start, counting the
ref region the DJ jumped over and never played."""
from __future__ import annotations

from labeling.als_io import ParsedClip, WarpMarkers
from labeling.export_als_to_gt import ClipRow, _detect_loops


def _clip(path, arr_start):
    return ParsedClip(group_name="g", track_name="t", path=path,
                      arr_start=arr_start, arr_end=arr_start + 10,
                      loop_start=0.0, loop_end=10.0, pitch_coarse=0, pitch_fine=0,
                      warp=WarpMarkers(points=((0.0, 0.0),)))


def _row(path, arr, set_s, set_e, ref_s, ref_e):
    return ClipRow(
        clip=_clip(path, arr), set_start_s=set_s, set_end_s=set_e,
        ref_start_s=ref_s, ref_end_s=ref_e, recording_id=None, slot_label="003",
        display="Two Friends - Emily (Remix)", claimed_stem="instrumental",
        ref_source="demucs", tempo_ratio=None, pitch_shift_semi=0,
    )


def test_loop_tempo_uses_played_segments_not_envelope():
    # 3 segments, each played at 1.0x, jumping around the song
    rows = [
        _row("/s/emily/instrumental.flac", 0,  64.4,  92.8,  30.8,  59.2),  # 28.4s
        _row("/s/emily/instrumental.flac", 50, 92.9, 123.9, 154.4, 185.4),  # 31.0s
        _row("/s/emily/instrumental.flac", 90, 123.9, 129.8, 200.6, 206.5),  # 5.9s
    ]
    merged = _detect_loops(rows)
    assert len(merged) == 1
    m = merged[0]
    # 3 DISTINCT segments = a SPLIT/CUT, NOT a loop (no bit-identical repeat)
    assert not m.is_loop and len(m.ref_segments) == 3
    # played 65.3s of song over 65.4s of mix => ~1.0x, NOT 175.7/65.4 = 2.69
    assert abs(m.tempo_ratio - 1.0) < 0.05, m.tempo_ratio


def test_loop_requires_back_to_back_not_reprise():
    # same ref section played BACK-TO-BACK (repeat starts where it ended) = loop
    bb = _detect_loops([
        _row("/s/x/v.flac", 0, 10.0, 14.0, 61.0, 65.0),
        _row("/s/x/v.flac", 4, 14.0, 18.0, 61.0, 65.0),
    ])[0]
    assert bb.is_loop
    # same ref section replayed far apart (other content between) = reprise, NOT a
    # loop (Beach Boys "Wouldn't It Be Nice": ending at mix 851 then 882, ~30s gap)
    reprise = _detect_loops([
        _row("/s/x/v.flac", 0,  10.0, 14.0, 61.0, 65.0),
        _row("/s/x/v.flac", 40, 40.0, 44.0, 61.0, 65.0),
    ])[0]
    assert not reprise.is_loop and len(reprise.ref_segments) == 2


def test_loop_is_bit_identical_repeat_not_split():
    # repeated identical segment (ref 153-157 x3) => LOOP
    loop = _detect_loops([
        _row("/s/x/vocals.flac", 0, 79.0, 82.9, 153.2, 157.1),
        _row("/s/x/vocals.flac", 4, 82.9, 86.6, 153.2, 157.1),
        _row("/s/x/vocals.flac", 8, 86.6, 90.3, 153.2, 157.1),
    ])[0]
    assert loop.is_loop
    # distinct sections played in sequence => SPLIT, not a loop
    split = _detect_loops([
        _row("/s/x/vocals.flac", 0, 79.0, 90.0, 10.0, 21.0),
        _row("/s/x/vocals.flac", 4, 90.0, 99.0, 80.0, 89.0),
    ])[0]
    assert not split.is_loop and len(split.ref_segments) == 2


def test_loop_envelope_handles_backward_jump():
    # Avicii Fade: loop ref 153-157 x3, then jump BACK to ref 39.6-65.4.
    # ref_end must not be < ref_start (the old last-end/first-start gave -87.8).
    rows = [
        _row("/s/avicii/vocals.flac", 0,  79.0,  82.9, 153.2, 157.1),
        _row("/s/avicii/vocals.flac", 4,  82.9,  86.6, 153.2, 157.0),
        _row("/s/avicii/vocals.flac", 8,  86.6,  90.3, 153.2, 157.1),
        _row("/s/avicii/vocals.flac", 30, 111.1, 136.7, 39.6, 65.4),
    ]
    m = _detect_loops(rows)[0]
    assert m.ref_start_s == 39.6 and m.ref_end_s == 157.1   # min/max envelope
    assert m.ref_end_s - m.ref_start_s > 0                   # span >= 0, the bug
