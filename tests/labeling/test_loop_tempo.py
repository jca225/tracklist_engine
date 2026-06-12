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
    assert m.is_loop and len(m.ref_segments) == 3
    # played 65.3s of song over 65.4s of mix => ~1.0x, NOT 175.7/65.4 = 2.69
    assert abs(m.tempo_ratio - 1.0) < 0.05, m.tempo_ratio
