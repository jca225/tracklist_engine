"""Static track-fader gain must reach the exported gain curve.

Live records a track's fader in two places: a static ``Mixer/Volume/Manual``
value, and (when the DJ rode it) a volume AutomationEnvelope that *overrides*
the manual. The reader already captures the envelope; before this fix a track
with NO automation but a non-unity static fader exported as flat unity — the
curve lied about audibility. A track parked at -6 dB read as fully audible, and
one parked between the hard-zero floor (1e-4) and the mute floor (MUTE_THR)
read as audible despite being effectively silent.

These pin the fold-static-gain-into-the-curve behavior end to end.
"""

from __future__ import annotations

import json

from labeling.als import load_als_xml, parse_layer_clips
from labeling.als.semantics import MUTE_THR, clip_gain_breakpoints, linear_to_db
from labeling.export_als_to_gt import collect_kept_clip_rows
from tests.labeling.synth_session import LayerSpec, session_als_file


def _set_dir(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"set_id": "synth01", "mix_duration_s": 256.0, "tracks": []})
    )
    return tmp_path


def test_reader_captures_static_fader_gain(tmp_path):
    als = session_als_file(
        tmp_path, layer_specs=(LayerSpec("half-fader", volume_manual=0.5),)
    )
    clip = parse_layer_clips(load_als_xml(als))[0]
    assert clip.track_gain == 0.5
    assert clip.silence_reason == ""  # -6 dB is quiet, not silent


def test_partial_static_fader_reflected_in_exported_curve(tmp_path):
    # A track statically at 0.5 (~-6 dB) with no automation: the exported gain
    # curve must carry 0.5, not unity — and the clip stays audible (0.5 > mute).
    als = session_als_file(
        tmp_path, layer_specs=(LayerSpec("half-fader", volume_manual=0.5),)
    )
    _sid, rows, _review = collect_kept_clip_rows(als, _set_dir(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row.gain_curve, "static fader must produce a non-empty curve"
    assert all(abs(g - 0.5) < 1e-6 for _, g in row.gain_curve), row.gain_curve
    assert row.skip_training is False


def test_static_fader_below_mute_floor_is_skipped(tmp_path):
    # 0.03 sits above the hard-zero drop (1e-4) but below MUTE_THR (0.05): the
    # old code kept it as audible; folding it into the curve marks it silent.
    assert 1e-4 < 0.03 < MUTE_THR
    als = session_als_file(
        tmp_path, layer_specs=(LayerSpec("near-silent", volume_manual=0.03),)
    )
    _sid, rows, _review = collect_kept_clip_rows(als, _set_dir(tmp_path))
    assert len(rows) == 1
    assert rows[0].skip_training is True
    assert rows[0].audible_frac == 0.0


def test_true_zero_static_fader_still_dropped(tmp_path):
    # A genuinely-zero fader is silence and must not become a GT span at all
    # (unchanged behavior — the hard-drop stays).
    als = session_als_file(
        tmp_path, layer_specs=(LayerSpec("muted", volume_manual=0.0),)
    )
    _sid, rows, review = collect_kept_clip_rows(als, _set_dir(tmp_path))
    assert rows == []
    assert any(r.action == "dropped" and r.reason == "track-fader-zero" for r in review)


def test_clip_gain_breakpoints_honors_base_gain():
    # No automation -> flat curve at the static base_gain, not unity.
    curve = clip_gain_breakpoints((), 10.0, 20.0, base_gain=0.5)
    assert curve == [(10.0, 0.5), (20.0, 0.5)]
    # Default base_gain is unity (back-compat).
    assert clip_gain_breakpoints((), 10.0, 20.0) == [(10.0, 1.0), (20.0, 1.0)]
    # With automation present, the envelope wins and base_gain is ignored.
    pts = ((10.0, 1.0), (20.0, 0.0))
    curve = clip_gain_breakpoints(pts, 10.0, 20.0, base_gain=0.5)
    assert curve[0] == (10.0, 1.0) and curve[-1] == (20.0, 0.0)


def test_linear_to_db():
    assert linear_to_db(1.0) == 0.0
    assert abs(linear_to_db(0.5) - (-6.0206)) < 1e-3
    assert abs(linear_to_db(2.0) - 6.0206) < 1e-3
    # A true mute is -inf dB, not a crash.
    assert linear_to_db(0.0) == float("-inf")
