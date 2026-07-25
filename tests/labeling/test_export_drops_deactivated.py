"""GT export must not emit spans for silenced clips.

A clip is silent — and therefore not ground truth — when its track's Activator
is off (``Mixer/Speaker/Manual=false``), the clip itself is deactivated
(``AudioClip/Disabled=true``), or the track fader sits at 0. Audibility is not
the same as clip extent: Live keeps the clip in the arrangement either way, so
the exporter has to check activation, not just read ``CurrentStart/End``.

Regression for the BB11/BB12 worst-span audit (WS0): a deactivated Kungs
"This Girl" track exported a live 79 s GT span and became a top "loss".
"""

from __future__ import annotations

import json

from labeling.als import parse_layer_clips, load_als_xml
from labeling.extract._shared import collect_kept_clip_rows
from tests.labeling.synth_session import LayerSpec, session_als_file

SPECS = (
    LayerSpec("layer-audible"),
    LayerSpec("layer-deactivated", speaker_on=False),
    LayerSpec("layer-clipdisabled", clip_disabled=True),
    LayerSpec("layer-faderzero", volume_manual=0.0),
)


def _set_dir(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"set_id": "synth01", "mix_duration_s": 256.0, "tracks": []})
    )
    return tmp_path


def test_export_drops_silenced_clips_keeps_audible(tmp_path):
    als = session_als_file(tmp_path, layer_specs=SPECS)
    set_dir = _set_dir(tmp_path)

    _set_id, rows, review = collect_kept_clip_rows(als, set_dir)

    kept_tracks = {r.clip.track_name for r in rows}
    assert kept_tracks == {"layer-audible"}, kept_tracks

    dropped = {r.track for r in review if r.action == "dropped"}
    assert {"layer-deactivated", "layer-clipdisabled", "layer-faderzero"} <= dropped


def test_reader_keeps_all_clips_but_flags_silence(tmp_path):
    # The reader stays total (the export drop-count gate depends on it): every
    # non-mix AudioClip is still returned, but silenced ones carry a reason.
    als = session_als_file(tmp_path, layer_specs=SPECS)
    root = load_als_xml(als)
    by_track = {c.track_name: c for c in parse_layer_clips(root)}

    assert set(by_track) == {s.name for s in SPECS}
    assert by_track["layer-audible"].silence_reason == ""
    assert by_track["layer-deactivated"].silence_reason == "track-deactivated"
    assert by_track["layer-clipdisabled"].silence_reason == "clip-disabled"
    assert by_track["layer-faderzero"].silence_reason == "track-fader-zero"
