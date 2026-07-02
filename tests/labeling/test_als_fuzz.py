"""Mutation fuzz: the codec must be TOTAL on hostile sessions.

Random structural damage to a synthetic session tree — dropped elements,
deleted/corrupted attributes — must never make extraction or validation
raise. Malformed pieces are skipped by the parsers and surfaced by
validate_session; crashing is the only failure mode.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from labeling.als.read import build_vol_envelopes, parse_layer_clips
from labeling.als.semantics import (
    ArrangementMapper,
    TempoArrangementMapper,
    parse_master_tempo,
    select_arrangement_mapper,
)
from labeling.als.validate import validate_session
from tests.labeling.synth_session import session_root

_BAD_VALUES = ("", "nan", "inf", "-inf", "abc", "1e999", "-63072000", "0x10", "🎛")


@settings(deadline=None, max_examples=200)
@given(st.data())
def test_codec_total_under_mutation(data):
    root = session_root(layer_tracks=("layer-a", "layer-b"))
    n_mutations = data.draw(st.integers(1, 4))
    for _ in range(n_mutations):
        els = list(root.iter())
        el = els[data.draw(st.integers(0, len(els) - 1))]
        kind = data.draw(st.sampled_from(("drop", "attr_bad", "attr_del")))
        if kind == "drop":
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
        elif el.attrib:
            key = data.draw(st.sampled_from(sorted(el.attrib)))
            if kind == "attr_bad":
                el.set(key, data.draw(st.sampled_from(_BAD_VALUES)))
            else:
                del el.attrib[key]

    # None of these may raise, whatever the damage:
    validate_session(root)
    parse_layer_clips(root)
    parse_master_tempo(root)
    build_vol_envelopes(root)
    ArrangementMapper.from_mix_track(root, mix_duration_s=100.0)
    TempoArrangementMapper.from_root(root, root, mix_duration_s=100.0)
    select_arrangement_mapper(root, root, mix_duration_s=100.0, label_arr_max=512.0)
