"""Ableton `.als` ↔ structured-data codec.

The `.als` is treated as a bidirectional grammar (source ↔ AST): `read` parses
a session into frozen records, `identity` resolves clip paths to slot / stem /
manifest identity, `write` mutates a session tree (seeding primitives), and the
round-trip law ``parse ∘ print = id`` is the verification pillar (BB12 152/152
clips; see docs/als_codec_subpackage_plan.md).

Uses `lxml` (Py3.14 venv lacks working stdlib expat). Always re-read the
`.als` from disk — never cache a parse across runs.
"""

from __future__ import annotations

from labeling.als.identity import (
    build_manifest_index,
    classify_path,
    display_from_path,
    labels_overlap,
    match_manifest_for_path,
    normalize_stem_value,
    resolve_identity,
    slot_from_path,
)
from labeling.als.models import (
    AudibleSpan,
    ManifestIndex,
    ManifestSlot,
    MixClipSpan,
    ParsedClip,
    WarpMarkers,
)
from labeling.als.cst import dump_als_bytes, load_als_xml, save_als_xml
from labeling.als.read import (
    build_vol_envelopes,
    clip_content_identity,
    clip_original_path,
    parse_layer_clips,
    track_display_name,
    volume_automation_id,
)
from labeling.als.semantics import (
    MUTE_THR,
    ArrangementMapper,
    TempoArrangementMapper,
    audible_from_curve,
    audible_span,
    clip_gain_breakpoints,
    envelope_value,
    parse_master_tempo,
    select_arrangement_mapper,
    split_clip_at_mix_span_edges,
    tempo_beat_to_sec,
    tempo_sec_to_beat,
    tempo_ratio,
)
from labeling.als.tags import strip_user_tags
from labeling.als.write import write_locators, write_tempo_envelope

__all__ = [
    # models
    "AudibleSpan",
    "ManifestIndex",
    "ManifestSlot",
    "MixClipSpan",
    "ParsedClip",
    "WarpMarkers",
    # cst
    "dump_als_bytes",
    "load_als_xml",
    "save_als_xml",
    # read
    "build_vol_envelopes",
    "clip_content_identity",
    "clip_original_path",
    "parse_layer_clips",
    "track_display_name",
    "volume_automation_id",
    # semantics
    "MUTE_THR",
    "ArrangementMapper",
    "TempoArrangementMapper",
    "audible_from_curve",
    "audible_span",
    "clip_gain_breakpoints",
    "envelope_value",
    "parse_master_tempo",
    "select_arrangement_mapper",
    "split_clip_at_mix_span_edges",
    "tempo_beat_to_sec",
    "tempo_sec_to_beat",
    "tempo_ratio",
    # identity
    "build_manifest_index",
    "classify_path",
    "display_from_path",
    "labels_overlap",
    "match_manifest_for_path",
    "normalize_stem_value",
    "resolve_identity",
    "slot_from_path",
    # tags
    "strip_user_tags",
    # write
    "write_locators",
    "write_tempo_envelope",
]
