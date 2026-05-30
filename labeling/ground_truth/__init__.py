"""Hand-annotated ground-truth fixtures — the DSL.

Canonical home for reading, writing, and validating ground-truth labels
(`tests/fixtures/*_ground_truth.yaml`). This is the schema the manual
labeling workflow produces and the future algorithmic aligner will train
against — the seam between `labeling/` and `alignment/`. One schema owner.

Schema (v2, 2026-04-22):

  set_id:         canonical set_id from `dj_sets`
  source:         typically 'ableton_session'
  annotated_by:   free-form
  tracks: [
    track:          human-readable label
    track_id:       1001tracklists data-trackid (DB FK). Optional for
                    DJ-added tracks missing from the tracklist.
    claimed_stem:   regular | acappella | instrumental
    set_start_s:    mix-side start, seconds
    set_end_s:      mix-side end, seconds
    ref_start_s:    MANDATORY — seconds into the reference where the
                    DJ first dropped in.
    ref_end_s:      optional companion to ref_start_s
    is_loop:        optional, default False. True requires ref_segments.
    ref_segments:   [{ref_start_s, ref_end_s, mix_start_s}, ...] per
                    loop iteration / cutup slice.
    media_links:    {youtube|spotify|soundcloud|other: url}
  ]
"""
from .schema import (
    GroundTruthError,
    GroundTruthSet,
    GroundTruthTrack,
    MediaLinks,
    RefSegment,
    dump,
    load,
    save,
)

__all__ = [
    "GroundTruthError",
    "GroundTruthSet",
    "GroundTruthTrack",
    "MediaLinks",
    "RefSegment",
    "dump",
    "load",
    "save",
]
