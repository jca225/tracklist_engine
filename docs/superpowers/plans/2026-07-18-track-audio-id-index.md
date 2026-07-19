# Track-audio-id local index — implementation plan

## Goal

Ship `audio_index.json` keyed by `track_audio_id` so aligner resolvers prefer
stable ids over slot-string globs (follow-on to PR #21).

## Steps

1. Add `labeling/audio_index.py` (build / load / lookup / refresh CLI).
2. Write the index at the end of `pull_set_for_alignment.py`.
3. Consult the index in `stem_resolve.resolve_stem` and `infer_fused._resolve_ref`
   after the manifest path check and before the unique-glob fallback.
4. Tests: build behavior, ambiguous omission, resolver index hit, refresh roundtrip.
5. Spec: `docs/superpowers/specs/2026-07-18-track-audio-id-index-design.md`.

## Done when

- `pytest tests/test_audio_index.py tests/test_fail_closed_audio_resolvers.py` green
- Pull writes `audio_index.json` beside `manifest.json`
- Resolvers can recover a path via `track_audio_id` even when slot globs are ambiguous
