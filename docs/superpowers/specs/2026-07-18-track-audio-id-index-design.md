# Track-audio-id local index design

**Goal:** resolve pulled-set audio by stable `track_audio_id` instead of
slot-string globs, completing the fail-closed work from PR #21.

## Scope

Add `audio_index.json` beside each aligning-folder `manifest.json`, keyed by
`track_audio_id`. Resolvers consult the index before any slot-prefix fallback.

In scope:

- `labeling/audio_index.py` — build / load / lookup helpers + refresh CLI
- write the index at the end of `pull_set_for_alignment.py`
- `stem_resolve.resolve_stem` and `infer_fused._resolve_ref` prefer the index
- tests for build + resolver lookup

Out of scope:

- rewriting annotator-tagged M4A comments
- mashup_demo bench join (already shipped)
- changing inventory / baby-rule acquisition

## Index shape

```json
{
  "version": 1,
  "by_track_audio_id": {
    "20925": {
      "local_path": "/…/tracks/004w1__….m4a",
      "stems": {
        "vocals": "/…/stems/004w1__…/vocals.flac",
        "instrumental": "/…/stems/004w1__…/instrumental.flac"
      }
    }
  }
}
```

Only paths that exist on disk are recorded. Ambiguous slot globs are omitted
(fail closed), matching PR #21.

## Resolution order

`resolve_stem`:

1. Manifest `track["stems"][name]` if the file exists
2. Index lookup by `track_audio_id`
3. Unique slot-glob fallback (existing PR #21 behavior)
4. Else abstain

`_resolve_ref`:

1. Manifest `local_path` if the file exists
2. Index lookup by `track_audio_id`
3. Unique slot-glob fallback
4. Else abstain

## Refresh

`python -m labeling.audio_index <set_dir>` rebuilds from the current
manifest plus unique on-disk fallbacks, so post-pull re-stems and unambiguous
renames can be re-indexed without a full re-pull.
