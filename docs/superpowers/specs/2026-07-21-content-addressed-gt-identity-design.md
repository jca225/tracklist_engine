# Content-addressed GT identity binding (Operation Crush — de-poisoning)

**Date:** 2026-07-21
**Status:** design, pending implementation plan
**Tracking:** #40, #47, #51, #63; handoff `docs/archive/crush_handoff_depoison_20260721.md`
**Crush exit gate:** BB11+BB12 pass L0–L4, pi read-back exact, one clean SSOT re-measure.

## Problem

The GT exporter (`labeling/export_als_to_gt.py`) binds each `.als` clip to a
`recording_id`. When a clip's path does not match the local `manifest.json`, it
falls back to `slot_id_map` — a frozen `slot_label → recording_id` fixture
(`labeling/fixtures/id_maps/{set}_slots.json`). This is the **poison**: it stamps
a confidently-wrong cross-song id onto GT rows.

### Verified state (BB12 `1fsnxchk`, `main` @ 5b9f05d)

- **292 of 301 clips** currently take their `recording_id` from `slot_id_map`, not
  a path match. It is not "3 poison ids" — it is nearly the whole set.
- Confirmed cross-song poison: slot **028** Beatles→`2p25k23p` (also on slot 012 →
  the #63 collision), **031** CCR→`1q8nc02p`, **144** Snakehips→`2uq9800f`.
- **Root cause is a coordinate-system mismatch.** The `.als` numbers slots by
  *placement order* (002–155, 146 slots). pi/manifest number by *tracklist row*
  (001–042 + w-suffixes, 165 rows). They are different coordinate systems, so
  **slot number is not a valid identity key across the two artifacts.** That is why
  `slot_id_map` was invented and why it poisons on any renumber.
- The local `manifest.json` is **truncated to slots 001–042**, so it cannot be the
  mapping source for the full set. The authoritative, complete, current mapping
  lives on **pi-storage**.
- **Content is the only invariant that bridges the two coordinate systems.**

### Feasibility (measured)

- `track_audio.sha256` (full-file, chunked — `ingest/adapters/downloader.py::_sha256`)
  is **fully populated** for this set: regular 205/205, acappella 97/97,
  instrumental 16/16.
- Local clip files: **301/301 exist on disk.**
- Bind-by-`sha256` coverage of local files vs pi `track_audio.sha256`, by kind:
  candidate **84/84**, demucs **1/34**, master **3/42**, other 0/4.
  - Candidates (the poison-prone acappella/instrumental downloads) bind cleanly.
  - Demucs stems are not in `track_audio` → covered by pull-time stem hashing.
  - **Masters bind at 7%**: `tag_aligning_folder.py` writes BPM/key into the m4a
    iTunes atoms, mutating file bytes so local sha256 ≠ pi sha256.
- **Tag-mutation is container-only.** Proven: local tagged master
  `154 Chainsmokers - Honest (Virtu Remix)` mdat-payload sha256 `fe374e…` **exactly
  equals** pi canonical `2vmxu50p` mdat sha256, while full-file sha256 differs. The
  audio payload (mp4 `mdat`) is tag-invariant and disambiguates precisely (6
  "Honest" masters on pi; mdat selects the one correct recording).

## Approach

Bind every clip to identity **by content**, against a catalog built from pi. On a
content miss, **abstain** (null id, flagged) — never guess. This kills the poison
class by construction (`labeling/content_resolver.py` is the landed, no-fallback
resolver; this design wires it and sources its catalog).

Two content keys per artifact:
- `content_sha256` — full-file sha256. Free from `track_audio`; binds candidates,
  demucs stems, and untagged masters.
- `payload_sha256` — sha256 of the mp4 `mdat` box. Tag-invariant; binds
  locally-tagged m4a masters. Built only for m4a (the only tag-mutated kind); no
  FLAC payload hasher is needed.

## Components

### A. Pull-time content-catalog sidecar
`labeling/build_content_catalog.py`, invoked by `pull_set_for_alignment.py`.
- Query pi `track_audio` for every recording in the set (`set_track_slots.recording_id`):
  emit `{content_sha256, payload_sha256, recording_id, track_audio_id, stem}`.
  `payload_sha256` computed on pi for m4a rows (read the `mdat` box, no decode).
- Hash demucs stem files on pi (`track_stems` joined to `track_audio` for the set's
  recordings): `content_sha256` of each stem file, recording via
  `track_audio_id → recording_id`, stem = `acappella` (vocals) / `instrumental`.
- Write `content_catalog.json` into the set-dir (sibling of `manifest.json`).
- **Requires re-pulling BB12** to refresh the truncated manifest + emit the sidecar.
  The build needs pi's DB + stem files only; no audio re-sync required if the folder
  is already populated. Re-pull must not disturb the annotator's tagged files or the
  `.als` (per `labeling/CLAUDE.md` consistency model; `--prune` NOT used). Snapshot
  `set_ground_truth` + the yaml before any write-back (#51; backup tag
  `wip/bb12-enrichment-backup`).

### B. Resolver wiring in the export (offline)
`export_als_to_gt`:
- Load `content_catalog.json` → `ContentCatalog.from_entries(...)`.
- Catalog registration uses the resolver's existing `by_head_hash` map: each entry
  is keyed by its `content_sha256` **and** (for m4a) its `payload_sha256`, both
  pointing at the same `CatalogEntry`. No resolver change is required.
- In `_clip_row`, bind in two passes against the landed resolver: pass 1
  `resolve_clip_identity(clip, catalog, head_hash_of=<full-file sha256>)`; on `Err`
  for a `.m4a` path, pass 2 with `head_hash_of=<mdat payload sha256>`. (The resolver's
  `by_size_crc` fast-path is unused — Ableton's CRC is not in the catalog; content
  keys are our own sha256s.)
- `Ok` → `recording_id` + `id_source="content"`. `Err` → null id +
  `id_source="abstain"`, diagnostic recorded, row still emitted (abstain is a
  positive label). `slot_id_map` and its loader are deleted.
- `claimed_stem`/display keep coming from `classify_path` (the `.als`-declared stem);
  the catalog supplies `recording_id` + `id_source` only.

The mp4 `mdat` hasher is the single new hashing primitive; it lives in a shared
module (e.g. `labeling/content_hash.py`) imported by both A (pi side) and B (export
side) so both compute identical digests.

### C. `id_source` on `GroundTruthTrack`
`labeling/ground_truth/schema.py`: add `id_source: str` to `GroundTruthTrack`
(values `content | abstain`; default `""` for legacy fixtures). Serialize in `dump`,
parse in `_parse_track`. (Handoff says "add to `RefSegment`"; that is a per-row
attribute — it belongs on `GroundTruthTrack`, not the per-segment record.)

### D. Delete the guess-ladder (handoff Step 2)
- Remove the `slot_id_map` fallback (`_clip_row` :175-181, `_load_slot_id_map`,
  wiring at :512/:542) and delete `labeling/fixtures/id_maps/*_slots.json`.
- Fail-close the **weak tiers** of `match_manifest_for_path` (`als/identity.py:85`):
  keep only the exact (tag-stripped) path tier used for the `manifest`
  display/inventory; identity `recording_id` now comes from content, so the folder/
  stem-root guess tiers are removed. Callers (`enrich_gt_track_ids.py`,
  `als_path_audit.py`) must still import cleanly.
- Acceptance: `grep -rn 'slot_id_map' --include=*.py . | grep -v attic/` empty.

### E. Export gate reconciliation
`ID_COVERAGE_MIN` currently gates on any `recording_id`. Re-base it on
`id_source == "content"` and surface the abstain list. Measure the real abstain rate
after wiring; expectation is high content coverage (candidates + stems + payload-
bound masters), with abstains confined to genuine placeholders (`mix.m4a`) and
ad-hoc files not in `track_audio`.

### F. C1b renumber-metamorphic gate (handoff Step 4)
Extend `tests/test_alignment_metamorphic.py`: renumbering `.als` slots must NOT change
the content-bound identity (the property `slot_id_map` violated by construction).
Green gate, then regenerate `docs/alignment_status.md` via `/align-checkpoint` on the
de-poisoned GT — the first honest post-Crush numbers. **This is Crush exit.**

## Out of scope (related, sequenced separately)

- **Step 3 — audio round-trip law** (#37 m4a-decode false-fail on py3.14): an
  independent denotational `.als → mix` gate. Not required for the content-binding
  mechanism; specced/handled on its own track.
- **BB11 (`2nvzlh2k`)**: same wiring applies; validate after BB12 lands.
- No `track_stems` schema change (chose "hash at catalog-build").

## Testing

- `test_content_resolver.py` (landed) covers the resolver.
- New: mp4 `mdat` hasher unit test — tagging a copy leaves `payload_sha256` stable
  (the validated invariant); full-file sha256 changes.
- New: catalog-build unit test over a fixture DB slice (sha256 + payload + stem
  entries; recording via `track_audio_id`).
- New: export integration test — BB12 slots 028/031/144 resolve to the correct
  recording or abstain; **none carry a cross-song id**; `id_source` stamped on every
  row.
- C1b metamorphic (Component F).
- `make check` (guardrails + `gt_als_gate` + pytest). Worktree off `main`; symlink
  `venvs` so the pre-commit hook finds `venvs/audio/bin/python`.

## Risks

- **Re-pull disturbs the annotator's aligning folder.** Mitigate: no `--prune`;
  build only regenerates `manifest.json` + emits the sidecar; snapshot GT first.
- **AAC decode nondeterminism** — sidestepped: `mdat`-box hashing reads compressed
  bytes, no decode, so it is bit-exact across machines (unlike PCM-decode hashing).
- **Abstain rate could still trip the gate** on ad-hoc candidate files absent from
  `track_audio`; measure before finalizing `ID_COVERAGE_MIN` semantics.
