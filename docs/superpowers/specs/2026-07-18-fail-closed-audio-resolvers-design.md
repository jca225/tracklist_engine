# Fail-closed audio resolver design

**Goal:** prevent the aligner from silently selecting audio by mutable filename
or slot label when a manifest path is stale.

## Scope

Harden the two active fallback resolvers identified by the wrong-audio audit:

- `stem_resolve.resolve_stem`
- `infer_fused._resolve_ref`

Valid manifest paths continue to resolve normally. A missing or stale path may
fall back to a slot-prefix glob only when that glob identifies exactly one
candidate. Multiple candidates cause an explicit warning and an abstention
(`None`). The warning identifies the requested asset with `track_audio_id`,
slot label, and stem name where applicable.

Stem-routed callers must propagate the abstention instead of falling through to
the full reference track.

This patch does not create a stable-ID file index, repair manifests, change
audio inventory, or alter the tags-first mashup bench.

## Rationale

A filename glob cannot prove recording identity, but current manifests omit
many stems generated after pull time. Removing fallback resolution entirely
would drop those valid inputs. Requiring one candidate preserves unambiguous
post-pull stems while removing the current first-hit behavior where
content-divergent duplicates exist.

A stable `track_audio_id` to local-path index remains the stronger long-term
solution.

## Behavior

`resolve_stem`:

1. Return `track["stems"][stem_name]` when it exists as a file.
2. Otherwise collect matching on-disk stem files across normalized slot forms.
3. Return the candidate when exactly one distinct file exists.
4. If multiple candidates exist, warn with stable identity context and return
   `None`.

Stem-routed callers return `None` after that abstention. They do not substitute
`track["local_path"]`.

`_resolve_ref`:

1. Return `track["local_path"]` when it exists as a file.
2. Otherwise collect non-ASD files matching the slot prefix.
3. Return the candidate when exactly one file exists.
4. If multiple candidates exist, warn with stable identity context and return
   `None`.

Callers already treat `None` as unavailable input, so no new exception path is
introduced.

## Verification

Tests must first reproduce the current silent substitutions:

- a stale stem manifest path plus multiple slot-matching stem directories;
- stem-routed callers falling through to a full track after abstention;
- a stale reference path plus multiple slot-matching track files.

After the change, ambiguous cases must return `None` and warn with
`track_audio_id`. Existing valid-path behavior and unambiguous fallback behavior
must remain unchanged. Run the focused tests, then the repository gate.
