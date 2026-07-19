# Agent handoff — "wrong audio fed silently" audit (2026-07-18)

**Status:** investigation COMPLETE; resolver hardening **merged** as PR #21
(`a13be41`). Acappella proxy contamination claim **RETRACTED**. Bench tag join
shipped in mashup_demo; BB12 `--tagged-only` re-render wrote 4 WAVs.
Stable `track_audio_id`→path index: branch `feat/track-audio-id-index`.
Canonical memory: `project_identity_by_string_bug_class`.

## How this started

A prior agent added a **tags-first tempo** path to the mashup compiler: `analyze_song`
now prefers `annotator bpm=NN key=KK` (written into `~/aligning` M4A comment tags) over
its own estimate, falling back to Rekordbox/Serato `TAG:BPM=` then estimation. It asked
to re-render the BB12 GT set to judge the result by ear.

Before rendering I verified the mechanism and found the render would be **inert** — which
opened a much larger question the user posed: *is feeding the wrong song/audio unknowingly
why alignment is hard, and are there other bugs like this?*

## What was verified (evidence)

### The tags-first render is inert through the bench
- `analyze.py:read_annotator_tags` greps ffprobe `format_tags` for `annotator bpm=`. The
  real `~/aligning/1fsnxchk/tracks/*.m4a` files **do** carry that tag (e.g.
  `104__Mako … [128bpm 1B].m4a` → `comment=annotator bpm=128 key=1B`), so the reader works
  on those files. My first hunch that the regex never matches was **wrong** — corrected.
- BUT the bench (`~/Desktop/mashup_demo/studio/tools/bench_gt.py`) reads tags from the
  **manifest `local_path`**, which points at a **different, untagged copy** of the track.
  `--tagged-only` finds **0/59** pairs; the default run's tag-inheritance block (lines
  285-314) silently no-ops for every track.
- Root of the miss: annotator-tagged copies use a **flat 1–154 renumber** disjoint from the
  manifest's slot labels (`029`, `002w1`). **0/155** BB12 manifest tracks reference a tagged
  copy. An artist-title join (ignore slot number + acap/full) recovers **147/155 = 95%**,
  4 conflicting-bpm cores, 8 true misses — a viable fix if we want it.

### Three parallel audits (all read-only at first)

**1. Duplicate-copy blast radius** — annotator rename spawns parallel copies:
- Present in 4/5 GT sets (BB11 99% of slots; Disco Lines clean). Mostly **same-inode
  hardlinks** (wrong copy = identical audio).
- BUT **content-divergent** copies exist: **BB12 47 / BB11 46 / BB10 20** slot-cores hold
  copies with >2s duration spread — e.g. Daft Punk – Around The World 429.5s vs 31.9s
  fragment; Curbi – Triple Six 190.5s bare vs 251.5s extended; Chainsmokers – Honest with
  SAVI/Virtu/Acappella/bare all under one core.

**2. Aligner audio-loading trace** — where does the *real* pipeline pick a file:
- ✅ CLEAN (stable-id joins, no filename matching): GT scorer
  `workspaces/alignment_prototype/score_timeline_vs_gt.py:185-192` (joins by
  track_id/recording_id) and manifest writer `labeling/pull_set_for_alignment.py:343-361`.
  **⇒ headline scores and the `project_identity_miss_decomposition` "84% not data" verdict
  are NOT contaminated by this bug.**
- ✅ MITIGATED (PR #21): `stem_resolve.resolve_stem` and `infer_fused._resolve_ref` now
  fail closed on ambiguous slot-glob fallback (warn + abstain with `track_audio_id`).
  Stem callers no longer substitute full-track audio after abstention. Residual: a true
  `track_audio_id`→path index (separate PR).

**3. Content-vs-label correctness** (BB11 + BB12) — **proxy claim retracted:**
- Ghost files: 1 (BB11 `027 Class & Clowns - ID`). Missing stems: 0.
- First-pass DSP proxy (~23% BB12 / ~7% BB11 "kick in acappella") measured the
  **full-mix `local_path`**, not the channel alignment actually consumes.
- Of those suspects, **8/9 unique IDs are intentional baby-rule `wrong_stem` fallbacks**
  (`track_audio.stem='regular'`): manifest marks them; aligner uses Demucs/Roformer
  `vocals.flac`. Loud instrumental stems on those sources are expected.
- Consumed vocal stems: vocal−sub60 ≈ **27–41 dB**, matching clean acap controls.
  **No alignment-input vocal contamination confirmed.**

## Verdict

The "audio joined to identity by a **mutable string** (filename / slot label / claimed_stem)
instead of a **stable id** (track_audio_id / recording_id)" pattern is a **real, recurring
bug class** — today's instances + three already in memory (routing mis-route −12.6pp,
scorecard stem-axis contamination, BB11 `[bpm key]`→track_id=None). But:
- It is **NOT the headline difficulty** — the scorer and decomposition are clean.
- It **was** an active contamination of the **bench/listening loop** and a **latent landmine**
  in aligner resolvers — the landmine is fail-closed in PR #21; the bench tag miss is
  mitigated in mashup_demo `bench_gt.py` (artist-title join; BB12 0→35/59 tagged pairs).
- The acappella "wrong audio costs accuracy" hypothesis is **retracted** for BB11/BB12
  alignment inputs (proxy measured the wrong file).

## Next actions

1. ~~Verify the acappella contamination~~ — **retracted** (measured consumed vocals; clean).
2. ~~Harden aligner resolvers~~ — **merged PR #21** (`a13be41`).
3. ~~Fix manifest↔tag namespace (bench)~~ — **done in mashup_demo** (`bench_gt.py`
   artist-title join, fail-closed on BPM conflicts; BB12 **0→35/59** tagged-both).
   Durable follow-up: key tags by `track_audio_id` so inheritance needs no fuzzy join.
4. ~~Stable-id local index~~ — `feat/track-audio-id-index`: `audio_index.json` keyed by
   `track_audio_id`; written at pull; resolvers consult before slot globs; refresh via
   `python -m labeling.audio_index <set_dir>`.
5. ~~BB12 listening re-render~~ — **done** (`--tagged-only -n 4`):
   `studio/bench/gt/{1,2}_CalvinHarris-TheChainsmokers.wav`,
   `3_CleanBandit-MakoSmoke.wav`, `4_FrozenLet-JAUZCrankdat.wav` (2026-07-18 21:40–43).

## Parallel agents (do not collide)

- `acap-instance-separability` edits `joint_ref_decode.py` (same file as PR #21)
- Stay out of other worktrees' active files

## Reproduction

```bash
# bench (tagged pairs after join; was 0/59):
cd ~/Desktop/mashup_demo && ~/Desktop/mashup_compiler/venv/bin/python \
  studio/tools/bench_gt.py --set bb12 -n 4 --tagged-only

# clean production path where the tag DOES fire (drag two tagged files in):
cd ~/Desktop/mashup_compiler && venv/bin/python -m compiler.main \
  "<vocal .m4a with [bpm key]>" "<instrumental .m4a with [bpm key]>" --out out/test
```

Set ids: BB11 `2nvzlh2k`, BB12 `1fsnxchk`, BB10 `w1mgcjt`, Disco Lines `1rfb0yl9`,
Murph `pwgrrb1`. Aligning folders under `~/aligning/`.
PR: https://github.com/jca225/tracklist_engine/pull/21
