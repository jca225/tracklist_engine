# Follow-ups — worst-span audit (BB11 + BB12)

> Companion to [HANDOFF.md](HANDOFF.md) (state + prior TODO) and
> [FINDINGS.md](FINDINGS.md) (detail taxonomy). Headline numbers live ONLY in
> [docs/alignment_status.md](../../../docs/alignment_status.md) — never hand-typed here.
> BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.

Hand-auditing the worst-scoring aligner spans on BB11 and BB12 revealed that several
"aligner errors" are actually **our-side scoring bugs**, plus two genuine algorithmic
gaps. Fix in priority order — **WS0 first**, because it invalidates the metric the
rest are measured against.

**Canonical inputs**
- BB11 GT: `~/aligning/2nvzlh2k…/BB11 align Project/BB11 align.als`
- BB12 GT: `~/aligning/_backups/20260616_150150/big bootie 12 labeling Project/big bootie 12 labeling_fast.als`
  (lives inside `_backups/` — **not disposable**)
- GT write-back → `set_ground_truth` on the **pi-storage canonical DB** (repo `data/db` is a stale dev copy).
- Use `venvs/audio/bin/python` from repo root; run `make check` before pushing.

**Guardrails**
- WS0 before anything else — a corrupted metric makes WS1–WS4 unmeasurable.
- Canonical DB is on pi-storage; write-back before re-scoring if the scorer reads `set_ground_truth`.
- Headline numbers only in `docs/alignment_status.md`; new findings → `eda/…/findings.md` + `aux.db`.
- Some file paths here are from memory — verify against current code before editing.
- Report confidence honestly; don't fit laws on n<50.

---

## WS0 — GT export counts phantom spans (BLOCKER)

**Confirmed bug:** a deactivated Ableton clip (Good Charlotte, BB11) was exported as a
live 184 s GT span and became the #1 "loss." The export reads clip extents without
honoring the deactivation flag.

- Find the GT-export path (start: `labeling/write_back_ground_truth.py`, the
  `labeling/als` codec's `parse_layer_clips`, and whatever the scorer / worst-spans
  loader reads).
- Filter out any clip where `Disabled == true`, and — per the fader-silence lesson
  ([feedback: .als volume slider silences clips]) — clips with clip/track volume 0 or
  muted. **Audibility ≠ clip extent.**
- Add unit tests with fixtures: a deactivated clip and a volume-0 clip must not appear
  in exported GT.
- Re-audit all GT sets (BB10, BB11, BB12, Murph) for phantom spans; report counts.

**Deliverable:** export honors activation + audibility; test coverage; phantom-span
audit table.

### WS0 — DONE (2026-07-12)

**Mechanism was mis-stated in the brief.** The phantom spans are real, but they come
from the **Track Activator** (`DeviceChain/Mixer/Speaker/Manual="false"`), not the clip
`Disabled` flag. Across BB11+BB12, **0** phantoms came from clip-`Disabled` or a
fader-at-0 — every one is a deactivated *track*. `parse_layer_clips` filtered mix tracks
by name but never checked activation, so clips on deactivated non-mix tracks exported as
live GT.

**Good Charlotte was a red herring** — not a phantom. No clip is `Disabled=true`; its GT
span today is a clean 37 s at 383 s. The old 214 s/205 s "184 s loss" was a GT *placement*
mislabel the annotator has since corrected (the machine's ~391 s prediction was ~right).

**Fix.** `ParsedClip.silence_reason` (`labeling/als/models.py`) is populated by
`parse_layer_clips` (`labeling/als/read.py`, `_silence_reason`): `"track-deactivated"`
(Speaker off) / `"clip-disabled"` (clip `Disabled=true`) / `"track-fader-zero"` (static
fader ≤1e-4, only when no volume automation). The reader stays **total** — silenced
clips are still returned (the export drop-count gate depends on it) — and
`collect_kept_clip_rows` (`labeling/export_als_to_gt.py`) drops them with a
`ReviewRow(action="dropped", reason=silence_reason)`. Every other clip consumer can filter
on the same one-liner. Tests: `tests/labeling/test_export_drops_deactivated.py`
(+ `LayerSpec` in `synth_session.py`).

**Phantom-span audit (GT-seconds removed):**

| set | kept rows | phantom spans | phantom GT-sec | all track-deactivated? |
|-----|-----------|---------------|----------------|------------------------|
| BB11 (`2nvzlh2k`) | 148 | 3 | 108.6 s | yes (Kungs "This Girl" 79.4 s, Calvin "My Way" 15.2 s, Jason Mraz 14.0 s) |
| BB12 (`1fsnxchk`) | 163 | 5 | 141.6 s | yes (175-vocals 58.5 s, Idina "Let It Go" 31.4 s, 180-vocals 23.0 s, Calvin "Slide" 14.5 s, 175-vocals 14.0 s) |
| BB10 (`w1mgcjt`) | — | — | — | no hand-labeled GT session (manifest only) |
| Murph (`pwgrrb1`) | — | — | — | not GT-complete (4-clip stub / empty `bb10 labeling.als`) |

**⚠ Metric NOT yet de-contaminated.** The scorer reads `set_ground_truth` on pi-storage,
still written from the *old* export. WS1's first step — re-export both sets and write back
(`labeling/write_back_ground_truth.py`) — is required before the scorecard reflects this.
Also `workspaces/alignment_prototype/fibers/gt_als.py` reads GT loops directly from the
`.als` via `parse_layer_clips`; it should filter `silence_reason` too (follow-up, not on
the headline metric path).

## WS1 — regenerate worst-spans & split "real miss" vs "scoring artifact"

The current worst-spans list is stale (sessions edited all day) and contaminated by WS0.

- After WS0 + a fresh GT write-back, regenerate the worst-spans and the scorecard.
- For every **STEM-AXIS row** (8 of the original 20): verify the scorer slices the form
  axis from **GT**, not from a stale `claimed_stem` / timeline tag (known contamination
  — see FINDINGS Finding 1). Quantify how many "errors" disappear once axis is read from GT.

**Deliverable:** refreshed, de-phantomed worst-span tables for BB11 + BB12, each row
tagged `real-miss | scoring-artifact | phantom`.

## WS2 — stem-axis discrimination (acappella / instrumental / regular)

**Root cause:** our identity signal is HuBERT, which keys on the voice, and the voice
is identical across a song's acappella and full versions → HuBERT is form-invariant, so
it can't tell which stem form it matched.

- Route the form decision through the **instrumental stem-to-stem fingerprint** channel
  (mix-instrumental ↔ ref-instrumental fp — already shown to flip instrumental identity
  74–88%, [project_instrumental_stem_fp]). Presence/absence of the accompaniment is the
  discriminator.
- Handle the confound: a pure acappella layered over **another** track's bed has
  instrumental energy that isn't its own — don't let that read as "regular."

**Deliverable:** per-span form verdict from the fp channel, evaluated against the WS1
stem-axis rows.

## WS3 — crossfade onset masking + scoring tolerance

Two failure shapes that are really convention/strictness, not placement errors:

- **Fade-in under a crossfade** (e.g. Follow Me ~27 s "off"): the track fades in under
  the outgoing song; the aligner locks where it becomes dominant, GT marks first-audible.
  The offset ≈ the fade length.
- **Small offsets** (Out Of Love ~14 s, Sweet Nothing ~15 s): acappella pickup/anacrusis
  — GT marks the vocal pickup, machine snaps to the downbeat.

- Define a "track start under crossfade" convention; add pickup-aware + tolerance scoring
  so a span that's otherwise tracked isn't zeroed by a sub-fade offset; optionally
  onset-under-masking detection. Reference the placement-structure EDA (acap entries are
  phase-uniform pickups — [project_placement_structure_eda]).

**Deliverable:** revised scoring tolerance + a before/after on the affected spans.

### WS3 — fade-in-under-crossfade scoring: DONE for the gain-envelope case (2026-07-12)

Galantis "You" (BB12 slot 112) was the canonical example and it's **fixed**. Root
cause: the GT clip is placed at `set_start_s=2527.8 s` but the fader ride holds it at 0
until `audible_start_s=2592.0 s` — a **64 s silent lead-in** (it fades in under the
outgoing track). The scorer measured placement against clip extent and sampled
trajectory over the whole envelope, so it charged the machine ~66 s of placement error +
the silent 64 s of "missed" trajectory for content nobody hears. The GT even carries a
`ref_segment` over the silent region, so the existing WS0 `_audible_intervals`
(segment-gap aware, but **not** gain-ride aware) didn't catch it.

**Fix** (completes the WS0 audible accounting): `_audible_intervals` now intersects with
the gain-ride window (`audible_start_s`/`audible_end_s`) via `_gain_audible_window`, and
`gt_placement_onset(row)` = `audible_start_s or set_start_s` is used for placement error
in both scorers (`score_timeline_vs_gt.py`, `build_span_table.py`). Trajectory anchoring
is unchanged (the `_audible_intervals` clip fixes sampling; the anchor must stay equal to
`trajectory_acc`'s internal `s0`). Tests: `tests/alignment_prototype/test_audible_recall.py`
(`TestGainEnvelopeAudibility`, `TestPlacementOnset`).

**Before → after (Galantis "You"):** placement error **65.6 s → 1.4 s**; GT-seconds-lost
weight **133.1 s → 68.9 s** (the span was half phantom lead-in); audible window
`[2592.0, 2660.8]`.

**Still open in WS3:** the *tolerance*/pickup-anacrusis cases (Out Of Love ~14 s, Sweet
Nothing ~15 s) — those are small downbeat-vs-pickup offsets with no fader silence, so
`audible_start_s` doesn't move; they need the pickup-aware tolerance band. Also: this
changes the **canonical headline metric** — regenerate `docs/alignment_status.md` only in
the batched WS1 rescore (with the WS0 write-back), not piecemeal.

## WS4 — looping-instrumental decode (design + build)

Feel So Close (instrumental) fails TRAJECTORY because it loops, and inside a pure loop
there is no information to distinguish lap N from N+1 ("clone-unwinnable"). Content-only
tracking can't fix this — the signal isn't there. Most pieces exist; wire for the
instrumental channel:

1. **Loop structure via fibers** (self-repeat classes, [project_fibers]) on the ref
   instrumental — know which spans are identical laps.
2. **Fingerprint phase-lock** (landmark constellation + offset histogram,
   [project_fingerprint_localizer]) mix-instrumental ↔ ref — works on repetitive EDM
   where chroma/HuBERT don't.
3. **Boundary anchoring** — detect the non-looping events (fills, drops, filter sweeps,
   transitions); those are the only phase-bearing moments. Pin loop entry/exit there.
4. **Viterbi path-decode** over offset between anchors ([project_path_decode]), constrained
   to loop-consistent jumps.
5. **Fiber-aware scoring** — credit right-content/wrong-lap, since the lap is genuinely
   unknowable.

**Deliverable:** instrumental-loop decoder wired end-to-end + eval on Feel So Close and
other `shape=loop` spans; document the clone-unwinnable ceiling honestly.

## WS5 — wrong-reference re-acquisition (small)

Near-full-semitone offsets with 0 coarse (Whethan – Savage −98 ¢; Pat Benatar – Hit Me
−99 ¢, likely BB12) are **wrong-key/version reference rips, not detunes**. Test a −1
semitone transpose by ear; if it still clashes, re-acquire the correct version (this is
**ingest/identity QA, not the aligner**).

**Deliverable:** corrected rips or a documented "no better source" verdict.

> **Detune vs wrong-version — cross-ref (not folded here).** The tool that decides
> WS5's "is this a genuine detune or a wrong rip?" question is the micro-pitch detune
> estimator ([project_micropitch_detune]): `workspaces/alignment_prototype/pitch_detune.py`,
> `measure_detune.py`, corpus arm `eda/corpus_empirics/bb_pitch_detune.py`, write-up in
> `docs/alignment_paper_draft.md`. That work is a separately-owned thread (alignment
> paper + corpus-empirics, own `aux.db` findings home) and is intentionally **kept
> separate** from this scorer/aligner brief. WS5 consumes its verdict; it is not a WS6.
