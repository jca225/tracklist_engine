# DJ mashup craft — practitioner rules (2026-07-11)

Mined from DJ education sites + Reddit + artist interviews (deep-research,
adversarially verified — hard-gate claims were *refuted*, which validates our
soft-gate design). Each rule tagged **ENCODABLE-NOW** / **NEEDS-NEW-SIGNAL** /
**TASTE-ONLY**, with contradictions against our mined grammar
([[mashup-grammar-prior]]) flagged. Companion to
[mashup_decision_model_plan.md](mashup_decision_model_plan.md).

## ★ The headline finding — pairing is ear-validated, not rule-selected

**Two Friends (the artists behind our BB ground truth) make mashups by
batch-and-whittle: export ~200 candidate pairings, keep ~45 by ear.** Key/BPM
are *manipulated to enable* a chosen pair, explicitly NOT used to select it —
"the most unexpected combinations become the most memorable." (Joe Vulpis
podcast + DJ Times 2019, 3-0 verified.)

This is the strongest possible validation of the product architecture:
- The pairing **decision** is **TASTE-ONLY** → needs a learned scorer or human
  loop; the compiler must **propose + rank many candidates, not decide one**.
- The harmonic/level/phrasing rules gate **playability**, not quality.
- **The seam feed + refinement verbs ARE the whittling mechanism** — we are
  rebuilding, in software, exactly how the ground-truth artist actually works.
  This retroactively justifies the Genie "propose → refine" UI and the
  decision-model's preference-log (P0-3).

## Rules by area

### Harmonic (ENCODABLE-NOW — and a real gate gap)

Compatible Camelot moves (closed set, 3-0 verified): same key; **±1 adjacent
same-letter** (5A→4A/6A, = perfect fifth, ±7 semitones); **relative
major/minor swap** (8A↔8B); diagonal B→A/A→B; and the **+2-semitone energy
jump** (short sections only). Pitch-shift to force a match: **soft ≤2
semitones** (artifacts beyond; NOT a hard gate — the ±1-2 hard-limit claim was
refuted).

⚠️ **GATE GAP:** our `compiler.gate` allows same-key + relative
(canonical-pc delta 0) + **±1 semitone** transpose. It correctly handles the
relative swap (Am≡C both map to pc 0) but **rejects Camelot-adjacent** (±7
semitones → our wrap gives ±5 → rejected) and the +2 energy jump. We are
**too strict** — offering fewer valid pairs than a real DJ would. v2.1 gate
should accept the perfect-fifth and +2 relationships.

### Phrasing (ENCODABLE-NOW — matches our grammar)

Align entries/exits to the **8/16/32-bar grid**, start elements on **beat 1 of
the next phrase**; ~16-bar overlaps for house, shorter for pop chorus cuts.
✔ Consistent with our mined grammar (16-bar hook, phrase-aligned entry) — no
contradiction. Our pickup-led entry is a refinement craft sources don't
name but don't forbid.

### Levels / EQ (PARTIALLY ENCODABLE — two cheap wins we're missing)

- **High-pass the acapella ~80 Hz (up to ~150)** — removes vocal-stem
  low-end mud. **ENCODABLE-NOW, not yet done.** Likely a real chunk of any
  residual muddiness in demo_v2.
- **Bass-swap: only one dominant low end at a time.** ENCODABLE-NOW (HPF the
  vocal + trust the bed's bass); a fuller version (duck bed sub under vocal
  sub) is NEEDS-NEW-SIGNAL (per-band energy).
- **Vocal-vs-bed LUFS offset is genre-relative** (~0 acoustic, −2 pop, −3
  rock; safe envelope +1.5 to −4.5 dB), not a flat equal-match. ✔ Consistent
  with our near-flat-ducking finding (level set pre-entry). Our v2 pure
  LUFS-match (offset 0) is inside the envelope but should bias the vocal a
  touch under the bed. Minor `_auto_gain` refinement.

### Acapella craft (mixed)

Cut on the phrase (not mid-word) — ENCODABLE-NOW via onset+downbeat (we snap
to downbeat; "not mid-word" needs lyric/onset alignment = NEEDS-NEW-SIGNAL for
a true fix). De-reverb / timing-correction prep = NEEDS-NEW-SIGNAL. Vocal-over-
vocal clash avoidance = TASTE-ONLY / NEEDS-NEW-SIGNAL.

## Actionable this month (ranked)

1. **HPF the acapella (~90 Hz) in the renderer** — cheapest perceived-quality
   win; one line in `_process`. Do it in v2.1 and re-A/B demo_v2.
2. **Widen the gate** to Camelot-adjacent (perfect fifth) + relative +
   +2-semitone energy jump — offers more (and more *interesting*) pairs, which
   the batch-and-whittle finding says is the whole point. Gate stays a
   playability filter, not a quality judge.
3. **Small vocal LUFS bias** (vocal ~−1.5 dB under equal) — trivial
   `_auto_gain` tweak toward the craft envelope center.
4. **Architecture confirmation (no code):** the decision model must *rank
   candidates*, the app must *propose many*. The verb log is the whittle.
   Fold into Stage-1 framing of the decision-model plan.

Nothing here contradicts the mined grammar; the craft ADDS the harmonic
breadth (adjacent keys), the EQ layer (HPF/bass-swap), and — most importantly —
reframes pairing as propose-and-whittle, which is exactly what we're building.
