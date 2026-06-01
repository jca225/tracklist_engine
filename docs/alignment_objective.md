# Alignment objective (north star)

**Target: August 1.** Build the **alignment** stage at the tail of this repo's DAG
(`core · scrape → ingest → analysis → labeling → (GT) → alignment`). This is *this
repo's* terminal step — distinct from the downstream mix-generation learning
codebase. Throughout: **labeling** = the human Ableton ground-truth pass;
**alignment** = the algorithm that learns from it. Never swap the words (see the
terminology block in [../CLAUDE.md](../CLAUDE.md)).

## Corpus scope

Define a **clean corpus** of DJ sets — roughly **20,000 sets** — by three filters:

1. **Low-noise tokenization** — few "messy" fields (proxied by suggestion tokens
   emitted by the tokenizer).
2. **Curated whitelist** — includes the full tracklists of Two Friends, It's Murph,
   John Summit, and Disco Lines, plus subjective additions.
3. **Has set audio** — the mix recording itself is present (hard requirement).

This 20k is the **inference target** the trained aligner must generalize over. It is
**not** the manual-labeling scope.

## Three deliverables (do not conflate)

### A. GT reader
A parser that ingests our manual Ableton alignment work (`.als`) into the data model
at high granularity: set-level **BPM** and **volume**, and per-track **sections
played, warping, key change, FX used**, and **flags where an outsourced song worked
better** (substitutability signal). This is an *input* to training, not an aligner
capability.

### B. The aligner
A model trained on the GT from (A) that, given **{tokenized tracklist, track audios,
set audio}**, produces a data structure rich enough to round-trip back into Ableton.
Target capabilities:

- **B1 (warp):** warp an acappella overlaid on an instrumental to the correct BPM.
- **B2 (key):** infer which key a stem should be in relative to the canonical set
  audio.

### C. Manual labeling
Hand-label a **small** subset of sets extensively in Ableton to produce the GT that
(A) reads and (B) trains on. Human task, not an algorithm. This is the gating
dependency on the timeline, not the 20k.

## Upstream capabilities (ingest/analysis, *not* the aligner)

These feed alignment but belong to earlier stages — keep code in the right module.
**Execution tracking:** [embedding_backfill_plan.md](embedding_backfill_plan.md)
(Phase 6 — 330M backfill, set-side MERT, variant MERT, re-sourcing).

- **Stem discovery** *(ingest / official-stems search)* — find or select better
  acappellas/instrumentals online, and choose among separators (Demucs vs UVR).
- **Version/variant QA** *(ingest / acquisition gate)* — detect when we downloaded
  the wrong **version** (remix, rework, …) or **variant** (extended, regular).

## Success criterion

The aligner reproduces held-out manual GT within a stated tolerance, then generalizes
from the small hand-labeled set to the ~20k clean corpus.

**Tolerances — UNSET (placeholders, pin before this is final):**

- transition/section timestamps within **±N bars** of GT
- key within the correct pitch class
- BPM within **±M**

## Open decisions

- **Tolerance numbers** above are placeholders, not decided values.
- **One algorithm or a family** — this doc attributes stem-discovery and version/QA
  to **ingest**, leaving the aligner focused on warp/key/section reproduction. If the
  intent is a single end-to-end model that also does acquisition, that split should be
  revised.

## Manual labeling review (2026-05-30)

First structured feedback after a full Ableton labeling pass:
[alignment_review_20260530.md](alignment_review_20260530.md).

**Critical context:** the audio used for that pass came from a **May 6–9 bulk
download** (17,920 / 18,044 `track_audio` rows on pi-storage) that ran **before**
several ingest fixes landed — notably variant-aware YT Music search (`2cdb892`,
2026-05-13), chromaprint QA, three-axis identity, and the correction ledger.
Many "wrong remix / wrong version" observations match a known pre-fix failure
mode, not an aligner limitation. **Re-sourcing affected tracks is on the
critical path** before scaling GT or training.
