# Phase 6 — Analysis / embedding readiness

The bridge between the **finished identity work**
([identity_and_inventory_plan.md](identity_and_inventory_plan.md), Phases 0–5 ✅)
and the **alignment north star**
([alignment_objective.md](alignment_objective.md) deliverable **B**, the aligner).

Deliverable B trains on `{tokenized tracklist, track audios, set audio}` → MERT/
analysis features. Today those features are ~5% present and mostly on the wrong
model, and the **set/mix side is not embedded at all**. This plan tracks closing
that gap. Status tracked inline.

> Vocabulary: the identity **stem** axis (`regular`/`acappella`/`instrumental`)
> is **not** Demucs `stem_name` — see root [CLAUDE.md](../CLAUDE.md). MERT layer /
> 330M rationale lives in [analysis/CLAUDE.md](../analysis/CLAUDE.md).

## Current state (as of 2026-06-01, canonical pi-storage)

| Signal | State |
|--------|-------|
| `track_audio` rows | ~18,050 — **all `stem='regular'`** (no acappella/instrumental ingested) |
| `track_mert_measures` | 93,705 rows / **819 tracks** (~4.5% of reference tracks) |
| → 768-dim (**MERT-v1-95M**, superseded) | 815 tracks |
| → 1024-dim (**MERT-v1-330M**, target) | **0 tracks** (adapter flipped; nothing re-embedded yet) |
| `track_mert_sections` | **0** — populated post-BPE (8b), empty *by design* |
| Set/mix-side MERT | **no table, no path** |
| `track_audio_features` (Essentia) | ~1,650 (~9%) |
| Demucs-stemmed tracks | ~820 distinct |

The five "types" still owed map to orthogonal axes: **model** (330M) ·
**side** (set vs track) · **granularity** (measure vs section) · **layers**
(all-layer stack) · **stem** (regular/acap/instr).

## Tasks

### 6a — 330M reference backfill
- [x] Diagnose stall: pilot `BB_SETS` filter + `track_analysis IS NULL` queue skips
      the 815 stale 768-dim rows; driver is `scripts/mert_backfill_loop.py`.
- [ ] Re-embed the ~17.2k uncovered reference tracks with `m-a-p/MERT-v1-330M`,
      all hidden layers per measure (current production format,
      [analysis/models.py](../analysis/models.py)).
- [ ] Retire the 815 stale **768-dim (95M)** rows (different model *and* likely
      single-layer format — confirm before bulk delete).
- [ ] Cost/disk estimate first: ~50 KB/measure × all-layer → on the order of
      100+ GB for full corpus; size the GPU-hours (Vast 4090 vs Mac MPS) and the
      pi-storage disk headroom before kicking off.

### 6b — Set/mix-side MERT (does not exist yet)
- [ ] Design `set_mert_measures` (mirror of `track_mert_measures`, keyed on
      `set_audio_id`, on the beat-grid from `set_measures`).
- [ ] Worker path to embed the ~549 mix recordings (and the ~20k clean-corpus
      target as it lands). Alignment needs **both** sides embedded to match.

### 6c — Variant MERT (blocked on ingest)
- [ ] Gated on stem ingest: corpus is 100% `stem='regular'` today, so
      acappella/instrumental variants must be acquired first
      ([stem_discovery_playbook.md](stem_discovery_playbook.md)).
- [ ] Each variant gets its own `track_audio_id` → its own all-layer MERT;
      aligner picks the embedding by axes key.

### 6d — Re-sourcing dependency (critical path, from the GT review)
- [ ] The May 6–9 bulk download predates variant-aware YT Music search,
      chromaprint QA, three-axis identity, and the correction ledger
      ([alignment_objective.md](alignment_objective.md#L79), review
      [alignment_review_20260530.md](alignment_review_20260530.md)).
      Re-source affected tracks **before** scaling GT or embedding at scale —
      otherwise we embed the wrong audio.

### 6e — Section embeddings (downstream, not a blocker now)
- [ ] `track_mert_sections` is filled post-BPE by the cue optimizer (Phase 8b),
      reaggregating `track_mert_measures` — no MERT rerun. Sequence after 6a.

## Deferred — retrieval index (turbovec / quantized ANN)
Quantized ANN ([turbovec](https://github.com/RyanCodrai/turbovec), 2-bit/4-bit,
~1 GB for the corpus single-layer) is a **derived search index for candidate
generation**, not a store of record and not a storage win — it is *additive* over
the float16 all-layer cache, which stays canonical for training. **Premature**
until 6a/6b exist. When built: validate recall vs full-float on a labeled set
before trusting 2-bit (standing external-shortcut gate).
