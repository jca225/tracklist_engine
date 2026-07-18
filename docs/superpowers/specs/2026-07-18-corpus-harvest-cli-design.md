# Corpus-harvest CLI — design

> **Date:** 2026-07-18 · **Branch:** builds on `worktree-cotrain-accept-precision`
> (PR #13 — the co-training write-side lives there).
> **Status:** approved design, pre-implementation.

## 1. What this is

Step (2) of the co-training corpus flywheel, per
[docs/alignment_state_of_record.md](../../alignment_state_of_record.md) §3
("Remaining to actually RUN the corpus flywheel, in order: … (2) a thin
corpus-harvest CLI (pi DB queries → build cases + corpus scorer → write
ledger)").

A **thin batch runner** that turns already-downloaded+analyzed corpus sets on
pi-storage into a harvest-ledger of pseudo-labelled (ref ↔ mix-span) training
pairs, gated by the certified ACCEPT-precision policy. It is *glue*: it adds a
**corpus case-builder** (pi DB → cases) and a **batch loop**, then delegates all
alignment logic to existing, tested machinery
(`harvest.harvest` / `cotrain_seam.real_probe_scorer` / `corpus_mix_resolver` /
`harvest.write_ledger`). No new probes, no new banding, no new decode.

**It does not** touch the canonical DB or the correction ledger (inherits the
`cotrain_seam` invariant: ZERO autonomous canonical mutation — writes ONLY the
harvest-ledger JSONL).

## 2. Why it is safe (certification carries over)

The ACCEPT-precision gate (2026-07-18) certified `regular` poison-free at
2-channel agreement and `instrumental` at unanimous 3-channel, on GT-*windowed*
spans. The corpus has no GT windows; it uses the **scraped 1001TL cue time**
(`set_track_slots.cue_time_seconds` / `cue_seconds`) as the `MixSpan.set_start_s`
anchor ([[project_cue_time_placement_lf]], ~1s median on BB11).

A noisy cue-time window is a **recall** cost, not a precision cost: if the window
is wrong, the certified probes (fp / chroma / continuity) fail to agree on an
offset → the span bands ABSTAIN → it is **not harvested**. Bad placement never
manufactures a confident-but-wrong pseudo-label, so the per-axis precision
certification transfers. This is the flywheel's abstain-heavy philosophy.

## 3. No GPU required for the certified axes

`capture_votes._STEM_TO_PROBES`: `regular → (fp, chroma)`,
`instrumental → (fp, chroma, continuity)` — all DSP/librosa, **no HuBERT**. Only
`acappella` needs HuBERT, and acappella is uncertified → never harvested
(`CERTIFIED_POLICY` has no acappella key). So the corpus harvest for the
certified axes is CPU-only and runs **on pi-storage**, where the DB + audio +
stems live — no Vast box needed for this step (Vast is only for step (1), the
RoFormer mix-side stem separation this step consumes).

## 4. Run model & paths

Runs **on pi-storage** (deployed via `make deploy`, invoked with the CPU
`venvs/audio` python). Reads the canonical DB and `/mnt/storage` audio/stems
directly; writes the ledger locally. All paths are **overridable args with
canonical defaults**, so the same module is unit-testable against a fixture DB
with no pi access:

| Arg | Default (pi-storage canonical) | Meaning |
|---|---|---|
| `--db` | `/mnt/storage/data/db/music_database.db` | canonical DB |
| `--stems-root` | `/mnt/storage/stems/set` | mix-side stems: `<stems-root>/<set_audio_id>/{vocals,instrumental}.flac` |
| `--set-audio-root` | *(none — use `set_audio.path` verbatim)* | optional prefix if DB paths are relative |
| `--ref-audio-root` | *(none — use `track_audio.path` verbatim)* | fallback root for ref audio |
| `--out` | *(required)* | harvest-ledger JSONL (appended, idempotent) |

> **Mojibake caveat** ([[project_track_audio_path_mojibake.md]]): some
> `track_audio.path` / `set_audio.path` values are UTF-8-stored-as-Latin-1. The
> resolver must tolerate a missing file gracefully (it already does — absent
> audio → all probes abstain), and the census counts such rows as
> "audio-missing", not eligible.

## 5. Components (all in `workspaces/pws_aligner/corpus_harvest.py`)

A single new module, mirroring `harvest.py`'s shape. Pure functions at the core,
thin `main()` at the edge (repo Rust-flavoured-functional style).

### 5.1 `CorpusSlot` (frozen dataclass)
A DB-projected eligible slot, decoupling SQL from case-construction:
`set_id, set_audio_id, slot_label, recording_id, ref_path, claimed_version,
claimed_stem, claimed_variant, cue_time_s, duration_s, mix_full_path`.

### 5.2 `query_corpus_slots(conn, *, policy_stems, limit=None) -> list[CorpusSlot]`
One SQL join, no audio touched. Selects slots that are candidate-eligible:

- `set_track_slots` ⋈ `set_audio` (reference mix per set) ⋈ `track_audio`
  (reference ref per slot's `recording_id` at the slot's `claimed_stem`).
- Filters: `set_audio.is_reference=1`; `track_audio.is_reference=1 AND
  track_audio.stem = claimed_stem`; `claimed_stem IN policy_stems`
  (regular/instrumental only — the certified axes); a non-null cue time
  (`COALESCE(cue_time_seconds, cue_seconds)`).
- `duration_s` = `set_track_slots.duration_seconds` when present.
- Deterministic order (`set_id, row_index`) so `--limit` and resume are stable.

This is the "pi DB queries" half. It returns *candidate* eligibility; **on-disk
audio existence is checked later** by the resolver/scorer (disk is truth — see
§2/§6), not in SQL.

### 5.3 `build_corpus_cases(slots) -> list[tuple[RefCandidate, MixSpan, dict]]`
Pure map, one case per slot (**positive-only**; decoys were only for the
precision *gate*, not for harvesting):

- `RefCandidate(recording_id, source_url="corpus://<track_audio row>",
  source_path=ref_path, stem=claimed_stem, version=claimed_version,
  variant=claimed_variant)`.
- `MixSpan(set_id, slot_label, set_start_s=cue_time_s,
  span_dur_s=duration_s or DEFAULT_SPAN_S)`.
- `claim_axes = {version, stem, variant}` (drives the PROPOSED correction path in
  `cotrain_seam` when a candidate differs from the claim — here they match by
  construction, so correction is None; kept for symmetry).

`DEFAULT_SPAN_S` (e.g. 40.0s) is used when the slot has no scraped duration —
matches the BB smoke default and the typical play-span length.

### 5.4 `run_corpus_harvest(slots, *, stems_root, out, policy=CERTIFIED_POLICY) -> HarvestSummary`
The batch loop. Groups slots by `set_audio_id`; for each set builds **one**
scorer via
`real_probe_scorer(mix_resolver=corpus_mix_resolver(mix_full_path, stems_root/<set_audio_id>))`
(so the per-span mix-feature cache is reused across that set's slots — the ~1
min/case perf fix), calls `harvest(cases_for_set, scorer, policy=policy)`, and
`write_ledger(records, out)` incrementally per set (crash-safe + resumable —
`write_ledger` dedupes by `span_key`). Returns counts (sets, cases, harvested,
written, skipped-by-reason).

### 5.5 `census(slots, *, stems_root, set_audio_root) -> CensusReport` + `--census`
The value-today, no-probes mode. Walks the same eligible slots and classifies
each by **what blocks harvest**, checking disk for audio:

- eligible-now (mix present, mix-stem present for instrumental, ref present, cue
  present) — the recall ceiling *today*;
- blocked: `no-mix-audio` / `no-mix-stem` (instrumental only) /
  `no-ref-audio` / `no-cue-time`.

Aggregates by axis (regular vs instrumental) and prints a table + JSON. This
quantifies the flywheel's reach **before** the Vast stem pass runs, and
re-run after it shows the pass's payoff. `--census` short-circuits before any
probe import (fast, dependency-light).

### 5.6 `main(argv)` — argparse
`--db --stems-root --set-audio-root --ref-audio-root --out --limit --stem
--census`. Mirrors `harvest.main`'s option style; `--stem` restricts to one
certified axis; `--census` selects the report mode (makes `--out` optional).

## 6. Data-flow

```
pi DB ──query_corpus_slots──► [CorpusSlot]
                                  │
                    ┌─────────────┴──────────────┐
              --census                        (harvest)
                    │                             │
             census(disk-checked)         build_corpus_cases
                    │                             │
             CensusReport (table+JSON)     group by set_audio_id
                                                  │
                                    real_probe_scorer(corpus_mix_resolver)
                                                  │
                                    harvest(cases, scorer, CERTIFIED_POLICY)
                                                  │
                                    write_ledger(out)  ── idempotent JSONL
```

## 7. Error handling

- Missing audio (mix / mix-stem / ref) → the scorer's probes already abstain
  (safe no-op); the case bands ABSTAIN → not harvested. The census reports these
  as blocked rather than letting them fail.
- Per-probe exceptions are already absorbed into abstain by
  `capture_votes._run_probe_safe` (inherited).
- DB open failure / bad `--db` → fail-fast `sys.exit` (edge, per style guide).
- Ledger write is append + dedupe → interrupting mid-run and re-running is safe
  (resume = skip seen `span_key`s).

## 8. Testing (TDD, against a fixture DB + fake scorer — no pi, no real audio)

New `workspaces/pws_aligner/tests/test_corpus_harvest.py`:

1. **`query_corpus_slots`** on an in-memory SQLite fixture: returns only
   candidate-eligible slots; excludes non-reference mix/ref, wrong-stem ref,
   uncertified stem (acappella), missing cue time; respects `--limit` and
   deterministic order.
2. **`build_corpus_cases`**: correct `RefCandidate`/`MixSpan`/`claim_axes`
   mapping; `set_start_s` = cue time; `span_dur_s` falls back to `DEFAULT_SPAN_S`.
3. **`run_corpus_harvest`** with a **fake `RefMixScorer`** (agree → ACCEPT,
   disagree → ABSTAIN): only ACCEPTs written; instrumental requires 3-channel;
   idempotent re-run writes 0; per-set scorer built once (spy on factory).
4. **`census`** with a `tmp_path` fake stems/audio layout: eligible vs each
   blocked reason counted correctly per axis; instrumental blocked when
   `instrumental.flac` absent even if the row exists.
5. **certified-policy guard**: acappella slots never produce cases nor ledger
   rows regardless of scorer output.

Run with the repo's `venvs/audio/bin/python -m pytest`. All existing pws_aligner
tests must stay green (`harvest`, `cotrain_seam`).

## 9. Out of scope (explicit)

- The GPU RoFormer mix-side stem pass (step 1) — separate, Vast-bound, blocks the
  *real* run but not this CLI's construction.
- acappella harvesting (uncertified; off-Mac HuBERT cert is step 3).
- Consuming the ledger for training (downstream — trajectory/ml).
- Per-probe [0,1] confidence calibration (open work noted in state-of-record;
  banding leans on offset agreement first, unaffected here).
- Any canonical DB / correction-ledger writes.

## 10. Where it lands

Module + tests + this spec on a follow-on branch `cotrain-corpus-harvest`
**stacked off** `worktree-cotrain-accept-precision` — so PR #13 stays reviewable
as-is and this new work depends on (but does not amend) it. Merges after #13.
State-of-record §3
item (2) gets ticked via `/align-checkpoint` when merged. No numbers change
(`alignment_status.md` untouched — this is machinery, not a scored result).
