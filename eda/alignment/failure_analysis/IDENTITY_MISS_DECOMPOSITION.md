# Identity-miss decomposition — what the "84%" actually is

**Date:** 2026-07-18 · **Sets:** BB11 (`2nvzlh2k`) + BB12 (`1fsnxchk`) · **Timeline:** `_lt` (scorecard source of truth)
**Reproduce:** `venvs/audio/bin/python eda/alignment/failure_analysis/identity_miss_decompose.py`
**Canonical numbers:** [docs/alignment_status.md](../../../docs/alignment_status.md) (this doc does not restate headline metrics; it explains their composition).

## Why this exists

The headline **identity 83% / 84% (BB11/BB12)** is routinely read as either a
discrimination-model weakness or a data-supply ("no fair candidate") problem, and
the session that produced the acquisition data-engine (PR #15) + corpus-harvest
cache (PR #14) was premised on the latter. A per-miss census refutes both readings.

## Part A — candidate availability (canonical DB, pi-storage)

For all 98 recordings involved in the 58 identity misses, queried
`/mnt/storage/data/db/music_database.db` for download + fingerprint presence:

- **57 of 58 missed GT recordings were downloaded AND fingerprinted** — a fair,
  matchable candidate was present and the model still picked something else.
- **Exactly 1** genuine "no fair candidate": `tlp2853054` (Porter Robinson &
  Madeon – Shelter), a `tlp`-prefixed *Rvmor sided-row* with **no recording in the
  DB at all** (the known SC-only-row / `data-trackid` gap — see
  memory `project_tlp_gap`). The acquisition engine, run against these two sets,
  fixes **one span**.
- **Work-grouping is 100% unpopulated:** `0 / 18,812` works group more than one
  recording (`work_id == recording_id` for every recording). Sibling versions
  ("Roses" ↔ "Roses (Acappella)", "Emily" ↔ "Emily (Remix)") are **not linked**.
  This touches ~2 of the 58 misses — a data-model-integrity gap, not an accuracy
  lever.

**Verdict A:** identity is not a data-supply problem. ~1/58 is acquisition.

## Part B — mechanism decomposition (what the misses ARE)

Classifying each miss by predicted-span duration and how many GT tracks overlap
the predicted window:

| # | Mechanism | Spans | Span-s | % of miss-s | Real lever |
|---|---|---:|---:|---:|---|
| 2 | **SEGMENTATION** — one span (up to 196 s) labelled a single recording over a region where **5–12 GT tracks play** | 10 | **1690** | **69%** | Placement/decode: cut spans at track boundaries |
| 3 | **LAYER/transition** — 2–4 tracks co-present; single label is wrong | 26 | 645 | 26% | Layered / stem-wise **multi-label output** |
| 1 | **micro-fragment** (<5 s) — boundary sliver | 18 | 28 | 1% | Span-boundary snapping |
| 4 | **TRUE single-track discrimination** — clean solo region, unrelated song picked | **4** | **76** | **3%** | Better ID model / more training data |

**Verdict B:** only **4 spans / 76 s (~3%)** are the identity model actually
failing to tell one clearly-playing song from another. The 84% is heavily
contaminated:

- **69%** is **segmentation** — the aligner emits one span per tracklist slot
  (`infer.py:187-188`), and where a slot's window covers a mashup/medley of many
  GT tracks, one recording gets stamped over all of it. This is the placement /
  span-extent wall (the ~75%-of-loss lane) leaking into the identity metric, not
  an identity failure.
- **26%** is **dense transitions/mashups** — 2–4 tracks playing at once. A
  single-label-per-span output **structurally cannot reach 100%** there; it must
  emit multiple recordings per layered moment (the repo's stated *stem-wise
  alignment* design intent, which the output format does **not** yet realize —
  there is zero multi-label/co-present machinery in `infer.py` / `harness/merge.py`).

The 4 true-discrimination misses (for the record):

| set | dur | GT | → picked |
|---|---:|---|---|
| BB11 | 23.9 s | Backstreet Boys – Everybody (Oski & Ap…) | Charli XCX – Break The Rules |
| BB12 | 18.9 s | Martin Garrix & Bebe Rexha – In The Name Of Love (acap) | Chromeo – Jealous (acap) |
| BB12 | 17.7 s | RetroVision – Here We Go | A$AP Rocky – … |
| BB11 | 15.0 s | DJ Snake & Yellow Claw – Ocho Cinco | JAY-Z – Forever Young |

## Consequence for "get identity to 100%"

Ranked by payoff, the road to 100% identity is almost entirely **not** the
identity model and **not** acquisition:

1. **Segmentation (69%)** — decoder must not emit giant single-label spans; cut at
   track boundaries. Same lever as the 75%-of-loss placement/decode wall → one fix,
   two payoffs. **Lives in the trajectory-decoder lane.**
2. **Layered/multi-label output (26%)** — realize stem-wise output so simultaneous
   tracks each get labelled; the scorer must credit any co-present track. A
   **representation/output-contract** change, not a model-accuracy change.
3. **Boundary snapping (1%)** — cheap windowing cleanup.
4. **Identity model (3%)** — 4 spans; more training data helps (the harvest's real
   role), smallest lever.
5. **Acquisition (1 span)** — the Porter Robinson Rvmor gap; one download.

**Bottom line:** the identity model is already ~97% correct on clean single-track
spans. Neither the acquisition engine nor a better MERT head is the lever to 100%
— **span-segmentation and a layered output format are.** Plan:
[docs/superpowers/plans/2026-07-18-identity-to-100-structural-levers.md](../../../docs/superpowers/plans/2026-07-18-identity-to-100-structural-levers.md).

## Caveats

- Bucket-2/3 thresholds (`>=5 GT & >=90 s`; `2–4 GT`) are heuristic; the mechanism
  (one label over many co-present GT tracks) is unambiguous regardless of cutoff.
- Availability was checked *globally* (downloaded + fingerprinted), not per-set
  candidate-pool membership; a few "discrimination" misses could be tokenizer/
  materialize pool gaps. The picks are other real tracks, consistent with a
  populated pool; a per-set pool check is one further query if certainty is needed.
- Replication scored 78% / 83% vs the canonical 83% / 84% (a small overlap-rule
  delta); the miss *set* is correct and the 4-vs-54 split is robust.
