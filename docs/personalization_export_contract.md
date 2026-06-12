# Personalization export contract (producer → learning repo)

**Status:** draft v0 (2026-06-12). The frozen data seam between this repo (the
DJ-mix alignment/analysis pipeline) and the future **learning codebase** that
trains the step-2 generation pretrain ([[project_endgoal_mix_generation]]).

This repo is the **producer**: it scrapes, tokenizes, embeds, and exports a
read-only bundle. The learning repo is the **consumer**: it reads the bundle and
trains an HRM-Text conditional model ([[project_hrm_text_taste_pretrain]]). The
consumer imports **nothing** from `core/ analysis/ labeling/ tokenizer/` — it
reads only the export. When this contract stops changing, the consumer half lifts
out into its own repo with `git filter-repo`; until then both live here so a
schema change is one PR, not a cross-repo handshake.

---

## 1. Token space (the load-bearing decision)

A **token = one whole-song MERT embedding** ([[project_pretrain_whole_song_tokens]]):
mean-pooled over the track's measures, **continuous, not VQ**. VQ/codebooks are
excluded deliberately — the info-dynamics work found a codebook *hides* the
structure that the continuous signal exposes ([[project_information_dynamics_bb12]]).
The consumer feeds embeddings through a learned linear projection into the model
dim ("soft tokens").

Two sources of embeddings coexist **in the same space**:

| Source | Table | Key | Role |
|---|---|---|---|
| DJ-set tracklist tracks | `track_mert_measures` → pooled | `recording_id` (via `track_audio`) | **response** tokens |
| Listener prior likes | `sc_track_mert` | `sc_track_id` | **prefix** tokens |

**Unification is by `mert_version`, not by id.** The same song embedded from a
SoundCloud rip vs the catalog yields near-identical MERT vectors, so the embedding
space itself bridges the two — *no hard `sc_track_id ↔ recording_id` join is
required to train*. The hard constraint is that **both sides use one pinned
`mert_version` (1024-dim / MERT-330M)**. The catalog currently holds a mixed
768/1024 population; the 768 subset is **out of contract** and must be re-embedded
at 1024 before it can supply response tokens.

Id-resolution (`sc_track_id ↔ recording_id`) is still wanted, but only as a
**dedup / eval** nicety (vocabulary stats, leakage checks), never as a training
gate.

---

## 2. Exported artifacts

Ship one **read-only `personalization_export.db`** (SQLite — lowest friction; the
whole project is already SQLite) with a `manifest` row carrying `schema_version`
and the pinned `mert_version`. Tables:

### `token_catalog`  — the embedding vocabulary (continuous)
```
token_id      TEXT   -- stable id: 'rec:<recording_id>:<stem>' or 'sc:<sc_track_id>'
source        TEXT   -- 'catalog' | 'soundcloud'
recording_id  TEXT   -- nullable (set for catalog; for sc only if resolved)
stem          TEXT   -- regular | acappella | instrumental
mert_version  TEXT   -- must equal manifest.mert_version
dim           INT    -- 1024
embedding     BLOB   -- float16[dim], whole-song mean-pool
```

### `set_sequences`  — the response targets (one row per token-appearance)
Projection rule: **one token per contiguous mix-appearance of the same song+stem**;
a mix-time gap starts a new row; loops collapse inside a row; reprise = two rows
([[project_pretrain_whole_song_tokens]]). Ref-time/segment detail is intentionally
dropped.
```
set_id          TEXT
seq_idx         INT    -- order within the set
token_id        TEXT   -> token_catalog
stem            TEXT
mix_start_s     REAL   -- coarse mix-time extent (kept)
mix_end_s       REAL
concurrency_grp INT    -- shared id for simultaneously-layered tokens (mashup stack)
```

### `taste_timeline`  — per-user like history (raw, timestamped)
```
user_id       TEXT
liked_at      TEXT   -- ISO; the causal-cut clock
sc_track_id   INT
token_id      TEXT   -> token_catalog ('sc:<sc_track_id>', null if not yet embedded)
```

### `engagement_cuts`  — per-(user, target-set) causal cutoff
```
user_id       TEXT
target_set_id TEXT
cut_at        TEXT   -- user's liked_at of the set's own SC track (preferred)
cut_source    TEXT   -- 'engagement' (~83% for BB12) | 'release_date' (fallback)
```

### `manifest`
```
schema_version TEXT
mert_version   TEXT  -- the single pinned model; everything above must match
built_at       TEXT
```

> `user_prior_vectors` (existing) is a **pooled mean** per user — keep it for
> clustering/retrieval, but it is NOT the prefix. The HRM-Text prefix is the
> **unpooled bag** of `token_catalog` rows for the user's pre-cut likes.

---

## 3. Objective binding (what the consumer builds)

For each `(user_id, target_set_id)` with the target in `set_sequences` and a
`cut_at`:

- **prefix `x_q`** = bag of `token_catalog.embedding` for the user's
  `taste_timeline` rows with `liked_at < cut_at` (PrefixLM → **bidirectional,
  no loss**; unordered bag).
- **response `x_a`** = `set_sequences[target_set_id]` ordered by `seq_idx`
  (**causal, loss here**), with `stem` + `concurrency_grp` as side features.
- **loss** = response-only, `−log P(x_a | x_q)`, PrefixLM mask — the HRM-Text
  recipe verbatim, **except the response head is retrieval/contrastive (InfoNCE
  over `token_catalog`), not a softmax over a fixed vocab**, because the token
  space is continuous and open ([[project_aligner_attention_design]]: "songs via
  retrieval not softmax").

Response diversity comes from mining **every** `(user, set)` pair in a user's
timeline, not just the 3 fully-enriched mixes — see the bottleneck note in
[[project_hrm_text_taste_pretrain]].

---

## 4. Open gaps the contract makes explicit

1. **MERT coverage ≈ 0.3%** of referenced tracks are embedded. `token_catalog` is
   near-empty until the embed job runs; this is the first producer task.
2. **768 vs 1024 split** — re-embed the 768 catalog subset at the pinned version.
3. **sc↔recording resolution** — optional, for dedup/eval only; not a blocker.
4. **Cut coverage** — `cut_source='engagement'` for ~83% (BB12); rest fall back to
   release date. Recorded per row so the consumer can weight/filter.

## 5. Versioning & the repo-split trigger

Bump `schema_version` on any table change. **Cut the learning repo** when:
`token_catalog` schema is frozen + the export builder is stable + HRM-Text
training wants its own cadence/deps. That is verbatim `taste_prior`'s promote
criterion ("corpus join works, MERT priors stable, export API defined").
