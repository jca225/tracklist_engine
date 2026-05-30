# Alignment review — manual labeling pass (2026-05-30)

**Author:** Johnny (manual Ableton labeling)  
**Status:** First structured capture after completing alignment on at least one
full set (mostly done). Ground truth **not yet written back** to pi-storage
(`set_ground_truth` row count = 0 as of 2026-05-30).

**Related docs:**

- North star: [alignment_objective.md](alignment_objective.md)
- Identity / inventory: [identity_and_inventory_plan.md](identity_and_inventory_plan.md)
- Reconcile session: [agent_handoff_reconcile_20260530.md](agent_handoff_reconcile_20260530.md)

---

## Executive summary

Manual labeling surfaced two distinct problem classes:

1. **Upstream audio quality** — wrong version, wrong variant (extended vs
   regular), bad Demucs acap/inst on non-EDM, incorrect Essentia BPM. These
   belong in **ingest**, not the future aligner.
2. **GT fidelity** — micro-level stem warping, section cuts, transition types,
   FX/reverb/quotes. These define what the **GT reader** must parse and what
   the **aligner** must reproduce.

Critically: the audio used for this labeling pass came from a **May 6–9 bulk
download** that ran on an **older codebase**. Several fixes landed *after* that
window (variant-aware YT Music search, identity axes, chromaprint QA,
correction ledger). Many download pain points are **already diagnosed and
partially fixed in git** — re-sourcing affected tracks should be the first
engineering response, not re-litigating the aligner design.

---

## Legacy bulk download — what the labeling pass actually used

### When audio landed (pi-storage canonical DB)

Queried 2026-05-30 via `track_audio.downloaded_at`:

| Day | Rows inserted |
|-----|---------------|
| 2026-05-06 | 1,594 (mostly Spotify spotdl batch: 1,526) |
| 2026-05-07 | 5,775 |
| 2026-05-08 | 6,549 |
| 2026-05-09 | 3,957 |
| **May 6–9 subtotal** | **17,920 / 18,044 total (99.3%)** |
| Post–May 9 | 124 (manual fixes, reconcile REGISTER, spot checks) |

Platform breakdown for the bulk window:

| Platform | Bulk-window rows | Notes |
|----------|------------------|-------|
| `youtube_music` | 16,335 | `redownload_via_ytmusic.py` rescue pass |
| `spotify` | 1,526 | spotdl retry on May 6 only |
| `youtube` / `soundcloud` | 14 | early yt-dlp scrape URLs |

Analysis (Essentia/Demucs/MERT) ran in parallel: `track_analysis` spans
2026-05-06 → 2026-05-29 (819 rows analyzed; 1,656 feature rows).

### What codebase was live during the bulk run

The bulk pass ran under `audio_pipeline/` (pre-split). Key commits **before or
during** May 6–9 that shaped download behavior:

| Date | Commit | What it did | Labeling impact |
|------|--------|-------------|-----------------|
| 2026-05-04 | `b2999a0` | Tier-1 corpus downloader entrypoint | Scoped which sets/tracks entered the pass |
| 2026-05-05 | `c1a2087` | YouTube → spotdl → SoundCloud fallback | May 6 Spotify batch only; spotdl dropped from main chain same week |
| 2026-05-06 | `eba1f22` | YT Music adapter + `redownload_via_ytmusic.py` | Bulk rescue started same day |
| 2026-05-07 | `e2dc6b1` | Unified acquire+replace + tier-1 filter | Accelerated May 7–9 YT Music volume |

Fixes that landed **after** the bulk window (labeling used audio *without* these):

| Date | Commit | What it fixes |
|------|--------|---------------|
| 2026-05-13 | `2cdb892` | **Variant-aware YT Music queries** — Remix/Rework/AltVersion tracks query `full_name` so the remixer qualifier is preserved. Pre-fix: bare `Artist - Title` collapsed distinct remix rows onto the original studio cut (documented in commit message with real track_id examples). |
| 2026-05-28 | `14ff5b5` | `audio_pipeline/` → `ingest/` + `analysis/` split |
| 2026-05-29 | `4ea868f` | Chromaprint variant identity sanity-check (advisory) |
| 2026-05-29 | `6754065` | `acquire_variant.py` v2 — canonical acap/inst ingest |
| 2026-05-29 | `99bc569` | `track_audio_correction` ledger |
| 2026-05-30 | `3b1a69f` | Three-axis `recording` model + `ingest/search_query.py` |
| 2026-05-30 | `3a39d98` | Atomic download→register (`insert_audio_or_reap`) |

**Smoking gun:** four retroactive `track_audio_correction` rows logged on
2026-05-29 explicitly cite the pre-2026-05-13 bare-query bug (Madison Mars,
Carnage Festival Trap, Jay Hardway, Elephante remixes resolved to originals).
42 additional corrections from `reconcile_orphans` on 2026-05-30. The user's
"wrong remix, not captured" observation matches a **known, fixed** failure mode
— not an open mystery.

### Identity / variant state on the bulk corpus

As of 2026-05-30 on pi-storage:

| Signal | Value | Implication |
|--------|-------|-------------|
| `recording.version = remix` | 4,082 | High remix density — pre-May-13 search bug hits hard |
| `recording.version = rework/altversion` | 2,312 | Same |
| `track_audio` with `variant = extended` | **3** | Extended axis essentially unset on downloads |
| `identity_mismatch` view rows | 436 | Claim vs canonical stem/version drift |
| `is_reference = 1` | 27 rows | Reference promotion barely used |
| `track_metadata` rows | 0 | Materialize not re-run post-migration |

The Joyride / Curbi "extended vs regular" cases align with **variant axis never
populated during bulk** — there is no extended-aware search or post-download
duration gate yet.

---

## Findings — downloading

*Maps to [alignment_objective.md](alignment_objective.md) § "Version/variant QA"
and "Stem discovery" (ingest, not aligner).*

### 1. Wrong version (usually remix, not captured)

**Observed:** Nontrivial rate during labeling.  
**Root cause (confirmed):** May 6–9 bulk used pre-`2cdb892` YT Music queries.
~6.4k remixish `recording` rows were sourced when bare-title search was the
default.  
**Already in repo:** variant-aware queries (`ingest/search_query.py` preserves
remixer qualifier; strips acap/inst markers intentionally).  
**Still needed:** Re-run `redownload_via_ytmusic` on affected remix/rework/
altversion rows; chromaprint gate before promoting to reference; log fixes in
`track_audio_correction`.

### 2. Wrong variant altogether (extended vs regular; vocals "off")

**Observed:** Link/content mismatch (Curbi, Bassjackers Joyride); sometimes
vocals sound wrong even when title matches.  
**Root cause:** No extended-aware acquisition; `variant` column is `regular` on
99.98% of rows. YT Music `filter='songs'` prefers album edits.  
**Still needed:** Extended classifier (scrape text + duration vs claim);
duration-ratio gate in `fingerprint.classify()`; manual replace for known bad
slots (replace-track-audio skill).

### 3. BPM incorrect; Tunebat often right

**Observed:** Essentia BPM wrong often enough to require manual warp or Tunebat
lookup during labeling.  
**Root cause:** Essentia on DJ-processed / scraped masters is weak; acapellas
skip Essentia (`stem != 'regular'`).  
**Still needed:** Store **warped BPM in GT** (per-slot, not just canonical
features). Tunebat as optional oracle for ingest QA, not automatic overwrite.

### 4. Poor acap/inst quality (especially non-EDM)

**Observed:** Demucs/UVR stems inadequate for pop/rock acapellas and
instrumentals. YouTube-sourced isolated versions are better but hard to rank.  
**Already in repo:** `acquire_variant.py` v2, UVR chain, phase-cancel path
(`~/aligning/phase-cancel/`), chromaprint advisory for instrumentals.  
**Still needed:** Candidate search → score (chromaprint + duration + optional
mix-overlay SNR) → top-1 pick methodology; do not hard-gate until calibrated on
this labeling pass's known-good/bad list.

### 5. Phantom track — Lux Holmes "Omega"

**Observed:** Track does not exist anywhere online.  
**Implication:** Need **`mix_embedded` provenance** — surgically extract segment
from set audio at GT boundaries, register synthetic `recording`/`track_audio`,
embed, flag substitutability. Reinforces "has set audio" as hard corpus filter
([alignment_objective.md](alignment_objective.md) § Corpus scope).  
**Still needed:** Ingest path for set-extract rows; GT field
`substitute_source: mix_extract`.

---

## Findings — labeling (GT shape)

*Maps to [alignment_objective.md](alignment_objective.md) deliverable A (GT reader)
and manual labeling scope.*

| Observation | GT / schema implication |
|-------------|-------------------------|
| Most work = micro-level stem alignment | `ref_start_s` mandatory; sub-bar precision matters for tolerance setting |
| Song start / change detection hard when tracks similar | Aligner needs section discrimination, not just onset detection |
| Instrumental BPM fixed; acap always warps to match | Aligner B1 (warp acap onto instrumental); GT stores per-layer warp BPM |
| Two Friends splice end-of-song + current section | **`ref_segments`** (already in `ground_truth/schema.py`) — first-class, not edge case |
| Transitions: immediate (same song) vs gradual (volume) | Add `transition_type` + set-level volume envelope to GT reader spec |
| Reverse/regular vocal reverb around drops | Tag as **FX spans**, not song content — otherwise aligner learns tails |
| Quotes between drops | `is_quote` or `skip_training` flag |
| Idea: label acap/inst by separating the BB mix | Complementary: `render_set_stems.py` for set-level continuous stems; **not** a replacement for per-slot reference tracks in GT |

**Schema gaps to close** (not yet in yaml): `warp_bpm`, `transition_type`,
`fx_notes`, `substitute_source`, `is_quote`.

---

## Findings — alignment tooling

### Same name, different stem / instrumental in one tracklist

**Observed:** Bug when two slots share a title but differ on stem axis.  
**Cause:** `_qualifier_suffix()` used to return only the remix parenthetical.  
**Fixed:** compound suffix — e.g. `(Syn Cole Remix) (Acappella)` — in
`pull_set_for_alignment.py`; covered by `tests/test_replace_track_stem.py`.

---

## Ideas backlog (prioritized)

Priorities updated given legacy-download context: **re-source before re-label**.

### Tier 0 — unblock GT write-back

1. **Write back completed GT** via `labeling/write_back_ground_truth.py`.
2. **Fix pull suffix bug** ✅ (`_qualifier_suffix()` compound `(Remix) (Acappella)` — `tests/test_replace_track_stem.py`).
3. **Capture this review** ✅ (this document).

### Tier 1 — fix the bulk corpus (high leverage; code mostly exists)

4. **Re-run `redownload_via_ytmusic`** on remix/rework/altversion rows still
   carrying pre-May-13 audio (chromaprint or manual list from labeling pass).
5. **Promote references** — `is_reference` is effectively unset; pull script
   ranks `manual` first but corpus hasn't been curated.
6. **Log every labeling replace** into `track_audio_correction` (46 rows exist;
   only 4 manual + 42 reconcile as of 2026-05-30).
7. **Deploy identity migration + materialize** per
   [identity_and_inventory_plan.md](identity_and_inventory_plan.md) — resolves
   436 `identity_mismatch` rows and populates `claimed_*` on slots.

### Tier 2 — new ingest capabilities

8. Extended classifier + duration gate.
9. Acap/inst candidate search with quality scoring — ops playbook:
   [stem_discovery_playbook.md](stem_discovery_playbook.md); tools:
   `replace_stem_audio.py`, `track_audio_correction` ledger.
10. Mix-extract path for phantom tracks (Lux Holmes Omega).
11. GT schema extensions: warp BPM, transitions, FX, substitutability.

### Tier 3 — defer

12. **Deerock Discord stem scrape** — manual curated ingest first; automate only
    if volume justifies.
13. Set-level BB separation for labeling — useful for EDA/FX detection, not for
    replacing slot-level refs.

---

## Implications for the Aug 1 plan

| Original assumption | Review adjustment |
|--------------------|-------------------|
| Small hand-labeled GT set gates the aligner | ✅ Still true — but GT quality depends on re-sourced audio |
| Ingest QA is parallel work | **Re-sourcing the May 6–9 bulk is now on the critical path** before scaling labeling |
| Aligner learns warp + key + sections | Labeling confirms **section selection + warp** dominate; key is secondary |
| 20k clean corpus for inference | "Clean" must include version/variant QA pass — raw bulk download is not clean |
| Tolerances unset (±N bars, ±M BPM) | Labeling suggests **bar-level section bounds** and **separate vocal/instrumental warp** |

Do **not** train an aligner on the raw May 6–9 audio without a QA pass. The
failure modes observed in labeling are largely explained by git-dated ingest
bugs, not by alignment algorithm limits.

---

## Evidence commands (repro)

```bash
# Download date histogram
ssh pi-storage 'sqlite3 -header -column /mnt/storage/data/db/music_database.db "
  SELECT date(downloaded_at) AS day, COUNT(*) AS n
  FROM track_audio GROUP BY day ORDER BY day;"'

# Pre-fix correction ledger
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "
  SELECT reason FROM track_audio_correction
  WHERE reason LIKE \"%bare-query%\";"'

# Git: variant-aware fix
git show 2cdb892 --stat

# Git: bulk-era downloader evolution
git log --oneline --format=\"%h %ad %s\" --date=short \
  --since=2026-05-04 --until=2026-05-10 -- audio_pipeline/
```

---

## Open questions

- Which specific set(s) did this labeling pass cover? (Needed to scope re-download
  and GT write-back.)
- Count of slots where manual replace was required vs Demucs-stem workaround vs
  mix-extract — turns qualitative "nontrivial" into a QA metric.
- Should Tunebat BPM be ingested as a secondary feature column, or only consulted
  at labeling time?
