# GT Binding Completeness — Implementation Plan (the final Crush exit)

> **For agentic workers:** execute with superpowers:subagent-driven-development. Steps use `- [ ]`. This plan closes Operation Crush: soundness is already shipped (de-poisoned GT on canonical, decision #15); this recovers the *sound* completeness and makes every rule provable.

**Goal:** Lift GT identity coverage (BB12 66%→~82%+, BB11 57%→~75%+) **without one new wrong label**, by adding only *provably-sound* channels, then re-export → re-write-back.

**Spec (authoritative, read first):** [2026-07-22-gt-identity-binding-completeness-design.md](../specs/2026-07-22-gt-identity-binding-completeness-design.md) — v1 model, v2 hash-ledger, v3 axes+acquisition, v4 certificate rule (definitive). Two Fable adversarial reviews folded in.

**Tech stack:** Python (stdlib + numpy), SQLite (pi canonical), Chromaprint/HuBERT (existing lanes), FLAC STREAMINFO, mp4 mdat.

## Global constraints (verbatim from spec)
- **(S) Per-axis soundness, non-negotiable:** never emit a wrong value on any axis of `id* ∈ Work×Version×Stem×Variant×Remixer`. Getting the work right but stem/variant wrong is a wrong label.
- **Never read `σ`** (placement index) or tracklist rank — proven unsound (v4/§4).
- **Bind-across a generation boundary only with a certificate** (payload-hash equality, or derivation-record+parent-hash, or perceptual+duration fuzz-grade) — **never** by op name (v4.1). retry/rescue is NOT identity-preserving.
- **Stratified soundness:** `content|historical|provenance` sound under their hypotheses; `fuzzy` ε-sound; write-back + eval consume **only sound strata** via `id_source`.
- **Ambiguity hard-abstains across ALL channels** (P7/P13).
- Ledger chain key = `(recording_id, stem, variant, kind)`, `kind∈{master,separated,derived}`; stem/variant are the `track_audio` row's realization values.
- No `σ`, no canonical pi writes without a GATE; migrations apply-exactly-once.

---

## Phase A — Sound unsoundness fixes + axis plumbing (no new data, ships correctness)

### Task A1 — Axis-scoped ambiguity hard-abstain (P7/P13)
**Files:** `labeling/als/content_resolver.py` (`from_entries` ~L44), `labeling/export_als_to_gt.py` (`_load_content_catalog` ~L197). **Test:** `tests/labeling/test_content_resolver_ambiguity.py`.
- [ ] Failing test: a catalog with two entries, same `content_sha256`, same `recording_id`, **different `stem`** → resolving that hash must **abstain** (not last-writer-wins).
- [ ] Change the ambiguity drop to key on `(content_sha256) → {(recording_id, stem, variant)}`; if the set has >1 distinct axis-tuple, drop the key (hard-abstain). `from_entries` **raises/drops** on conflicting duplicates instead of clobbering.
- [ ] Test green; `make check`.

### Task A2 — Plumb `stem`+`variant` through catalog → entry → bind → GT (P16)
**Files:** `labeling/build_content_catalog.py` (SELECT `variant`), `labeling/als/content_resolver.py` (`CatalogEntry`/`ClipIdentity` add `stem`,`variant`), `labeling/export_als_to_gt.py` (`_content_bind` returns `(recording_id, stem, variant, id_source)`; `ClipRow` stamps them). **Test:** extend `tests/labeling/test_content_resolver.py`.
- [ ] Failing test: bind an acappella-stem clip → returned identity carries `stem='acappella'` from the catalog, not derived out-of-band from the manifest.
- [ ] Plumb the two axes end-to-end; the GT row's `claimed_stem` comes from the **content bind** (sound) when content-bound, manifest only as a last resort tagged `id_source`.
- [ ] Test green.

### Task A3 — Catalog stem-derivation correctness: parent filter + `kind` (P14/P15)
**Files:** `labeling/build_content_catalog.py` (`track_stems` join ~L63). **Test:** `tests/labeling/test_catalog_stem_kind.py`.
- [ ] Failing test: an acappella-parent `track_audio` with an `instrumental` residual must **not** produce a catalog `stem='instrumental'` entry for that recording.
- [ ] Add `AND ta.stem='regular'` to the `track_stems` join; add `kind` to each entry (`master` for track_audio rows, `separated` for track_stems). Remove the `_STEM_TO_AXIS` raw-passthrough (P15): unknown stem names (`drums/bass/other`) are **excluded** (component-stem clips hard-abstain, sound).
- [ ] Test green.

### Task A4 — Widen catalog scope to pull-resolution parity (P10)
**Files:** `labeling/build_content_catalog.py` (recs query ~L29). **Test:** same file.
- [ ] Failing test: a set slot carried only by legacy `track_id` (NULL `recording_id`) must appear in the catalog (currently invisible → guaranteed abstain on pristine bytes).
- [ ] Change `recording_id IS NOT NULL` → `COALESCE(recording_id, track_id)` to match `pull_set_for_alignment`'s slot resolution.
- [ ] Test green.

**Phase A acceptance:** re-export BB12 (no new channels yet) — coverage unchanged or up slightly (A4), **zero** stem/variant mislabels, ambiguous hashes abstain. `gt_als_gate` green.

---

## Phase B — Content-history hash ledger (the sound completeness lever) + FLAC PCM key

### Task B1 — `content_history` table + migration (pi)
**Files:** `web_crawler/database/schema.sql`, `scripts/migrations/migrate_content_history.sql`. **Test:** `tests/ingest/test_content_history.py`.
- [ ] Schema: `content_history(id PK, recording_id, stem, variant, kind, track_audio_id, content_sha256, payload_sha256, op, source, generation INT, valid INT DEFAULT 1, ts)`. Chain key = `(recording_id, stem, variant, kind)`.
- [ ] Migration apply-exactly-once (header note); backfill current `track_audio.sha256` as generation 0.
- [ ] Roundtrip test: append a generation, read the chain.

### Task B2 — Never-drop-hash hooks in the acquisition path
**Files:** `scripts/replace_track_audio.py` (`_delete_old_row_if_exists` — append old sha256 to `content_history` before delete), stem regeneration (`analysis/…` / Mac re-stem — hash stems **before** overwrite). **Test:** `tests/ingest/test_history_on_replace.py`.
- [ ] Failing test: a replace appends the retired row's sha256 to `content_history` (op=`replace`) before the row is deleted.
- [ ] Wire the append; stamp denormalized `(version, stem, variant, kind, op, source)`.

### Task B3 — Certificate-gated bind-across in catalog emit (v4.1) + FLAC PCM key (v2.3)
**Files:** `labeling/content_hash.py` (+`flac_pcm_md5`), `labeling/build_content_catalog.py` (emit historical generations with `generation` + certificate class). **Test:** `tests/labeling/test_flac_pcm_md5.py`, `tests/labeling/test_history_catalog.py`.
- [ ] `flac_pcm_md5(path)` reads STREAMINFO decoded-PCM MD5; verify non-null; test on a fixture FLAC.
- [ ] Catalog emits historical-generation entries **only** with a certificate: payload-hash equality (mdat/FLAC-PCM) OR derivation-record+parent-hash. Entries stamp `id_source∈{content,historical-content}` + certificate kind.
- [ ] **retry/rescue generations are NOT emitted as sound** unless they carry the perceptual+duration certificate (Phase D) — they stay abstain-eligible, not silently bound.
- [ ] Tests green.

### Task B4 — Invalidation / tombstone events (v4.2, P17)
**Files:** `ingest/corrections.py` (on `relink`/`detach`, append a `content_history` tombstone `valid=0` for the moved generation), export reads `valid=1` only. **Test:** `tests/ingest/test_history_tombstone.py`.
- [ ] Failing test: a `relink` correction tombstones the old generation; the exporter refuses to bind a clip to a tombstoned generation.
- [ ] Wire it; re-selection-for-cause path emits tombstone + flags prior GT rows for re-audit.

**Phase B acceptance (GATE — pi read/backfill):** rebuild BB12 catalog with history; re-export; coverage 66%→~80%+, **every** new bind is `content`/`historical-content` (byte-level sound), **zero** new wrong labels. Consistency check: historical binds agree with any surviving content bind on the same clip.

---

## Phase C — Sound provenance for the legacy backlog (pull ledger)

### Task C1 — Append-only pull ledger (v2.5 Thm 3′)
**Files:** `labeling/pull_set_for_alignment.py` (emit `pull_ledger.jsonl`: `(path, τ, sha_written, inode, mtime, pull_generation)`). **Test:** `tests/labeling/test_pull_ledger.py`.
- [ ] The pull records the copy-time hash + generation per destination path (it already records `path→τ`; add the hash + epoch).

### Task C2 — `B_prov` with staleness check (Thm 3′, P8)
**Files:** `labeling/export_als_to_gt.py` (second channel after content/history). **Test:** `tests/labeling/test_prov_channel.py`.
- [ ] `B_prov(c)` binds `ρ(τ)` **only if** the current file is byte- or payload-equal to the ledgered write (else abstain `modified_since_pull`); tag-insensitive name-match is trusted only for same-inode renames.
- [ ] `B_prov` hard-abstains if `τ` participates in any ambiguous key (A1).
- [ ] **Consistency certificate (Thm 4′):** export fails if `content`/`historical` and `prov` disagree on any overlap clip; require a minimum overlap count.

**Phase C acceptance:** prov recovers legacy-backlog clips the ledger can't retro-fill; certificate passes at 100% overlap or the export fails.

---

## Phase D — Per-axis gated fuzzy (last resort, ε-sound) + candidates

### Task D1 — Rival-relative per-axis `B_fuzz` (v4.5)
**Files:** new `labeling/fuzzy_identity.py` (reuse chromaprint + HuBERT-L9 lanes). **Test:** `tests/labeling/test_fuzzy_axes.py`.
- [ ] Selection over same-work catalog rows; margin over each rival **per differing axis**; emit partial identity (`⊥` on axes fuzz can't discriminate). Stem axis = **instrumental-residual-silence** test, not vocal-presence. Version = declared `⊥` under fuzz unless a discriminator passes. Variant = duration ratio with the joint-failure guard (reject when a rival matches on duration too).
- [ ] Calibrate `(τ,δ)` **per T-subclass** on the content∩history-certified set; report measured ε.

### Task D2 — Negative candidate entries (v4.5)
**Files:** `scripts/candidate_vocal_gate.py` / `ingest_candidate_winners.py` (emit rejected candidates' hashes+embeddings as catalog **negative entries**). **Test:** `tests/ingest/test_candidate_negatives.py`.
- [ ] A byte-match to a rejected candidate **hard-abstains**; fuzz must beat the rejected rivals' fingerprints, not clear an absolute bar. Winner/rejected same-sha dupe is benign (documented).

**Phase D acceptance:** fuzzy adds only ε-sound rows, stamped `id_source=fuzzy` with per-axis certification; write-back still consumes sound strata only (fuzzy rows excluded from S-critical denominators unless human-confirmed).

---

## Phase E — Re-export, re-write-back, close Crush

### Task E1 — Fix `id_coverage` + `.als` unescape (P12)
- [ ] `id_coverage` counts the sound strata (content|historical|provenance), not only `content`; `html.unescape` `.als` refs before path match (the `&`-title join leak).

### Task E2 — Re-export both sets + GATE write-back
- [ ] Re-export BB12+BB11; expect BB12 ~82%+, BB11 ~75%+, **zero** new wrong labels (per-axis). Snapshot canonical GT; dry-run `write_back_ground_truth`; **GATE:** apply; read-back verify.
- [ ] Regenerate acquisition-case test counts (they track the GT).

### Task E3 — Final checkpoint
- [ ] `/align-checkpoint`: settled decision #16 "Crush completeness exit — sound multi-channel binding". Numbers → `alignment_status.md` after the RT1 scorer run.

**Crush is closed when:** every GT identity is `content|historical|provenance` (byte-sound) or a per-axis-gated `fuzzy` or an honest `⊥`; coverage lifted with **zero** new wrong labels on any axis; canonical written + read-back verified; the four per-axis assumptions each guarded (guard/audit/negative-entries/duration).

## Out of scope
- Acquisition *policy* (what to fetch). Placement/timing (identity only). Extending `Stem` to component stems (drums/bass/other) — hard-abstain for now.

## Self-review (spec coverage)
v2.4/P7→A1; P13→A1; P16→A2; P14/P15→A3; P10→A4; v2.2 ledger→B1/B2; v4.1 certificate→B3; v2.3 FLAC→B3; v4.2/P17 tombstone→B4; v2.5 prov/Thm3′→C1/C2; Thm4′→C2; v4.5 fuzz→D1; v4.5 candidates→D2; P12→E1; P8 staleness→C2; write-back gate→E2. All v2/v3/v4 corrections mapped.
