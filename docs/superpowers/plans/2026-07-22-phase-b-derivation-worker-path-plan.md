# Phase B continuation — the derivation lever (stem-regen hook + historical emit)

> **Status:** PLAN ONLY — do not build without owner go. This is the deferred
> half of Crush completeness Phase B (PR #80 shipped the foundation). It is the
> piece that actually delivers the **BB12 66%→~80% GT-binding coverage** lift,
> and it instruments **deployed GPU/Mac inference paths** + writes canonical pi
> — higher blast radius, and it auto-deploys via the #77 timer once merged.
> Execute with superpowers:subagent-driven-development **after** an explicit go.

**Spec:** `docs/superpowers/specs/2026-07-22-gt-identity-binding-completeness-design.md`
(v2.2 ledger, v4.1 certificate, v4.6 chain key). **Foundation already merged-ready:**
PR #80 (`ingest/content_history.py`, never-drop-on-replace, FLAC PCM key, tombstones).

---

## Why this was split out of PR #80 (the load-bearing finding)

The historical-generation emit is **inert on currently-available ledger data**:

- The exporter already indexes catalog entries on **both** `content_sha256` and
  `payload_sha256` (`export_als_to_gt.py:212,224`) and runs a second mdat-payload
  bind pass. So:
  - a **payload-equal** historical generation is *already* bound by the current
    entry's payload key → emitting it is **redundant**;
  - a **payload-different** historical generation has **no sound hash
    certificate** → v4.1 demotes it to Phase-D perceptual, must not emit.
- The one genuinely-sound, non-redundant lever is the **derivation certificate**
  (v4.1): a *separated* generation (demucs/roformer stem) binds across to the
  recording's stem axis iff its **parent generation's hash matches** the current
  sound parent master. This is the **26-stem re-separation class** — the bulk of
  BB12's ~31 recoverable false-abstains.
- That certificate needs **derivation data in the ledger** (old stem hash +
  parent link), which comes from the **stem-regen never-drop hook**. No hook →
  no derivation data → historical emit recovers nothing. Hence the two are one
  unit.

**Corollary:** the stem-regen hook has no clean single choke point — the old
stem files are overwritten *before* DB persistence runs, so the hash-before-
overwrite must live in the separation entry points themselves
(`analysis/pipeline.py::run_separation`, `analysis/vast_worker.py`,
`scripts/mac_analyze_loop.py`), which run inside the Vast/Mac inference loops.

---

## Global constraints (carry from spec)
- **Per-axis soundness (S):** a separated stem certifies the (recording, stem)
  axes only; `version`/`variant` come from the parent master's row. A derivation
  bind must inherit the parent's axes, never invent them.
- **Certificate, not op name (v4.1):** bind-across a generation boundary ONLY
  with payload-hash equality OR derivation-record+matching-parent-hash. A
  re-separation *after a parent replace* carries the NEW parent's identity —
  check the parent hash, don't assume.
- **Tombstones win:** a `valid=0` generation (B4) is never a bind target.
- **Auto-deploy discipline:** self-heal any new column/table (CREATE IF NOT
  EXISTS / additive migration); code may land before its migration runs (#73/#77).

---

## Task C0 — schema: record derivation lineage (migration, pi)
**Files:** `ingest/content_history.py::SCHEMA`, `web_crawler/database/schema.sql`,
`scripts/migrations/migrate_content_history_parent.sql`. **Test:** extend
`tests/ingest/test_content_history.py`.
- [ ] Add `parent_content_sha256 TEXT` (and optional `parent_payload_sha256 TEXT`)
      to `content_history` — the sound parent generation a `separated`/`derived`
      row was produced from. Additive, `IF NOT EXISTS`-guarded, drift-guard test
      updated (all three DDL sources).
- [ ] `Generation` dataclass gains `parent_content_sha256` / `parent_payload_sha256`;
      `append_generation_on` plumbs them.

## Task C1 — stem-regen never-drop hook (the deferred B2 half)
**Files:** `analysis/pipeline.py::run_separation` (primary), and the two loop
entry points that call it — `analysis/vast_worker.py`, `scripts/mac_analyze_loop.py`.
**Test:** `tests/analysis/test_stem_regen_history.py`.
- [ ] **Before** a re-separation overwrites `/mnt/storage/stems/{taid}/…`, hash
      any existing stem files (`file_sha256` + `flac_pcm_md5`) and append a
      `kind='separated'` generation per stem, stamped
      `(recording_id, stem=vocals→acappella / instrumental→instrumental, variant
      from parent, op='re-separate', parent_content_sha256=<current parent
      master sha256>)`. Atomic w.r.t. the persistence write where possible.
- [ ] Component stems (drums/bass/other) → **no generation** (no point in
      `Stem={regular,acappella,instrumental}`; P15).
- [ ] The parent master's sha256 is the current `track_audio.sha256` of the
      `stem='regular'` row the separation ran on — capture it at separation time.
- [ ] Self-heal + `TRACKLIST_DISABLE_FK` scratch-DB safe (Vast writes scratch).
- [ ] Guard: if the parent row was replaced since the old stems were written
      (parent hash moved), the old stem's parent link won't match → it will fail
      the C2 certificate (correct — don't emit).

## Task C2 — certificate-gated historical emit in the catalog
**Files:** `labeling/build_content_catalog.py`. **Test:**
`tests/labeling/test_history_catalog.py`.
- [ ] For each `(recording_id, stem, variant, kind)` chain of the set's
      recordings, read `content_history` **valid=1** generations (reuse
      `ingest.content_history.chain(valid_only=True)`).
- [ ] Emit a historical entry keyed on the historical `content_sha256`
      **only with a certificate**:
      - **payload equality** — historical `payload_sha256` equals a current
        sound entry's payload (identity-preserving re-encode/retag). *[Note: as
        established, usually redundant with the live payload key — keep for
        completeness/audit, low yield.]*
      - **derivation** — `kind='separated'` AND `parent_content_sha256` equals
        the current sound parent master's `content_sha256` (the 26-stem lever).
- [ ] Stamp `id_source='historical-content'` + `cert∈{payload,derivation}`.
      Never emit a tombstoned or uncertified generation.
- [ ] Consistency check: a historical bind must agree with any surviving content
      bind on the same clip (Thm 4′ overlap) or the export fails.

## Task C3 — id_source plumbing so historical binds count as sound (folds in E1)
**Files:** `labeling/export_als_to_gt.py`. **Test:** extend
`tests/labeling/test_export_content_identity.py`.
- [ ] `_content_bind` returns `id_source='historical-content'` when the match is a
      historical entry; `ClipRow` stamps it.
- [ ] `id_coverage` (`export_als_to_gt.py:~769`) counts sound strata
      `{content, historical-content}` (not only `content`), so the ≥50% export
      gate doesn't misfire; `html.unescape` `.als` refs before path match (P12).
- [ ] Write-back + eval denominators consume sound strata only (already the rule).

## Acceptance (GATE — pi read/backfill + re-export)
- [ ] Run the C0 additive migration + backfill on canonical pi (idempotent).
- [ ] Trigger one re-separation on a BB12 recording with pre-existing old stems;
      confirm the old stem hash + parent link land in `content_history`.
- [ ] Rebuild BB12 catalog with history; re-export. **Measure** coverage
      (do not assert) → expect a lift toward ~80% driven by the 26-stem class,
      **zero** new wrong labels on any axis, historical binds agree with content
      binds on overlap.
- [ ] Regenerate `docs/alignment_status.md` — the honest post-Crush number.

## Risks / notes
- **Blast radius:** C1 edits the GPU/Mac separation hot path. Land behind the
  self-heal, test on a scratch DB first, and watch the first Vast/Mac run after
  deploy. Coordinate with whoever owns the inference loops.
- **Yield honesty:** the payload-equality certificate is mostly redundant; the
  real lift is the derivation path. If the 26-stem class turns out smaller than
  the spec's census on re-measure, say so — don't back-fit the 80%.
- **Legacy backlog:** already-regenerated stems whose old hash was never
  recorded are NOT recoverable here (the hook is forward-only) — they route to
  Phase C (prov ledger) / Phase D (perceptual), or a one-time on-disk backfill
  that hashes surviving old stem dirs.
