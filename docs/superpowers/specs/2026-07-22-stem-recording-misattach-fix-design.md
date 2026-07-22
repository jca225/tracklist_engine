# Stem-candidate wrong-recording mis-attach — the full fix (Crush Phase 4)

**Date:** 2026-07-22
**Status:** design, pending implementation plan
**Tracking:** PR #70; Operation Crush Phase 4; acquisition lesson #19, requirement D7.
**Related memory:** `project_stem_cand_wrong_recording_gap`, `project_crush_depoison_content_identity`.
**Related docs:** `docs/acquisition_lessons.md` (#19, §D.7), `docs/superpowers/plans/2026-07-21-crush-exit-program.md` (the sibling *ops* runbook — not this effort).

## Problem

Stem-candidate acappella/instrumental downloads get attached to the **wrong
recording**, and the correction never sticks. This is the user's recurring pain:
*"I manually downloaded the correct version but it was never captured right."*

**Root cause (pinned 2026-07-21 via `track_audio_id 20911`):** `acquire_variant`'s
canonical ingest attaches a downloaded stem to the target recording with **no
same-song check that can refuse**. A `w`-layer's different-song acappella lands on
the base slot's recording — e.g. slot 148w1 "Come On Over Baby" (a Christina
Aguilera karaoke-cover acappella) filed under recording `42wv4vp` "Good Time".

Two structural gaps let this persist:

1. **The guard exists but cannot refuse.** `scripts/acquire_variant.py::_identity_check`
   runs a stem-aware chromaprint comparison *after* the insert and only **prints** a
   verdict — "never blocks the insert" ([acquire_variant.py:226-264](../../../scripts/acquire_variant.py#L226-L264)).
2. **The ledger cannot represent the error.** `track_audio_correction` is constrained
   to `CHECK (axis IN ('version','variant','stem'))`
   ([schema.sql:743](../../../web_crawler/database/schema.sql#L743)). A cross-song
   mis-attach logs as a legitimate `stem/add`; nothing compares the acquired song to
   the target recording's title. Requirement D7 ("sha256 cross-checks against
   wrong-recording attaches") was **documented but never built** — `0505217` is only
   an idempotence `--skip-if-ingested` guard.

**Detection precedent:** the ledger scan (`scratchpad/ledger_scan.py`, prior session)
parsed the acquired song from each `stem/add` correction's `file:` reason field and
compared it to the target recording's title via `labels_overlap`. It found 23
historical mis-attach events — the entire 2026-06-09 11:52–11:55 BB12 batch (lesson
#19's "22 acappella candidates attached to unrelated recordings"). 22 were already
remediated; **1 live survivor was `20911`** (interim: `recording_id` set NULL so slot
148w1 abstains honestly rather than binding to "Good Time").

## Non-goals (scoped out)

- **Same-title / wrong-version** `track_audio` registration errors (right song, wrong
  edit/master). CLAUDE.md already assigns this to a separate ingest/identity effort.
  This fix targets **cross-song** mis-attaches (the documented, high-overlap-failure
  class); the content channel corroborates but is not the primary detector.
- **Full pi deploy.** Canonical pi is 92 commits behind `main` with live parallel WIP;
  applying the migration and running remediation on canonical state is a separate,
  coordinated ops step (gated), not part of the PR that lands the code.

## Scope & sequencing

One spec, four parts. **PR #70 builds parts 1–2** (prevention-first: represent + prevent).
Parts 3–4 are specced here but land as **follow-on plans**.

| Part | What | PR #70? |
|---|---|---|
| 1. Ledger `recording` axis | Schema + `Correction` can represent a wrong-recording detach/relink | **build** |
| 2. Same-song guard | Fail-closed gate in the shared canonical-write path; abstain + log + `--force` | **build** |
| 3. Corpus audit + remediate | Durable `audit_stem_recording_links.py` → triaged CSV → gated re-link / MINT / delete | spec only |
| 4. Manual-capture flow | Content-verified "register hand-downloaded file → slot", reusing the guard | spec only |

## Part 1 — the ledger `recording` axis

The correction ledger must be able to *represent* a wrong-recording mis-attach so the
guard and the audit can record their decisions queryably (not buried in free-text).

**Schema migration** `scripts/migrations/migrate_correction_recording_axis.sql`:
SQLite cannot `ALTER` a `CHECK` constraint, so rebuild `track_audio_correction`
(create new → copy → drop → rename) with:

- `CHECK (axis IN ('version','variant','stem','recording'))`
- `CHECK (action IN ('replace','add','relink','detach'))`
- two new nullable columns: `old_recording_id TEXT`, `new_recording_id TEXT`

The table has **no foreign keys by design** (a correction outlives the `track_audio`
rows it references), so the rebuild is a clean copy. Preserve `correction_id` values
and both indexes.

**`ingest/corrections.py`:** extend the `AXES` / `ACTIONS` module constants, add
`old_recording_id: str | None = None` and `new_recording_id: str | None = None` to the
`Correction` dataclass, and widen the INSERT column list.

**Semantics:**

- `axis='recording'`, `action='detach'` → `new_recording_id` NULL = **abstain** (the
  20911 interim shape: audio kept, recording link nulled).
- `axis='recording'`, `action='relink'` → `new_recording_id` = the correct recording.
- `old_recording_id` = the wrongly-attached recording.

**`track_id`-overload wrinkle (flagged):** `track_audio_correction.track_id` is
`NOT NULL` and historically holds the 1001tracklists `data-trackid`. A recording
mis-attach does not always have a clean tlp id. **Decision:** populate `track_id` with
the affected **recording_id** for `axis='recording'` rows (recording_id is the stable
identity for this axis) and rely on the new `old_recording_id` column for the precise
old link. This overloads the column's meaning per-axis; the spec documents it rather
than adding a nullable `recording_id`-keyed second table (YAGNI for the volume here).

## Part 2 — the same-song guard

A **pure decision function** in `ingest/` (proposed `ingest/same_song_guard.py`):

```
verdict = same_song_guard(
    acquired_title: str,
    recording_title: str,
    stem_axis: str,
    fp_regular: Fingerprint | None,   # None when no 'regular' reference exists
    fp_candidate: Fingerprint | None,
) -> GuardVerdict            # ACCEPT | REFUSE(reason, channel)
```

No DB, no network, no file I/O inside it — fingerprints are computed by the caller and
passed in. This makes the safety logic unit-testable against the BB11/BB12 fixture
corpus with zero network.

**Two channels; REFUSE if *either* fires (fail-closed):**

1. **Title-token (metadata, primary detector).**
   `labels_overlap(acquired_title, recording_title, min_tokens=2)`
   ([labeling/als/identity.py:209](../../../labeling/als/identity.py#L209)) is `False`
   → REFUSE. Recording title from `recording.full_name` (fallback `work.title`).
   Acquired title from yt-dlp `%(title)s` (URL mode) or `--name` / filename stem
   (file mode). This alone catches the documented cross-song class — 20911's "Come On
   Over Baby" vs "Good Time" has zero token overlap.

2. **Stem-aware chromaprint (content, corroboration).**
   Promote the existing `_identity_check` logic: `fp.classify(stem_axis, sim, dur_ratio)`
   against the `regular` reference. When it returns a *different-recording* verdict →
   REFUSE. When **no `regular` reference exists**, this channel cannot run and
   **abstains from the decision** (it does not force ACCEPT and does not REFUSE on its
   own) — the title channel still governs. (Open item resolved below: no-reference does
   not escalate to a separate review queue in v1; it simply leaves channel 2 silent.)

**Behavior on REFUSE (settled policy — abstain + log + override):**

- **No `track_audio` row is written** (abstain). The guard runs post-download,
  pre-insert, inside the shared `replace_track_audio` canonical-write path so *every*
  caller inherits it — `acquire_variant`, `ingest_stem_url`, `apply_stem_matches`,
  `ingest_candidate_winners` — not just direct invocations.
- A `axis='recording', action='detach'` correction is logged (the acquired song, the
  target recording, `new_recording_id=NULL`, the firing channel + reason).
- **`--force` override lane** bypasses the guard with a loud, ledgered note (mirrors the
  HuBERT vocal-gate's existing auto-promote-with-override pattern, `docs/acquisition_lessons.md` §E).
  Default is fail-closed.
- Minting a new recording on REFUSE is **not** the guard's job — that is remediation
  (Part 3). The guard's contract is: never cross-link; abstain honestly.

**Atomicity:** honor requirement D4 ("acquire→register atomic; never delete-before-insert").
The guard decides before the insert, so a REFUSE simply skips the insert — no retire of
an existing row occurs on a pure `add`.

## Part 3 — corpus audit + remediation (spec only, follow-on plan)

Promote the throwaway `scratchpad/ledger_scan.py` into a durable
`scripts/audit_stem_recording_links.py`:

- For each `stem/add` correction (and, optionally, a direct `track_audio` scan), parse
  the acquired song from the `file:` reason field, compare to the target recording's
  title via `labels_overlap`, and rank by title-token disjointness.
- Emit a **triaged CSV**; **log what was NOT auto-classified** (no silent caps).

A gated remediator then resolves each confirmed suspect per the stem-pipeline bands:

- **re-link** — a correct recording already exists → `axis='recording', action='relink'`.
- **MINT** — no existing recording (e.g. no Christina Aguilera "Come On Over" in DB) →
  create one via `ActionKind.MINT_RECORDING` ([core/slot_inventory.py:44](../../../core/slot_inventory.py#L44)),
  since `acquire_variant` only ever attaches to *existing* recordings (the gap).
- **clean-delete** — junk/cover with no slot home → remove the row.

Canonical `track_audio` / `recording` mutation → **GATE** (snapshot first). Non-GT
corpus suspects defer to a tracked backlog; do not block on full-corpus cleanup.

## Part 4 — manual-capture flow (spec only, follow-on plan)

A reliable, content-verified path to register a hand-downloaded file to a slot — the
recurring user pain. Reuses the Part-2 guard so a manual attach cannot cross-link
either: capture acquired title + fingerprint, run `same_song_guard`, and on ACCEPT
register to the slot's recording; on REFUSE, offer the abstain/MINT choice explicitly.
Builds on `acquire_variant`'s existing staging mode + `ingest_stem_url`'s Mac driver.

## Testing & validation

**TDD, guard-first.** Fixtures drawn from BB11/BB12 GT:

- **REFUSE positives** — the 22 already-remediated 2026-06-09 mis-attaches; `20911`
  ("Come On Over Baby" → `42wv4vp` "Good Time") is the headline case. The guard must
  refuse every known cross-song attach on the title channel.
- **ACCEPT negatives** — correct acappella/instrumental attaches from the same GT.
  Measure and report the **over-block rate** on these; a correct attach must not be
  refused (title tokens overlap; content channel abstains or corroborates).

Additional tests:

- `same_song_guard` unit tests over the fixture pairs (pure function, no I/O).
- Ledger migration round-trip: insert a `recording`/`detach` row and a `recording`/`relink`
  row, read them back, assert the new columns + CHECKs.
- Caller integration: `acquire_variant` canonical mode on a REFUSE case writes **no**
  `track_audio` row and **one** `recording`/`detach` correction; `--force` writes the
  row plus a ledgered override note.

**Gates:** `make check` (guardrails + entropy fences) green before push. The entropy
audit's net-subprocess/bare-except fences apply to any new subprocess in the guard path.

## Rollout / ops

- PR #70 lands **code + migration file + tests** on the branch and merges through the
  gate. It does **not** touch canonical pi.
- Applying `migrate_correction_recording_axis.sql` and running Part-3 remediation on
  canonical state is a **separate, coordinated ops step** (pi is 92 commits behind;
  preserve live WIP first) — tracked, gated, not in this PR.

## Out of scope (tracked, not this effort)

- Same-title / wrong-version registration errors (separate ingest/identity effort).
- Full-corpus mis-link cleanup beyond the audit's triaged output (backlog).
- Step-3 audio round-trip law (#37).
