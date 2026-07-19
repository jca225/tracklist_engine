# labeling/ — manual ground-truth production

**This module is the *manual* half of the labeling/alignment split** (see the
root CLAUDE.md "Terminology" block). A human aligns a DJ set's stems against the
mix recording in Ableton; the output is ground-truth labels that the (future)
`alignment/` model trains on. Nothing here is automatic — these are the tools
that stage audio for, and bookkeep, the human annotation pass.

Do NOT confuse this with `alignment/` (the algorithmic aligner). Many *names*
here use "align" for legacy reasons (`~/aligning/`, `pull_set_for_alignment`,
the `set_section_alignment` table) — that's the labeling sense, not the model.

## ⚠️ Session provenance — seeded vs hand-labeled (NEVER export GT from the wrong one)

Seeded sessions are **SEEDER OUTPUT** — `seed_als_from_timeline` renders the
aligner's own *predictions* into a Live session for human correction. They are
machine noise until the human has actually worked them; exporting GT from one
feeds the model its own predictions back (this happened with BB10/BB11 on
2026-07-02). Since 2026-07-06 the seeder stamps provenance: output is
**`<SET> SEEDED.als`** (it hard-refuses to write any `* align.als` name) and
the first arrangement locator reads `SEEDED <date> — machine predictions,
NOT GT`. **Pre-stamp seeds floating around may still carry the old
`<SET> align.als` name** — treat any root-level `<SET> align.als` in
`~/aligning/<set>/` as seeder output. The **human hand sessions** live inside
Ableton **`<name> Project/`** folders (e.g. `BB11 align Project/BB11
align.als`, `_backups/.../big bootie 12 labeling Project/big bootie 12
labeling_fast.als`). Other seeder tells: the SEEDED locator, unfilled `[?]`
clip names, flat 60 BPM master tempo (the 1-beat=1-second seeding convention),
clip count ≈ predicted-span count. When in doubt, **ask the annotator** —
coverage/density stats cannot certify provenance.

## The `.als` codec — `labeling/als/`

The Ableton-`.als` ↔ structured-data layer lives in the **`labeling/als/`
sub-package**, structured as an interpreter
([../docs/als_interpreter_plan.md](../docs/als_interpreter_plan.md)):
`cst` (gzip XML ↔ lossless tree) / `models` (AST records) / `read` (CST→AST) /
`semantics` (beat↔sec evaluators, envelopes) / `validate` (diagnostics pass,
CLI: `python -m labeling.als.validate <als>`) / `write` (in-place mutation) /
`roundtrip` (executable laws) — plus the **private half**: `identity`
(manifest/slot/stem) and `tags` (annotator renames). The old
`labeling/als_io.py` is a deprecated re-export shim — import from
`labeling.als`. Internal package (decided 2026-07-02: not open-sourced), but
the layering is enforced: the core seven must not import project-side code
(`als_core_boundary` guardrail). Tests: goldens pin real sessions (skip
off-Mac), committed real-Live fixtures auto-discovered from
[../tests/labeling/fixtures/als/](../tests/labeling/fixtures/als/) (authoring
matrix in its README), Hypothesis properties + mutation fuzz keep extraction
total; extraction skips malformed nodes, `validate` reports them. Manifest matching is tag-insensitive (annotator
`[NNNbpm KK]` renames are stripped on both sides — see
`match_manifest_for_path`).

## Scripts

- **`pull_set_for_alignment.py`** — queries pi-storage's canonical DB over SSH,
  rsyncs the mix recording + per-track stems into
  `~/aligning/<set_id>__<sanitized-title>/{mix.<ext>, tracks/, manifest.json,
  stems/}`, ready to drag into Ableton. Flags: `--check`, `--check-warn` (inventory
  gate via [inventory_check.py](inventory_check.py)), `--fetch-candidates`.
  `make check-inventory SET=<set_id>` wraps `--check`.
- **`tag_aligning_folder.py`** — reads `manifest.json`, queries pi-storage
  `track_audio_features`, injects BPM + Camelot key + feature comment into each
  M4A's iTunes tags so Ableton's browser shows them.
- **`migrate_aligning_naming.py`** — renames files in existing `~/aligning/`
  folders to the current section-number + w-suffix scheme; preserves user tags.

`~/aligning/phase-cancel/` holds phase-cancellation instrumental extraction (see
the `project_phase_cancel` memory; winner config
`adaptive --smooth 0.5 --fft 4096 --cap 4`).

## Consistency model

The `~/aligning/<set>/` folder is a **read-replica of pi-storage**: the pull
script is the only writer, and pi-storage's DB is the source of truth for what
should be there. Two operations keep them consistent:

1. **Re-run the pull = delta refresh.** Rsync runs in archive mode
   (`-aL --partial --inplace`), so re-invoking `pull_set_for_alignment.py
   <set_id>` only transfers files that changed on pi-storage (regenerated stems,
   replaced audio). Unchanged files are skipped.
2. **`--prune` removes orphans.** When pi-storage's view diverges by *removal* —
   a track re-resolved to a different `track_audio_id`, a stem subdir-name
   change, an audio file replaced with a different codec — old local files are
   stale. `--prune` walks `tracks/` and the plan's stem subdirs and deletes
   audio-extension files not in the freshly-rebuilt manifest. Combine with
   `--dry-run` to preview. Gated behind the flag so a fat-finger can't wipe
   in-flight work.

## Annotator rename convention (one-sided, Mac-only)

The human annotator renames track files and stem subdirs to expose tempo + key
inline, e.g. `tracks/030__Going Deeper - Little Big Adventure [126bpm 8B].m4a`
and `stems/001__Carmen Twillie - Circle Of Life [84bpm 6B]/`. This makes
Ableton's clip browser show tempo/key at a glance, dramatically speeding the
workflow. Two known tags:

- `[NNNbpm KK]` — tempo + Camelot key, e.g. `[126bpm 8B]`, `[84bpm 6B]`
- `[no-features]` — flags tracks without Essentia rows on pi-storage so the
  annotator knows to skip them

These renames are **never written back to pi-storage** — canonical names there
stay `{Artist} - {Title}.{ext}`. `--prune` recognizes these tag patterns
(`_USER_TAG_PATTERN` in `pull_set_for_alignment.py`) and treats tagged
files/subdirs as user territory: never deleted. Anything inside a user-renamed
stem subdir (e.g. `phase_cancel_v*.wav`) is left alone because the parent subdir
isn't in the prune's plan-owned set.

Consequence: re-pulling a set deposits *fresh un-tagged copies* of files the
annotator previously renamed. Expected — the annotator either re-runs the rename
pass or ignores the duplicates. There's no automatic re-tag-on-refresh today.

**Automated `[NNNbpm KK]` pass (tag → relink → fill).** Three offline tools
reproduce BB12's tagged state on any pulled set. Run them in order — each later
step depends on the rename the previous one did:

1. `inline_tag_aligning_folder.py <folder> --stems` — renames `tracks/*` files
   and `stems/*` subdirs to append `[NNNbpm KK]`, reading the BPM/key from each
   M4A's already-written `tmpo`/`initialkey` atoms (offline counterpart to
   `tag_aligning_folder.py`, which writes those atoms from pi-storage). Missing
   features → `[no-features]`. Idempotent (skips already-tagged).
2. `relink_als_after_tag.py <folder>` — the rename above orphans the session's
   file references (clips go **offline**); this rewrites `Path`/`RelativePath`
   in every `*.als` from the old name to the tagged name.
3. `fill_als_clip_tags.py <folder>` — the seeder writes each clip's display
   **Name** as `<title> [?]` (a tempo/key placeholder); this replaces `[?]`
   with the real tag read from that clip's own referenced file (stem plays take
   the tag from the parent stem-dir, since `vocals.flac` itself is untagged).

All three edit the `.als` as gunzip→string-substitute→gzip (no lxml
re-serialization, no device/automation changes) and write a one-time backup
(`*.prerelink.bak` / `*.prefill.bak`). Per the `.als` crash history, **still
open the session in Live to confirm** afterward. BPM/key shown are Essentia's,
so acappella values are approximate (next caveat).

Do not Essentia-tag acapellas: vocals-only audio has no intrinsic BPM/key — use
the parent full song's features (see the `feedback_no_essentia_on_acapellas`
memory). Analysis skips Essentia when `track_audio.stem != 'regular'`
([analysis/pipeline.py](../analysis/pipeline.py)). Pull ranks `manual` platform
first after `is_reference`. Remix filenames must carry the full remixer qualifier
from `full_name` (`(SAVI Remix)`, not bare `(Remix)`).

**Manifest identity fields** (per track in `manifest.json`): `version`, `stem`,
`variant`, `axes_key` (`version__stem__variant`). These mirror pi-storage after
identity-axis migration; see root CLAUDE.md.

**Baby rule:** one file under `tracks/` per slot; acappella/instrumental plays
use `stems/vocals` or `stems/instrumental` from the sibling subdir — do not
expect a separate downloaded acappella master unless you explicitly acquired one
(`scripts/acquire_variant.py`).

## Ground-truth write-back (Phase 5 v1)

- Schema: [ground_truth/schema.py](ground_truth/schema.py) — YAML field
  **`claimed_stem`** (`regular` | `acappella` | `instrumental`); legacy
  `version_tag:` in fixtures still loads. P1 adds `slot_label`, `ref_source`,
  `tempo_ratio`, `pitch_shift_semi`.
- Export: `venvs/audio/bin/python -m labeling.export_als_to_gt --als ... --set-dir ...`
  reads the live `.als` + `manifest.json` → `*_ground_truth.yaml` (see
  [../docs/alignment_program_plan.md](../docs/alignment_program_plan.md) P1).
- Anchor-check: `venvs/audio/bin/python -m labeling.anchor_check` compares YAML
  vs fresh `.als` re-export (offline; no pi-storage).
- **Audio audit (run before trusting a GT export):** `make audit-gt SET=<set_id>`
  audio-verifies every clip of the labeling `.als` against the actual mix
  (identity / placement / ref-offset / pitch via chroma matched filter). The
  XML round-trip tests in `tests/labeling/` prove the codec; only this catches
  a session whose *assertions* silently drift from the mix audio. Wraps
  `workspaces/source_detection/als_audit.py`.
- **Release gate (required before write-back / status regen):**
  `make gt-gate SET=<id> ALS=<hand.als> YAML=<fixture.yaml> [ANCHORS=…]`.
  Runs validate → `anchor_check --strict-ref` → `als_audit`, then stamps
  `labeling/.cache/gt_gate/<set_id>.ok.json` bound to the YAML sha256.
  `write_back_ground_truth` refuses DB writes without a fresh stamp
  (`--force-ungated` escape; `--ack-audio-mismatches` for known leftover
  MISMATCH debt). Re-export matching old YAML is not enough.
- CLI: `venvs/audio/bin/python -m labeling.write_back_ground_truth --db ... --yaml ...`
  transactionally replaces [set_ground_truth](../web_crawler/database/schema.sql)
  for that set. Dry-run with `--dry-run`. Uses `slot_label` as DB `label` when
  present. Algorithmic aligner still in `workspaces/`.

## Folder lifecycle

The folder is ephemeral — delete a set once ground truth is written back to
pi-storage via `write_back_ground_truth.py` (or archived YAML is enough for your
workflow). Ableton → YAML export: `labeling/export_als_to_gt.py`.
