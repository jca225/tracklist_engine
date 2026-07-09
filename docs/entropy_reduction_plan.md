# Entropy-reduction plan — contracts, BB10 completion, Discord backlog

Drafted 2026-07-09. Context: one week produced five independent breakages that were
all the same class of bug — cross-stage artifacts (manifest / GT yaml / timeline /
slot spine) are raw dicts with no owning contract, so writer/reader pairs drift
silently. Meanwhile `labeling/als/` — the one typed, validated, round-trip-tested
layer — broke zero times. This plan extends that treatment to the rest of the
pipeline and closes the concrete data gaps it exposed (BB10, Discord).

Evidence base (all 2026-07-08/09): manifest `slot_label` None-drift; scorer `--gt`
silently joining BB11 predictions to BB12 GT (fake 0% "catastrophe", cost a day
across two sessions); stale `claimed_stem` in timelines; tlp↔recording id bridging
via side files; `acquire_variant` default-promote stealing `is_reference` from the
regular row.

---

## Workstream E — footgun eradication (do first: hours, stops the bleeding)

| # | item | state |
|---|------|-------|
| E1 | scorer `--gt` resolves from `--set-id`, errors on no match | DONE `43c24a6` |
| E2 | `pull_set_for_alignment` emits `slot_label`; existing manifests patched | DONE `43c24a6` |
| E3 | `acquire_variant`: canonical **stem** adds must NOT steal `is_reference` from the regular row — flip the default (promote only same-stem replaces); regression test | TODO |
| E4 | Sweep remaining hardcoded set defaults in the prototype (`train.py DEFAULT_TRAIN_YAML`, any `bb12` literals on CLI defaults); each becomes derive-from-set-id-or-error | TODO |
| E5 | Ad-hoc driver scripts (`reinfer_driver.sh` etc.) converge on `make scorecard` / documented entry points; a driver must never pass fewer args than the entry point requires to be set-safe | TODO |

Exit: no CLI in the repo silently assumes a set.

## Workstream A — the contracts layer (the abstraction)

Grounded in a deep mine of the git history (~200 fix commits categorized) and
the findings corpus. The failure classes, by recurrence:

| failure class | ~fixes | exemplars | killed by |
|---|---|---|---|
| time-base / coordinate confusion | **22** | `ec1d5c8` loop-SECONDS vs warp-beats; `c43fa62` warp anchor shifted GT 430 s; `d01b7ea` master-tempo; `a838cd6` double-counted placement | **A2 time algebra** |
| id-namespace confusion | **18** | `5250ab4` tlp↔recording in scorer; `f122d44` manifest keyed tlp, looked up recording; `420257a` 0/128→126/128 GT ids | **A3 id algebra** |
| stale artifact / regeneration | **18** | `0960565` 117/328 stems lost to stale manifest; `c35fafb` stale timeline claimed_stem; `670c66b` orphan stems | **A1 provenance + freshness** |
| schema / field drift | **16** | `f678f3a` claimed_stem row-text; `794b76b` axis-from-GT; `43c24a6` slot_label None | **A1 records** |
| silent defaults / hardcoded | **14** | `43c24a6` `--gt`=BB12; `fca5061` empty-stems assumption; `317f835` seeded-as-GT | **A1 + E** |
| path resolution | **14** | `9b74bf7` annotator-tag renames; `8d7d714` parents-depth; `3c03445` stem from FILE not folder | **A4 asset locator** |
| unit / convention | **11** | `a98eeb1` tempo_ratio = played-speed not envelope; `310bd65` gain inverted; `24f5a98` clamp vs extrapolate | **A2** |
| identity-axis conflation | **12** | claimed vs canonical, stem-file vs version-tag | existing `RecordingAxes` + A3 |
| multi-agent / multi-machine | **13** | `a9199d1` Mac/pi autoincrement collision; scratch-DB FK off | conventions now, protocol later (non-goal v1) |

The one layer with ~zero fixes: `labeling/als/`. Its transferable design moves
become the **laws** of the contracts layer:

1. **Frozen typed records + one `load()` per artifact** that validates and
   normalizes at the boundary (slot labels, id bridging) — never in consumers.
2. **Validation as values** — generalize als `Diagnostic(code, severity,
   location)` into a shared lib (today's `Result` loses location on deep
   chains); loaders never silently fix up.
3. **`set_id` stamped in every artifact + `join_guard`** — combining artifacts
   from different sets is a load-time error (the BB11-vs-BB12 scoring bug class).
4. **Provenance + freshness fields** on every artifact: `produced_at`, producer
   commit, and input fingerprints (pi query hash, stems-dir state). `is_stale()`
   turns the 18-commit silent-staleness class into loud staleness; the
   disk-truth fallbacks (`0960565`) consolidate here instead of accreting.
5. **Semantics as pure functions, goldens + Hypothesis properties** pin every
   codec (the als fixture-matrix pattern), wired into `make check`.

**The abstractions (in `core/`, which imports nothing upward):**

- **A2 — time algebra (`core/timebase.py`)** — the #1 class. Typed time
  domains (`MixSec`, `RefSec`, `ArrBeat`, `ContentBeat` as NewTypes under the
  contracts' mypy-strict gate) and a first-class **`TimeMap`**: a piecewise
  monotone mapping between two named domains with explicit anchor,
  out-of-domain policy (clamp vs extrapolate — `24f5a98` made this a bug class),
  `.inv()` (als `WarpMarkers` has no inverse today) and composition
  (`mix→arr_beat >> arr_beat→ref`). Warp markers, tempo envelopes,
  master-tempo mapping, span `tempo_ratio`, `ref_segments`, loop-collapse
  images — all become TimeMaps. **A GT span and a predicted span are the same
  object: a partial `TimeMap(mix→ref)` with provenance**; scoring
  (`trajectory_acc`) becomes a measure between two maps; als export renders
  one. Laws: `m.inv().inv() == m`, composition associativity, monotone-segment
  validation (kills backward-jump degeneracies `6c084eb`).
- **A3 — id algebra (`core/contracts/ids.py`)** — `TlpId`, `RecordingId`,
  `TrackAudioId`, `WorkId` NewTypes + a canonical `SlotLabel` (parsed once,
  one normal form — today three lexicons: `001w2` / `1w2` / annotator-tagged).
  One per-set **`IdResolver`** consuming `labeling/fixtures/id_maps/`
  internally and returning `Result` — callers never see raw bridging.
- **A1 — artifact records (`core/contracts/`)** — `GroundTruthSet`,
  `PredictedTimeline`, `Manifest` (with `manifest_version`), `SlotSpine`.
  Includes the **bed-row decision**: manifests emit audio-less rows
  (`local_path: null`, `satisfaction: "unresolved"`) so the als interpreter can
  map every slot (silent drop currently hides BB10's beds). Axis-resolution
  fallback chains (regular-sibling → disk-truth → miss) return a **logged
  reason, never a silent None** (`_vocal_ref_path` class).
- **A4 — asset locator** — one resolver for pi_path / local_path /
  annotator-tag-renamed variants (generalize `match_manifest_for_path`'s
  tag-insensitive matching); the glob-both-slot-forms and parents-depth logic
  live exactly once.

**Phases (incremental; each lands independently):**
- **A0** (½ day): inventory table — artifact × writers × readers × known drift —
  at the top of `core/contracts/README.md`, seeded from the tables above.
- **A1** (1–2 days): records + loaders + join_guard + Diagnostic lib; migrate
  `score_timeline_vs_gt` and `eda/alignment/failure_analysis/`. Highest
  immediate payoff (joins live there).
- **A2** (3–5 days, the big one): `TimeMap` + typed domains; migrate
  `path_decode.trajectory_acc` scoring first (pure, well-tested), then
  `joint_ref_decode` span emission, then als export (`tempo_sec_to_beat`
  becomes a TimeMap constructor). Differential-test against current outputs on
  BB11/BB12 goldens — byte-identical timelines expected.
- **A3** (1 day): id NewTypes + `IdResolver`; migrate infer/pull/scorer joins.
- **A4** (1 day): asset locator; migrate manifest stem resolution +
  `stem_resolve.py` into it.
- **A5** (½ day): guardrails — flag raw `json.load`/`yaml.safe_load` of
  artifact filenames outside `core/contracts/`; property tests in `make check`;
  mypy `--strict` on `core/contracts/` + `core/timebase.py`.

Exit: a new consumer imports a record, not schema-by-folklore; mixing time
domains or id namespaces is a type error; joining sets is a load error;
staleness is loud.

## Workstream B — BB10 to labeling-ready (gates the highest-ceiling lever)

BB10 labeling is John's non-delegable task and the gate for the
instance-selection model ("which chorus", 31% of loss). Current gaps vs the
BB11/12 method: 6 real-id slots never downloaded (`002`, `002w6`, `004`,
`014w3`, `024`, `024w1` — three are **beds**), 4 tlp gap rows, separation
18/132.

- **B1 [JOHN — blocked on permissions]** Revert the Birdy reference flip
  (`UPDATE track_audio SET is_reference=1 WHERE track_audio_id=21151;` +
  `=0 WHERE track_audio_id=23733;`) **before any BB10 re-pull**; then approve/run
  the idempotent one-set download pass (`ingest.main --job-file` with a
  w1mgcjt-only job) for the 6 missing.
- **B2** Separation: `mac_analyze_loop --set-ids w1mgcjt --only-reference`
  (~114 refs ≈ overnight on MPS), queued behind the BB11 separation currently
  running. Same watcher pattern.
- **B3** Re-pull BB10 (delta) → manifest regenerated with `slot_label` + stems;
  `--fetch-candidates` for acappella/instrumental slots; tag pass
  (`inline_tag → relink_als → fill_als_clip_tags`).
- **B4** Birdy "wrong version": John's ear decides what was actually wrong
  (content is verified DDR Radio Edit; leading theory = full remix vs radio
  edit → a **variant** replace, not version). If a full-length remix exists,
  acquire + `replace --axis variant`, ledger it.
- **B5** Exit gate: `make check-inventory SET=w1mgcjt` green → John labels.

## Workstream C — BB11 full-coverage acappella test (in flight)

- **C1** Separation of the 15 missing refs — RUNNING on MPS now.
- **C2** Delta-refresh the BB11 pull; verify stems on pi + locally.
- **C3** Re-run infer → looptrace decode → score (scorer now set-safe).
  Compare acappella trajectory vs the 19% coverage-limited baseline.
- **C4** Write the answer into FINDINGS/HANDOFF: coverage vs regular-sibling
  routing (`_vocal_ref_path`). If still flat at full coverage, the sibling
  path in `infer.py` is the next fix — then A2's manifest record is where the
  stem resolution logic consolidates (with `0960565`'s disk-truth resolver).

## Workstream D — Discord retrieval backlog (bulk, background)

The staged corpus (2,908 files) is ~12% of the channel content the misfire
ledger counted (~23k). Missing items are invisible to every stem search —
the Birdy KYHU instrumental was almost certainly one of them.

- **D1** Locate the retrieval state: where the 23,099-item list came from, what
  "leftovers 20911" points at, whether channel scrape state is resumable.
  (Not in the repo — likely session notes / Discord client side.)
- **D2** Resumable retrieval of the remainder into
  `/mnt/storage/staging/discord_stems/<source>/`, preserving the existing
  layout; archives kept as-is; a flat `INDEX.tsv` (filename, source, size,
  sha256) written as we go so searches never need `find` again.
- **D3** Stem-library matcher (the standing TODO): map staged stems →
  `recording` (`origin=library`, **additive `is_reference=0`** — E3's rule).
  This converts the corpus from a folder into a queryable asset.
- **D4** Re-check Birdy KYHU + all BB-set acappella/instrumental gaps against
  the completed corpus; prefer library stems over separations where they exist.

## Workstream F — latent-bug audit (the taxonomy as a detector kit)

The user's prior is correct: the classes above are still live in the repo. Each
failure class becomes a mechanical detector; the baseline counts (2026-07-09,
attic excluded):

| detector | baseline | class |
|---|---|---|
| hardcoded set-id CLI defaults / constants in live code | **20** (fp_probe, mert_store, fiber_ui, recon_probe, trajectory/train, neuro/*, eda/*, scripts/*, `analysis/canonical_cues` hard-asserts 2nvzlh2k) | silent defaults |
| files reading `manifest.json` raw | **67** | schema drift |
| raw `predicted_timeline` / `ground_truth.yaml` loads | **22** | schema drift / join hazard |
| `parents[N]` relative-depth paths | **118** | path resolution |

- **F1** (with A1/A5): the detectors land in `scripts/guardrails.py` as
  warnings with a committed baseline file; new occurrences fail `make check`
  (ratchet pattern — count can only go down).
- **F2** (rolling): burn down the 20 set-id defaults first (same fix as E1:
  derive-or-error). The 67/22 raw loads convert per-file as A1/A2 migrations
  touch them — no big-bang rewrite.
- **F3** (after A2): a semantic detector pass — grep candidates where mix-time
  and ref-time variables mix in one expression without a TimeMap; human-review
  the hits. This is the class grep can't fully see; the type system takes over
  once A2 lands.
- **F4** (opportunistic): re-run `make scorecard` + `make audit-gt` after each
  workstream lands — behavioral regression net over the abstractions work.

## Online research (in flight)

A web-research pass is running on prior art per class: typed time/units in
Python (pint / phantom-types / NewType+strict-mypy), invertible piecewise
time-map APIs (synctoolbox, librosa.sequence, dawtool), artifact contracts
(pydantic v2 frozen/strict vs attrs+cattrs vs dataclasses+beartype), lightweight
lineage/freshness (DVC vs content-hash fingerprints vs dagster-style asset
concepts), and 2-machine SQLite coordination (write API vs litestream/LiteFS vs
ULID keys). Results get folded into A1/A2 design choices before implementation
starts — decisions deferred until that brief lands are marked (research) above.

## Sequencing

```
now:        E3–E5 + F2 set-id burn-down (hours)  +  C1 running  +  B1 (John: two commands)
next 24h:   C2–C4 after separation  →  B2 queued overnight  +  research brief lands → pin A1/A2 choices
this week:  A0–A1 (records+join guard+diagnostics, F1 ratchet)  →  A2 (TimeMap)
then:       B3–B5 (BB10 labeling-ready)  →  A3–A4  →  F3 semantic pass
background: D1–D4 (Discord), interleaved as pi/network allows
```

Dependencies: B3 needs B1 (reference revert) and benefits from A2 (bed rows).
The instance-selection model needs B5 (BB10 GT) — everything else on this
plan an agent can drive; **B1's two commands and B4/B5's labeling are John's.**

## Non-goals

- No top-level reorg / renames beyond `core/contracts/` (the repo rule stands:
  new top-level folders need explicit justification).
- No open-sourcing or generalizing the contracts layer — internal, like als.
- Don't re-litigate closed experiments (attic/EXPERIMENTS.md, looptrace/NOTES.md).
