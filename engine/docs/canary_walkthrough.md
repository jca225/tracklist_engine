# Canary smoke — code walkthrough

Read this when you open the engine and want to know **which file does what**.
It is a map, not a line-by-line narration.

Run it:

```bash
cd tracklist_engine/core
make canary-smoke
```

Spine reminder: **Propose (Python) → Decide → Promote → Act (Rust)**.

---

## What you are proving

On BB11/BB12 fixtures (no live DB):

1. Two labeling functions vote on “which stem is this slot?”
2. Rust majority-votes those emissions into beliefs
3. An **independent** canary grades the **title** LF against held-out gold
4. If the gate passes → round is **promoted** and opposite-view pseudo-labels are written

Critical invariant: canary gold is **not** derived from `claimed_stem_lf`.
Gold lives in `fixtures/gold/canary_stem_gold.json`. The graded source is
`title_stem_heuristic_lf`.

---

## Call chain (`make canary-smoke`)

```text
Makefile:canary-smoke
  │
  ├─ pytest tests/test_claimed_stem_lf.py
  │
  └─ make cotrain-round
       │
       ├─ Python  write_vertical_slice(fixtures/gold → staging/)
       │            → staging/vertical_lf_bundle.json
       │
       └─ Rust    dj_migrate run-round
                    --bundle staging/vertical_lf_bundle.json
                    --out staging/cotrain_round_report.json
                    --out-pseudo staging/cotrain_pseudo_labels.json
                    │
                    └─ assert report.round.status == "promoted"
                       and report.gate.passed
```

`make vertical-lf` is the shorter cousin: same Propose bundle, then
`pws-fit` only (Decide + canary report, **no** Act / pseudo-labels).

---

## Step map (intent → file → function)

### 0. Place-in database (fixtures)

| What | Where |
|------|--------|
| Slot rows (title, claimed_stem, …) | `fixtures/gold/inventory.json` (or `*_slots.json`) |
| Held-out canary gold | `fixtures/gold/canary_stem_gold.json` |
| Staging outputs (gitignored) | `staging/*.json` |

No `music_database.db`. Paths are locators only; they never mint `RecordingId`.

### 1. Propose — Python LFs

| Intent | File | Function |
|--------|------|----------|
| Package the vertical slice | `python/sensors/.../lfs/claimed_stem.py` | `write_vertical_slice` |
| Vote from DJ-claimed stem field | same | `run_claimed_stem_lf` |
| Vote from title keywords only | `.../lfs/title_stem.py` | `run_title_stem_heuristic_lf` / `guess_stem_from_title` |
| Load independent gold | `.../lfs/canary_gold.py` | `load_canary_stem_gold` |
| Shared emission shapes | `.../lfs/types.py` | `Emission`, `InferenceUnitOut` |

`write_vertical_slice` concatenates both LFs’ emissions, attaches
`canaries` (title LF + expected accuracy) and `gold_by_unit`, writes
`staging/vertical_lf_bundle.json`. It does **not** Decide or Promote.

Abstain rule: missing `claimed_stem` or no title keyword → `abstained: true`,
never invent a stem.

### 2. CLI boundary — Rust migrate

| Intent | File | Function |
|--------|------|----------|
| Subcommand dispatch | `crates/dj_migrate/src/main.rs` | `Commands::RunRound` / `PwsFit` |
| JSON → kernel types | `crates/dj_migrate/src/pws_fit.rs` | `load_bundle` |
| Fit + gate only | same | `run` (`pws-fit`) |
| Fit + gate + Act | `crates/dj_migrate/src/cotraining_cli.rs` | `run` (`run-round`) |

`pws_fit::load_bundle` is shared: both CLIs parse the same Python JSON into
`InferenceUnit` / `EvidenceEmission` / `CanarySpec`.

### 3. Decide — majority vote

| Intent | File | Function |
|--------|------|----------|
| Matrix cell | `crates/dj_kernel/src/pws.rs` | `InferenceUnit` |
| One LF vote | same | `EvidenceEmission` (+ `validate`) |
| Votes → posterior | same | `majority_vote_fit` |
| Accept / abstain rule | same | `decide` |

Toy stand-in for a real Snorkel label model: count non-abstaining votes per
unit, normalize to a posterior, attach entropy + decision.

### 4. Promote — canary gate

| Intent | File | Function |
|--------|------|----------|
| Score each LF vs gold | `pws.rs` | `estimate_source_accuracies` |
| Pass/fail gate | same | `gate_canaries` |

Only the canary `source_name` (`title_stem_heuristic_lf`) must match
`known_accuracy` within `tolerance`. Fail-closed if that source is missing.
`claimed_stem_lf` still votes for majority; it is **not** the canary source.

### 5. Act — co-training (only if promoted)

| Intent | File | Function |
|--------|------|----------|
| Full round | `crates/dj_kernel/src/cotraining.rs` | `run_cotraining_round` |
| Opposite-view labels | same | `eligible_new_pseudo_labels` / `opposite_view` |
| Next-round safety | same | `validate_pseudo_label_for_next_round` |

On reject: report written, **no** `cotrain_pseudo_labels.json`.
On promote: CLI writes pseudo-labels for the opposite view (`symbolic` →
`signal`) so the next round cannot self-consume.

---

## Artifacts after a green smoke

| Path | Role |
|------|------|
| `staging/vertical_lf_bundle.json` | Propose product (units + emissions + gold + canary spec) |
| `staging/cotrain_round_report.json` | Decide/Promote/Act report (`gate`, `beliefs`, `round.status`) |
| `staging/cotrain_pseudo_labels.json` | Act product (only if promoted) |

Open the report first when debugging: check `gate.passed`, `gate.detail`,
and each source’s `est_accuracy`.

---

## Where *not* to look for this path

| Path | Why |
|------|-----|
| Repo-root `alignment.py`, `label.py`, … | Design-time schema notes, not the live engine |
| `tracklist_engine/archive/python_kernel/` | Frozen lab peer — not SoR; see `ARCHIVE.md` |
| Live pi `music_database.db` | Not required; do not mutate from this tree |

---

## Related

- Spine overview: [architecture.md](architecture.md)
- Agent ops / quality gate: [../AGENTS.md](../AGENTS.md)
- Storage / FeatureBlob: [storage.md](storage.md)
- Behavioral contract: [`../../dj_engine_pseudocode.md`](../../dj_engine_pseudocode.md)
