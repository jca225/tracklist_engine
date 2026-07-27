# Spec — the staged pipeline + ablation framework

**Status:** approved design (brainstormed 2026-07-18), implementing.
**One-line:** Formalize the alignment engine into a six-stage typed spine with a
per-stage impl registry and a reproducible ablation runner, so `classical` /
`agentic` / `ml` / `trm` are interchangeable actors that race on one referee.
**Companion spec:** [trm_decoder_bakeoff.md](trm_decoder_bakeoff.md) — the TRM
*actor build* that fills the `trm` skeleton this framework registers. Two specs,
one seam.

---

## 0. Why this exists

The status SSOT ([docs/alignment_status.md](../../../docs/alignment_status.md))
names two co-equal walls — **placement (37% of loss)** and **which-instance
decode-residual (38%)** — and one lever: the **learned actor** (trajectory
decoder + pseudo-label flywheel), with perception **frozen (2026-07-09)**. Today
the three end-to-end actors (`classical`/`agentic`/`ml`) are monolithic
`align_set` functions raced by an ad-hoc `drivers/race.py`. To find the lever to
SOTA we need to **isolate** a stage's contribution and **compose** it back —
reproducibly, against a frozen referee, with the null control always in view.
This framework is that harness. It is the host the TRM bake-off plugs into.

**Honesty boundary (do not oversell).** This framework does **not** re-plumb the
monolithic drivers into six cleanly-separated internal stages — that would be a
100-file rewrite the workspace policy forbids. It defines the six-stage *typed
seam* and *wraps* existing actors behind it. The live pluggable surface today is
**Actor + Bridge**; Perception / Identity / Placement / Assemble are
**single-impl registries** (the seam exists; we do not multiply impls we don't
have).

---

## 1. The stage spine

```
Perception(frozen) → Identity → Placement → Bridge → Actor/Decode → Assemble → PredictedTimeline
```

| Stage | Typed contract (target) | Impls today | Pluggable now? |
|---|---|---|---|
| Perception | `set_id → PerceptionBundle` (frozen sensor channels) | `std` | single-impl |
| Identity | `span → ranked recording candidates (+conf∈[0,1])` | `mert` | single-impl |
| Placement | `span → set_start / coarse placement (+conf)` | `std` (fp+hubert+chroma) | single-impl |
| Bridge | frozen feats → actor input band (trainable projection) | `none`, `linear` | **yes** |
| Actor/Decode | `id+placement+feats → RefSegment trajectory per span` | `viterbi`, `agentic`, `trajectory_decoder`, **`trm` (skeleton)** | **yes** |
| Assemble | per-span results → `PredictedTimeline` JSON | `std` | single-impl |

A **Pipeline** is one impl bound per stage. The four canned pipelines:

- `classical` = `{…, bridge:none, actor:viterbi}`
- `agentic`   = `{…, placement:agentic, actor:agentic}` (agentic is a *placement*
  intervention + actor, not just decode — this is the §4 placement-champion result)
- `ml`        = `{…, bridge:linear, actor:trajectory_decoder}`
- `trm`       = `{…, bridge:linear, actor:trm}` (skeleton: passes through to
  `viterbi`, logs a pointer to the bake-off spec)

Legacy monolithic drivers register as a `MonolithicActor` impl bound to the
`actor` stage with the other stages `inherit` — honest to how `agentic`/`ml`
already refine the `classical` base timeline in `drivers/base.py`.

---

## 2. Two ablation grains (the core new capability)

An ablation study is **isolate-then-compose**.

### 2.1 Stage-isolation grain
Hold every other stage at oracle/GT, swap only the stage under test, score with
that stage's **local** referee. For the Actor stage that referee is the
bake-off's: `path_decode.trajectory_acc` (**strict, no fibers**) with GT
placement given — the status §3 oracle-ceiling, generalized. This is where TRM
is judged. Backed by the existing `trajectory/train.py` (`--split set`,
`--train-set`/`--eval-set`, `--seed`; a `--model {conv,trm}` flag added by the
bake-off).

### 2.2 End-to-end grain
Swap the stage impl, run the whole spine, score via `score_timeline_vs_gt`
(**per-axis, strict AND fiber-aware**). This is `drivers/race.py` generalized —
it confirms an isolated win **survives composition**. `make race` becomes an
alias for a canned end-to-end config.

The runner supports both grains through one `AblationSpec`; a study runs
isolation first (attribute the win), then end-to-end (confirm it composes).

---

## 3. Invariants the runner enforces (not left to discipline)

1. **Baselines are auto-injected, never optional.** Any Actor-stage matrix is
   prepended with `control:raw_argmax` (raw match-sim argmax) and
   `baseline:conv_viterbi`, both **regenerated in the same run**. A row that
   doesn't beat the control is flagged `learned-nothing`. (`train.py` already
   prints the control — the framework surfaces it as a first-class row.)
2. **The referee is frozen.** The runner calls the canonical scorers
   (`trajectory_acc`, `score_timeline_vs_gt`) as-is. There is **no** score-variant
   knob. Modifying the scorer/split to flatter a model is structurally impossible
   through this framework.
3. **LOSO enforced.** The runner rejects any spec where `train_set == eval_set`,
   or where a synthetic span could land in eval (honors
   `synthetic_adapter` train-only + keeps `Dataset.skipped` LOUD). Never
   train+eval on the same mix.
4. **Determinism.** Fixed seed per run; `config_hash` is a stable hash of the
   normalized spec. Same `(config, seed)` → identical metrics.
5. **Ledger ≠ SSOT.** Each run appends one row to `out/ablation_runs.jsonl` keyed
   by `{config_hash, git_sha, timestamp, seed, split, grain}` → metrics. Headline
   numbers still live **only** in `docs/alignment_status.md`; promotion goes
   through the canonical regeneration, never a hand-copy from the ledger.

---

## 4. Architecture — thin orchestration over existing modules

```
alignment/pipeline/
  stages.py      # 6 typed stage Protocols + Pipeline / AblationSpec / result dataclasses
  registry.py    # register(stage, name)->impl ; the 4 canned pipelines ; lookup/list
  runner.py      # matrix expansion, guards, baseline injection, both grains, ledger
                 #   depends on INJECTED Scorer + AlignBackend protocols (see §5)
  adapters/
    classical.py agentic.py ml.py   # wrap drivers/*  (MonolithicActor)
    trm_skeleton.py                  # actor:trm passthrough + bake-off pointer
    baselines.py                     # control:raw_argmax, baseline:conv_viterbi (isolation)
    scorers.py                       # real Scorer impls: subprocess score_timeline_vs_gt / trajectory_acc
  configs/
    race_default.yaml        # end-to-end: classical,agentic,ml over BB11+BB12 (== make race)
    decoder_bakeoff.yaml     # isolation: control,conv_viterbi,trm over the cross-set holdout
```

No existing module is rewritten; adapters are thin wrappers. `pipeline/` is the
only new subpackage.

---

## 5. Testability — inject the heavy backends

The runner's **logic** (guards, matrix expansion, baseline injection, config
hashing, ledger) is pure and must be unit-tested without audio. The heavy work
(producing a timeline, scoring against real GT) is behind two injected protocols:

```python
class AlignBackend(Protocol):
    def run(self, pipeline: Pipeline, ctx: RunContext) -> Artifact: ...   # timeline path OR pred_segs
class Scorer(Protocol):
    def score(self, artifact: Artifact, ctx: RunContext) -> Metrics: ...  # strict/fiber/per-axis dict
```

Production wires the subprocess-backed impls (`adapters/scorers.py`); unit tests
inject fakes. This is how we get real coverage of the guards and injection
without needing pulled sets in CI.

---

## 6. Scope of THIS effort (locked)

- Six typed stage Protocols + `Pipeline`/`AblationSpec` dataclasses.
- Registry with the four canned pipelines; single-impl frozen stages.
- Runner: both grains, all §3 guards, baseline auto-injection, JSONL ledger,
  injected backends.
- Thin adapters wrapping the three existing drivers + `trm` skeleton +
  isolation baselines + real subprocess scorers.
- `configs/race_default.yaml` (reproduces §4 board — the regression anchor) and
  `configs/decoder_bakeoff.yaml`.
- `make race` re-pointed to the runner (alias kept).

**Out of scope (follow-ups):** the real TRM actor (→ bake-off spec); decomposing
monolithic drivers into separate internal stage impls; multi-impl Perception/
Identity/Placement; the generation/forward dual-use (bake-off §6).

---

## 7. Testing plan (TDD)

- **Contract** — every registered impl satisfies its stage Protocol; results are
  valid schema; `confidence∈[0,1]`.
- **Regression golden (anti-bug anchor)** — ported `classical`/`agentic`/`ml`
  reproduce the §4 race board within tolerance (fake-backend unit form checks the
  wiring; a real-data integration form, gated on pulled sets, checks the numbers).
- **Runner guards** — LOSO guard rejects `train==eval`; synthetic-in-eval guard
  rejects; baseline control+conv always present in the row set;
  `learned-nothing` flag fires when a row ≤ control; matrix expansion; stable
  `config_hash`; same `(config, seed)` → identical metrics (fake backend).
- **Ledger** — append-only, one row per run, documented schema.

---

## 8. Pointers

- Driver contract + `finalize`: [drivers/base.py](../drivers/base.py)
  (`EndToEndDriver`, `SetContext`, `finalize`).
- Probe contract: [harness/contract.py](../harness/contract.py)
  (`AlignmentResult`, `RefSegment`).
- Isolation referee: [path_decode.py](../path_decode.py) (`trajectory_acc`) +
  [trajectory/train.py](../trajectory/train.py) (`--split`, control print).
- End-to-end scorer: `score_timeline_vs_gt`; board parser in
  [drivers/race.py](../drivers/race.py) (`_METRICS`).
- TRM actor build: [trm_decoder_bakeoff.md](trm_decoder_bakeoff.md).
- Status SSOT: [docs/alignment_status.md](../../../docs/alignment_status.md).
  Set ids: BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.
```
