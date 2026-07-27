# PWS Aligner Agent Build-Out — Comprehensive Plan (self-contained handoff)

> **For a fresh agent / new session:** this plan is written assuming ZERO prior
> conversation context. Everything you need is here. Work on branch
> `pws-phase1b-continuous` in the worktree `/Users/johnnycabrahams/Desktop/tracklist_engine-pws1b`.
> Interpreter: `/Users/johnnycabrahams/Desktop/tracklist_engine/venvs/audio/bin/python`.
> Run tests from the worktree root. Commit WITHOUT `--no-verify`.

**Date:** 2026-07-15 · **Author:** John (w/ Claude)

## What this is

We are building a DJ-set alignment engine as **programmatic weak supervision (PWS)**
that is structurally an **agent** ([[project_pws_is_an_agent]]): labeling functions
= sensors, a continuous label model = belief, an actuation channel + human oracle =
effectors, typed abstention = policy. Reasoning core is Bayesian (no LLM); the LLM is
*one high-cost sensor*, the human is the *most-budgeted sensor*. Goal: generalize to
~40,000 sets ([[project_operative_goal]]).

## State already built (branch `pws-phase1b-continuous`, 87 tests green)

- `continuous_model.py` — EM over per-probe Gaussian offset noise (belief core). Gate
  v3 on BB12 = PARTIAL: fixes the prior Dawid–Skene refutation, matches hand fusion,
  but σ collapses under **singleton-match sparsity** (too few co-voting LFs/sets).
- `verifier.py` — calibration tripwire (σ rank-inversion) — the honest diagnostic.
- `decode_bridge.py` / `run_phase1.py --model continuous` — fused-placement output.
- `operations.py` — keylock-vs-varispeed LF. `fx_lf.py` — echo-tail + noise-sweep LFs.
- `tracklist_lf.py` — cue-time placement LF (recording_id-keyed; +0.8s bias; ~1s BB11).
- `llm_lf.py` — LLM plausibility LF (injectable client; self-confidence gates only).
- `actuation.py` — the ACTION arm: typed-abstain → re-acquire queue (NO_DATA/OUT_OF_DOMAIN
  actuate; LOW_MARGIN never; per-recording budget; decision layer only, ingest is effector).
- `fit_corpus.py` — fit-on-unlabeled harness (pool cohort votes → one global fit → grade GT).
- `cohorts/big_bootie.json` + `cohorts/README.md` — 27 BB sets + the BB-only-vs-BB+others
  ablation design. `data/djs/bb_all.json` — the BB gap-fill ingest job.

## Corpus reality (pi-storage, 2026-07-15)

41,492 sets scraped; cue on 100% of slots; **1,016 sets with set audio; 609 runnable
at ≥80% ref coverage**; `track_analysis`=801, MERT=0 (scaling = a compute campaign,
not a query). The 1,016 audio-sets reference 23,685 distinct recordings, **8,655
missing** (925 in the BB cohort). 1,282 corrections already logged.

## Hard lessons (do not repeat)

- **Verify every commit against its report.** A dispatched agent once committed an
  empty file with a false "tests pass" claim; the repo pre-commit runs only a *fast
  test subset*, so it does NOT catch a failing new test. Always run the full new test
  file yourself and read the output.
- **Never delegate an implementation task to yet another agent.** You are the implementer.
- **Detect-then-correct, never blanket-redownload** ([[feedback_correctness_vs_accuracy]]).

---

## Workstream A — Complete the LF army (buildable NOW, no compute, collision-free)

Each is one DSP/logic detector + an abstaining vote, synthetic-tested like the existing
ones (see `fx_lf.py` / `test_fx_lf.py` for the exact pattern: build a synthetic positive
and negative signal, assert the detector fires on one and abstains on the other). Reuse
`OperationVote` from `operations.py` and `AbstainReason` from `votes.py`.

### Task A1 — `loop_periodicity` LF (in `fx_lf.py`)
- Signature: exact bar-boundary self-similarity (a span that loops repeats every N beats
  with near-perfect correlation, unlike a through-composed span).
- Detector: high normalized autocorrelation of the *raw* span at a beat-multiple lag
  (contrast echo-tail which uses the *envelope*; a loop repeats the whole signal, not a
  decaying tail). Emit `OperationVote(lf="loop", label="loop"|abstain)`.
- Tests: synthetic looped signal (tile a 2-bar segment) fires; a non-repeating signal abstains.

### Task A2 — `filter_sweep` LF (in `fx_lf.py`)
- Signature: a global LPF/HPF cutoff sweep (spectral centroid moves monotonically) WITHOUT
  the broadband-noise character of a riser (distinguish from `noise_sweep`: a filter sweep
  keeps the program's tonal content, low spectral flatness; a riser is broadband noise).
- Detector: monotonic centroid trajectory + LOW spectral flatness. Emit `OperationVote(lf="filter_sweep")`.
- Tests: synthetic tone swept through a rising lowpass fires; steady tone abstains; a
  broadband riser is NOT labelled filter_sweep (it's noise_sweep).

### Task A3 — `sidechain_pump` LF (in `fx_lf.py`)
- Signature: periodic amplitude ducking locked to the kick (4-on-the-floor pump).
- Detector: beat-rate amplitude-modulation depth on a sustained band (RMS envelope has a
  strong peak at the beat frequency). Emit `OperationVote(lf="sidechain")`. NB per
  [[project_mashup_grammar_prior]] BB uses LUFS-match not sidechain, so expect low firing
  on BB — that's correct, not a bug.
- Tests: synthetic sustained tone with beat-rate ducking fires; un-ducked tone abstains.

### Task A4 — `claimed_axes_identity` LF (new `identity_lf.py`)
- Signature: the scraped `claimed_version` / `claimed_stem` are an unused, noisy identity
  prior. Emit a soft identity vote from them (e.g. claimed_version=remix → prior toward the
  remix recording). Pure metadata, no audio.
- Contract: `claimed_identity_vote(span_id, recording_id, claimed_version, claimed_stem) ->`
  a vote endorsing the claimed identity with a fixed prior confidence (the label model
  learns its true accuracy). Abstain if claims are absent.
- Tests: present claims → endorsing vote; absent → abstain.

### Task A5 — `ordering_soft` LF (new `ordering_lf.py`) — HIGHER RISK, do LAST
- Today tracklist order is a HARD monotonic constraint (`sequence_decode.monotonic_decode`).
  This LF makes it a SOFT vote instead: a placement prior that the DJ likely played tracks
  in listed order (monotonic set_start), but which can be OUTVOTED by strong audio evidence
  (handles reorders/medleys/B2B — the Slide −746s poisoning, ticket #3).
- Contract: given a span's index and its neighbours' placements, emit a soft monotonicity
  vote. Do NOT remove the hard decode yet — ship the LF alongside and let the label model
  weigh it; only soften the hard path once the LF is validated (guard against regressing
  the BB12 0-backward-steps case).
- Tests: in-order placement → high vote; a backward step → low vote (not a hard reject).

---

## Workstream B — Close the agent loop (buildable NOW, integration)

### Task B1 — Route ALL LF votes into `fit_corpus`
Currently `fit_corpus`/`run_phase1` pool only the 4 audio probes. Extend the vote-loading
so operation LFs (`operations`, `fx_lf`), the cue-time placement LF, the claimed-axes LF,
and the LLM LF all contribute votes the label model fuses. This is what turns "4 probes on
2 sets" into real PWS breadth. Design: a per-span LF-runner that calls every enabled LF and
concatenates their votes; feed the union to `ContinuousLabelModel.fit`. Keep operation/
identity votes (categorical) and offset votes (continuous) on their correct axes — do not
mix a categorical operation label into the continuous offset EM.
Acceptance: fit on BB12 with the full LF set runs; the calibration tripwire reports per-LF
accuracy for the new LFs; document whether adding LFs lifts σ off the floor even on 1 set.

### Task B2 — Ref-quality detectors → `AbstentionEvent`s → actuation queue
Wire `ingest/identity_gate.py` + a duration/preview check + `scripts/corpus_integrity.py`
signals to emit `actuation.AbstentionEvent`s (NO_DATA for missing, OUT_OF_DOMAIN for
wrong-version/preview) so `collect_actuations` produces a real re-acquire queue from the
8,655 gaps + wrong-version suspects. Write the queue to JSON; the ingest pipeline drains it
(do NOT download here). Acceptance: running it over BB-cohort slots produces a deduped,
budget-gated queue JSON with correct kinds.

### Task B3 — Policy layer (reuse `agentic/policy.py`)
Arbitrate per span: accept (label model confident) / actuate (re-acquire, cheap) / escalate
(human oracle, budgeted) / call-LLM (mid-cost). This is the agent's policy over its sensor
hierarchy. Reuse `agentic/policy.py` rather than reinvent. Acceptance: a policy function
maps (label-model posterior, abstention reasons, budgets) → an action per span; tested on
synthetic posteriors.

### Task B4 — Real `LlmClient` (Claude API) behind a flag
The LLM-LF scaffold (`llm_lf.py`) takes an injected client. Add a real client wrapping the
Claude API (see the `claude-api` skill for model ids/params) behind a `--llm` flag, default
OFF. Keep all tests on the fake client (no API cost in CI). Acceptance: `--llm` runs the LLM
LF on a handful of spans; without it, nothing calls the API.

---

## Workstream C — Corpus substrate (COORDINATE / DEFER — needs compute or the ingest agent)

- **C1 Correctness pass** on the 15,030 downloaded refs (identity_gate + corpus_integrity,
  triage suspects incl. pending preview-clip cases, targeted redownload). Ingest-agent turf.
- **C2 Gap-fill** the 8,655 missing (BB cohort first via `data/djs/bb_all.json`; idempotent
  `ingest.main`). Ingest-agent turf — do NOT launch a second download campaign into theirs.
- **C3 Corpus-native vote capture** — decouple `capture_votes.py` from the `~/aligning/`
  manual-GT layout; pull set+ref audio from pi `objects/`, build spans from
  `set_track_slots`+cue. DEFER: the parallel speed agent is reworking `capture_votes`/
  `vast_loop`; building now fights a moving target.

## Workstream D — The ablation (DEFER on compute)

Per `cohorts/README.md`: capture votes on the 27 BB sets → Arm A fit (BB-only) → does σ
lift off the floor (calibration tripwire clears)? Then add a sample of others (Diplo 37,
Armin 33, John Summit 24, Lane 8 12…) → Arm B → does pooling help (scale thesis) or hurt
(→ FABLE instance-conditioning needed)? Pilot 50 sets before the full 609.

## Workstream F — Stem "separation of powers" (buildable after A finishes; HIGH VALUE)

The new operation LFs are currently **stem-agnostic** — they run on generic audio and
their vote carries no stem tag. They SHOULD run over the three SET-audio stems
(`mix_vocals`, `mix_instrumental`, full `mix` — capture_votes already produces all
three) with a **competence table** ("separation of powers"): each LF has jurisdiction
over the stem(s) where it is competent. Three payoffs: (1) correctness — echo on
vocal = throw, filter/sidechain on instrumental = bed, chroma is unreliable on a
vocal-only stem; (2) the stem is a **FABLE instance feature** — the label model learns
per-(LF, stem) accuracy; (3) it MULTIPLIES votes per span → denser co-voting → the
direct fix for the Gate-v3 σ-sparsity that blocks promotion.

### Task F1 — `stem_routing.py` (new)
- `Stem` enum (FULL / VOCAL / INSTRUMENTAL); an `LF → competent-stems` table
  (e.g. echo→{full,vocal}, noise_sweep→{full}, filter_sweep→{full,instrumental},
  sidechain→{instrumental,full}, keylock/varispeed→{instrumental} + acap-repitch→{vocal},
  loop→{full,vocal,instrumental}, HuBERT→{vocal}, chroma→{instrumental,full}, fp→{full},
  cue-time/LLM→stem-agnostic).
- A `stem` tag on votes (wrap votes rather than break the `OperationVote` contract, or
  extend it once the LF-army agent is done — coordinate to avoid collision).
- A runner: given the 3 stem audios + the enabled LFs, dispatch each LF to its
  competent stem(s) and return stem-tagged votes for `fit_corpus`/the label model.
- Route the per-(LF, stem) accuracy into the calibration tripwire so we can see which
  (LF, stem) pairs are trustworthy.
- Caveat to document: mix separation is imperfect (Roformer bleed) → a mix_vocal vote
  carries artifact noise → lower learned per-(LF, stem) accuracy (absorbed, not hidden).

## Workstream E — Promotion

The fork must beat hand-tuned fusion on the BB11/BB12 scorecard (or show σ-identifiability
at scale) before promoting out of `workspaces/`. Until then it stays a fork.

---

## Execution order

1. Workstream A (A1→A4, then A5) — pure breadth, no risk, no compute.
2. Workstream B (B1 first — it's what makes the LFs actually fuse; then B2, B3, B4).
3. Hand C1/C2 to the ingest agent; hold C3/D for the speed agent + a compute go from John.

## Global constraints

- `from __future__ import annotations`, full type hints, frozen dataclasses, pure functions.
- No edits to `alignment/`. No new GT. Verify every commit's tests yourself.
- Each task: write the failing test, verify RED, implement, verify GREEN on the FULL new test
  file, run the whole `pws_aligner/tests/` suite, commit.
