# TRM real pseudo-label flywheel — design spec

**Status:** DESIGN + first buildable step (2026-07-18). The pivot from synthetic
pretraining after the measured sim2real gap. Diagnostics only — headline numbers
live ONLY in [docs/alignment_status.md](../../../docs/alignment_status.md).

> **Motivation (measured, not hypothetical).** The TRM decoder architecture works
> (`attic/EXPERIMENTS.md` "TRM decoder graft — sim2real gap MEASURED 2026-07-18"):
> v0 overfit **0.95**, but **synthetic-only → real** train-fit climbed to **0.87**
> while real-BB eval stayed **flat ~0.09 < 0.306** control — a *sim2real gap, not
> underfitting.* More GPU only memorized synthetic harder. The bake-off names two
> levers: (a) synthetic REALISM, or (b) **the real pseudo-label flywheel with TRM
> as the decoder trained on real pseudo-labels.** This doc scopes (b): train TRM on
> the *real* DJ-set distribution using pseudo-labels the agentic loop produces on
> UNLABELED real sets, so the training distribution *is* the eval distribution by
> construction.

---

## 0. Why this closes the gap (the one-paragraph argument)

The synthetic program fails because `sources + operations → mix` (bb12-lite) is
too clean: no EQ/effects/crowd/transition modeling, so the TRM learns a decision
surface that does not exist in real mixes. Pseudo-labels are drawn from *real*
mixes — the exact distribution the referee scores — so there is no distribution
shift to transfer across. The cost is that pseudo-labels are noisy where the
agentic loop is wrong. The whole design is therefore a **precision-for-recall
trade**: accept only the fraction of predictions the loop is confidently right
about (the resolvable fraction — grid-lock, monotone offset, single-instance
content), train on those, and *abstain* (drop, do not guess) on the rest. This is
sound precisely because the walls the bake-off identifies (placement, which-
instance) overlap the loop's *low-confidence* region — so gating on confidence
removes the same spans that are hard, giving clean supervision on the easy mass
and no supervision (rather than wrong supervision) on the hard mass. The TRM then
generalizes the easy-mass decision surface on the real distribution; it does not
learn the hard tail from pseudo-labels (that still needs GT / the learned instance
selector).

**The failure mode this must avoid** is self-amplification: the loop's systematic
errors becoming training targets, which the TRM then reproduces and reinforces.
Every quality gate below exists to keep systematic error out of the training set,
and the eval protocol (§4) is designed to *detect* amplification if a gate leaks.

---

## 1. Pseudo-label source — the agentic AUTO_COMMIT rung

The pseudo-label source is **already a designed-in concept**, not a new mechanism.
`agentic/policy.py` defines the permission ladder whose top rung is literally
"write pseudo-GT":

```python
class Mode(str, Enum):
    AUTO_COMMIT = "auto_commit"  # write pseudo-GT, log only
    REVIEW      = "review"       # place + flag into the review queue
    SUGGEST     = "suggest"      # propose candidates, commit nothing
    ESCALATE    = "escalate"     # human labeling queue (active labeling)
```

`Ladder.mode(belief)` routes a span by `belief.quality() ∈ [0,1]` (share-of-mass ×
cluster trust; `agentic/belief.py`). **Only `AUTO_COMMIT` spans become
pseudo-labels.** REVIEW/SUGGEST/ESCALATE spans are *dropped from training* (they
are the abstain set — no label is better than a wrong label). The default `auto`
bar is 0.75; it is a tunable that we set from the ACCEPT-precision measurement
(§3, G1), never guessed.

The pseudo-label carrier is the **PredictedTimeline span** the drivers already
emit (`drivers/base.py finalize`; schema in `core/contracts/timeline.py`,
round-tripped through `core.contracts.load_timeline`). A span from
`out/<set_id>_<driver>_timeline.json` — the agentic driver adds `driver_mode` +
`agentic_quality` per span (`drivers/agentic.py`) — carries exactly the fields the
target machinery reads (see §2):

```jsonc
{
  "slot_label": "1w1", "recording_id": "281u6p4x", "claimed_stem": "acappella",
  "set_start_s": 64.785, "set_end_s": 185.745,
  "ref_start_s": 16.0, "ref_end_s": 147.8, "ref_stretch": 0.92,
  "ref_segments": [ {"mix_start_s": 64.78, "ref_start_s": 16.0, "ref_end_s": 50.5}, ... ],
  "confidence": -0.002, "ref_peak": 0.648, "ref_path_conf": 16.583,
  "probe_proposals": {"fp": 1281.3, "lyrics": 64.785, "mert_decode": 4.5},
  "start_source": "agentic:fp+lyrics", "driver_mode": "auto_commit",
  "agentic_quality": 0.81
}
```

**The key structural fact that makes this cheap:** a predicted span's
`ref_segments` list `[{mix_start_s, ref_start_s, ref_end_s}]` is the *same shape*
`path_decode._gt_pieces(row)` reads from a GT row. Within a segment the clip-start
offset `ref_start_s − mix_start_s` is constant; segment boundaries are the DJ's
jumps. That is exactly the piecewise-constant offset trajectory the TRM answer
encodes (`offset_coords`, bake-off §2.3). **A high-confidence predicted span is a
drop-in "row" for `raster_targets` / `TrajectorySpanDataset` — no new target
machinery is needed.** The pseudo-label→TRM-target path is:

```
PredictedTimeline span (AUTO_COMMIT)
  → pseudo_gt_row(span)        # this doc's NEW pure fn: span dict → GT-row-shaped dict
  → raster_targets(row, times, ref_dur_s)     # EXISTING, unchanged
  → encode_offset_labels(ref_pos_s, times, kind, bin_s, vocab)   # EXISTING, unchanged
  → per-frame offset-class labels  → TRM CE loss (trm_offset_ce)
```

`pseudo_gt_row` is the ONLY new data-path code. It is pure and testable in
isolation (§Prototype). Because it emits a GT-row-shaped dict, the *entire*
existing `TrajectorySpanDataset` consumes it unchanged — the flywheel dataset is
`TrajectorySpanDataset(sets=[(sid, pseudo_gt_yaml)], ...)` where `pseudo_gt_yaml`
is a materialized `{set_id, tracks: [pseudo_gt_row(span) for accepted spans]}`.

---

## 2. Target-machinery reuse (what a "row" must carry)

`raster_targets(row, times_s, ref_dur_s)` (`trajectory/targets.py`) →
`_gt_pieces(row)` (`path_decode.py`) reads:

| field | used for | predicted-span source |
|---|---|---|
| `set_start_s`, `set_end_s` | span mix window (`_gt_pieces` s0/s1) | present |
| `ref_segments: [{mix_start_s, ref_start_s, ref_end_s}]` | piecewise ref(mix_t) | present |
| `ref_start_s` (fallback when no segments) | linear-span ref anchor | present |
| `tempo_ratio` / slope | segment slope default | `ref_stretch` (rename) |
| `gain_curve` | NULL (inaudible) frames | **absent in predictions → unity** |
| `unalignable` | abstain (masked placement) | **absent → False** |
| `skip_training` | dataset skip | not set |
| `claimed_stem` | feature routing (acap→HuBERT) + recon_ok | present |
| `slot_label`, `track_id`/`recording_id` | audio resolution | present (see note) |

Two fields are GT-only and correctly *default*: `gain_curve` (the hand fader —
predictions have no fader, so every played frame is audible = unity gain, which is
the honest assumption for a machine-placed clip) and `unalignable` (an
annotator's abstain — a prediction that reaches AUTO_COMMIT is by definition not
abstaining). `pseudo_gt_row` sets these to their defaults and maps
`ref_stretch → tempo_ratio`.

**Audio resolution note.** `TrajectorySpanDataset.__getitem__` resolves span audio
via `resolve_span_audio(aligning, by_tid, mix_stems, mix_full, row)` keyed on
`row["track_id"]`. Predicted spans carry `recording_id`; the pulled set's
`manifest.json` keys tracks by `track_id`. `pseudo_gt_row` must therefore also
carry a `track_id`. The drivers already know it per span (the classical timeline
resolves audio); at materialization time we pass it through from the timeline / the
set manifest. This is a wiring detail handled at materialization, not in the pure
offset encoder — which is why the prototype (§Prototype) is testable with no audio.

---

## 3. Quality gates — keeping systematic error out of the training set

Layered, cheapest-first. A pseudo-label must pass ALL of them.

**G0 — AUTO_COMMIT only.** `Ladder.mode(belief) == AUTO_COMMIT`
(`span["driver_mode"] == "auto_commit"`). Primary gate; everything below sharpens.

**G1 — ACCEPT-precision calibration (sets the G0 bar).** The `auto` threshold is
NOT guessed. On the two GT sets we can *measure* clean-rate per rung: run the
agentic loop, bucket spans by `quality()`, compare each rung's placement/identity
to GT, and pick the `auto` bar where the AUTO_COMMIT bucket's clean-rate clears a
target (e.g. ≥0.9 placement-within-tol). This is the existing ACCEPT-precision
gate discipline (memory: *ACCEPT-precision gate (flywheel safety)*). The measured
per-rung clean-rate on GT is the calibration; it transfers to unlabeled sets only
insofar as the probes' precisions do (they are cross-set calibrated — that is what
`belief.py` precisions encode).

**G2 — cross-probe agreement (independence-aware).** `belief.quality(combine=True)`
uses `_combine_trust` (noisy-OR over INDEPENDENT probe groups — fp+chroma grouped
as `content`, surprise grouped with mert). Require the AUTO_COMMIT cluster to
contain ≥2 independent groups OR one ≥0.9-precision probe. Agreement of correlated
probes does not count — this is the guard against a single systematically-wrong
channel (e.g. fp on a wrong diagonal) minting a confident-but-wrong label.
The general driver currently uses `Ladder(combine=False)` for backward-compatible
scoring; the E1 pseudo-label path MUST construct `Ladder(combine=True)` explicitly
and apply a final acceptance predicate: the winning cluster has at least two
`INDEPENDENCE_GROUP`s, or one member probe has calibrated precision ≥0.9.

**G3 — fiber-instance abstain.** `belief.fiber_gate` down-weights a placement that
lands in a self-repeat class (chorus ×2–3, content-undecidable). The
which-instance residual is 38% of loss and is exactly where pseudo-labels would be
wrong-but-confident. Gate: if the accepted span's `ref_start_s` falls in an
ambiguous fiber, route to REVIEW (drop from training) — do NOT pseudo-label the
instance. This is the belief-layer expression of the standing abstain-on-instance
stance and directly prevents the flywheel from amplifying the which-chorus error.
`LiveContext.apply_fiber_gate` currently defaults to `False`; the E1 path MUST
enable it explicitly rather than assuming the general runtime default is safe for
pseudo-label production.

**G4 — segment-shape sanity (cheap structural filter, in the pure fn).** Reject
spans whose `ref_segments` imply an impossible trajectory: negative played
duration (`ref_end_s < ref_start_s` within a forward segment), offset outside the
TRM vocab window `[lo_bin, hi_bin]`, or a degenerate span (`set_end_s <=
set_start_s`, no segments AND no `ref_start_s`). These are decode pathologies, not
placements; they should never be targets. `pseudo_gt_row` returns a `Result`-style
rejection (raises / returns `None`) so the caller counts the drop.

**G5 — abstain-not-guess is the default.** Any span failing G0–G4 contributes NO
label. Coverage (fraction of unlabeled spans that become pseudo-labels) is a
*reported diagnostic*, not something to maximize. Low coverage with high precision
is the correct operating point for a flywheel — it is the opposite failure (high
coverage, leaked errors) that kills it.

**Amplification tripwire.** After each flywheel round, re-measure the AUTO_COMMIT
clean-rate on the held-out GT set (§4). If clean-rate *drops* round-over-round,
the flywheel is amplifying its own errors — halt. (Confident-learning check applied
to the pseudo-label stream.)

---

## 4. Eval protocol — the honest referee (unchanged, non-negotiable)

**Referee:** `path_decode.trajectory_acc` (strict, no fibers) — the same oracle the
bake-off and conv scaffold use. NEVER modified to flatter the model.

**LOSO discipline, three disjoint roles for a set:**

- **GT eval-only** — BB11 (`2nvzlh2k`) / BB12 (`1fsnxchk`). Their hand GT is the
  referee target. A set used for eval is NEVER in the pseudo-label training pool
  (no leakage — the mix the TRM is scored on never contributed a pseudo-label).
- **Pseudo-label train pool** — the UNLABELED sets (§5): BB10, Disco Lines, Murph.
  The agentic loop runs on these; AUTO_COMMIT spans become TRM training targets.
  These are NEVER eval targets (they have no GT).
- **Agentic-probe calibration** — probe precisions in `belief.py` are cross-set;
  G1 calibration reads GT but only to *set the bar*, not to train the TRM.

**Baselines to beat (regenerate in the same run — do not trust stale figures):**
1. **Raw match-sim argmax control** (no-model). Prints alongside every train run.
   TRM must beat it or it learned nothing (bake-off §4).
2. **Synthetic-only → real TRM** (the measured `~0.09` flat curve). The flywheel's
   whole claim is beating THIS: real pseudo-labels > synthetic transfer.
3. **conv+Viterbi scaffold** on the same split (the current non-learned ceiling).

**Success:** flywheel-trained TRM beats the synthetic-only control AND the raw
control on held-out real GT, by a margin surviving the train/eval reverse (train
on {other unlabeled sets + one GT held out for eval both directions}). Beating the
conv scaffold is the promotion bar.

**Kill criteria:**
- Flywheel-TRM ≤ synthetic-only TRM on held-out real → real pseudo-labels bought
  nothing over synthetic; the noise floor of the pseudo-labels ate the
  distribution-match gain. Record verdict in `attic/EXPERIMENTS.md`; stop.
- AUTO_COMMIT clean-rate on GT drops round-over-round → amplification; halt (§3).
- Wins only when pseudo-label and eval sets share a DJ/mix (leak) → not a win.

---

## 5. Unlabeled real sets available NOW (verified on-disk)

`~/aligning/` has 5 pulled sets. Two are GT (eval-only); **three are unlabeled,
fully pulled (mix + tracks + stems), and immediately usable as the pseudo-label
pool** — no GPU, no new download:

| set | set_id | tracks | stems | mix | role |
|---|---|---|---|---|---|
| BB10 | `w1mgcjt` | 216 | 275 | present | pseudo-label pool |
| Disco Lines @ Mammoth | `1rfb0yl9` | 32 | 30 | present (+`mix_vocals.flac`) | pseudo-label pool |
| it's murph @ Club Space | `pwgrrb1` | 73 | 69 | present | pseudo-label pool |
| BB11 | `2nvzlh2k` | — | — | present | **GT eval-only** |
| BB12 | `1fsnxchk` | — | — | present | **GT eval-only** |

BB10 is the richest pool (216 tracks). Disco Lines is a live-set (memory:
*Live-set fingerprint*: placement 16/16, residual = unreleased reworks) — a
different distribution, good for coverage. Murph is another live club set. Note
these are *not registered in `drivers/base.py GT_BY_SET`* (which gates on GT
fixtures) — the flywheel driver path must bypass that registration for the
pseudo-label pool (they have no GT by design). `preflight_set` still validates the
pull. **Prerequisite to verify before the first run:** the agentic `--live` probes
need MERT + fingerprint hit caches for these set_ids on pi-storage; BB10/Disco
Lines have been through infer before, Murph needs a check. This is the one
non-trivial setup cost.

---

## 6. FIRST buildable experiment (smallest thing that tests the claim)

**Claim under test:** *do real pseudo-labels beat synthetic transfer / the raw
control on held-out real?* Smallest experiment that answers it:

**E1 — single-pool, single-direction, offset-target round-trip first.**

1. **Prototype (this doc, DONE):** `pseudo_labels.pseudo_gt_row(span)` — pure fn,
   predicted span → GT-row-shaped dict; `pseudo_span_to_offset_labels(span, ...)`
   proves the full path to `encode_offset_labels`. TDD'd against the
   `raster_targets` / `encode_offset_labels` round-trip (a straight predicted span
   → constant-offset labels → one segment). This proves the data path with zero
   audio/GPU.
2. **Materialize pseudo-GT for ONE pool set (BB10 `w1mgcjt`):** run the agentic
   driver on BB10, keep AUTO_COMMIT spans through gates G0–G4, write
   `out/w1mgcjt_pseudo_gt.yaml` (`{set_id, tracks: [...]}`). Report coverage
   (accepted / total) and per-gate drop counts — this alone tests whether the
   gates leave *any* usable signal (if coverage ≈ 0, the flywheel is starved and
   we learn that cheaply).
3. **Train TRM on the pseudo-GT pool, eval on BB11 GT** (`--split set`,
   train-set = pseudo-BB10, eval-set = BB11), referee = `trajectory_acc`. Print
   all three baselines (§4) in the same run.
4. **Read the three curves** (identity / placement / structure) not one scalar —
   per `alignment_recharacterization`. The flywheel's first target is *placement*
   (37% wall, where synthetic transfer was flat); which-instance is expected to
   stay abstained (G3).

**Expected cost:** step 1 is free (pure fn + unit test, minutes) — DONE. Step 2 is
one agentic `--live` run over BB10 (CPU/MPS; the loop's probes are the expensive
part — MERT/fp cached; budget ~tens of minutes if caches exist, longer if
cache-cold). Step 3 is a TRM train (7M params, CPU/MPS-friendly per the bake-off;
minutes-to-an-hour, no GPU). **No money, no pi-storage writes** — all reads from
`~/aligning/` + local `out/`.

**Expected outcome / risk:** the honest prior is *uncertain but cheap to falsify*.
Upside: real pseudo-labels carry the real placement distribution the synthetic
program lacked, so placement traj-acc on held-out real should move off the flat
~0.09 synthetic-transfer floor toward the control (0.306) and ideally past it. Risk
1 (starvation): the gates are precise enough that coverage on BB10 is too low to
train — mitigated by BB10's 216 tracks and by tuning the G1 bar on the
precision/coverage curve, but a real possible dead-end, caught at step 2 for
near-zero cost. Risk 2 (noise floor): accepted pseudo-labels are clean enough for
placement but their residual error (the loop's own placement MAE) caps what the TRM
can learn — the TRM cannot beat its teacher's precision on the mass it trains on.
This is why E1 targets *placement* (the loop's strong axis via fp+HuBERT) and
abstains on instance (its weak axis) — training on the teacher's strong axis is
where student ≈ teacher is already a win over synthetic-flat.

**Pointers:** `agentic/policy.py` (Mode/Ladder), `agentic/belief.py`
(quality/fiber_gate/masking_gate), `agentic/loop.py` (`Resolution`),
`drivers/base.py` + `drivers/agentic.py` (PredictedTimeline schema + finalize +
`driver_mode`/`agentic_quality`), `core/contracts/timeline.py` (`RefSegment`/
`TimelineSpan`), `trajectory/targets.py` (`raster_targets`),
`trajectory/offset_coords.py` (`encode_offset_labels`), `trajectory/data.py`
(`TrajectorySpanDataset`), `path_decode.py` (`trajectory_acc` referee,
`_gt_pieces`). New code: `trajectory/pseudo_labels.py` (pure `pseudo_gt_row` +
`pseudo_span_to_offset_labels`), tested in
`trajectory/tests/test_pseudo_labels.py`.

---

## 7. E1 implementation design (approved 2026-07-18)

E1 uses an **artifact-first pipeline**. The agentic timeline and pseudo-GT YAML
are explicit, restartable boundaries; training never consumes transient
in-memory labels. This makes starvation, gate leakage, provenance, and set
leakage inspectable before expensive feature extraction.

### 7.1 Unlabeled agentic refinement

Extract the set-independent refinement logic from `drivers/agentic.py` into a
shared function that accepts a base timeline, a stem lookup, live runners, a
ladder, and an event log. The existing supervised driver calls it with GT stem
overrides and preserves current behavior. The E1 path calls it without GT:

- preflight the pulled set and validate the base timeline;
- route stems from timeline/manifest claims;
- run live probes with `Ladder(combine=True)` and fiber gating enabled;
- enforce the G2 independent-group/single-high-precision predicate before a
  resolved span is eligible for pseudo-label materialization;
- emit the normal agentic timeline with `driver_mode`, `agentic_quality`, and
  unchanged ref-content mapping.

Unknown unlabeled sets MUST NOT be added to `GT_BY_SET`; absence of GT is the
point of this path. The unlabeled runner does not score or calibrate against
BB10.

### 7.2 Pseudo-GT materializer

Add a CLI under `trajectory/` that consumes the validated agentic timeline and
the pulled manifest, converts accepted spans with `pseudo_gt_row`, and atomically
writes `out/<set_id>_pseudo_gt.yaml`.

The YAML contains `set_id`, `tracks`, source timeline path and SHA-256,
the ladder thresholds/fiber-gate setting, and per-gate drop counts. `track_id`
is resolved from the timeline first and then by recording/slot identity from
the manifest; unresolved audio identity is a counted rejection, never a row
with `track_id: null`.

The materializer reports accepted/total coverage but does not optimize coverage.
Only AUTO_COMMIT spans surviving G0–G4 are emitted. Generated timelines, event
logs, YAML, and checkpoints remain local artifacts and are not committed.

### 7.3 Training boundary and leakage guard

Extend `trajectory.train` with `--train-yaml PATH`. In set-split mode this
replaces the training fixture only; `--eval-set` continues to resolve through
the hand-GT fixture registry. The command fails before feature loading when:

- the pseudo YAML set ID equals the eval set ID;
- the YAML lacks pseudo-label provenance;
- no trainable rows survive dataset/audio resolution.

The existing synthetic and hand-GT paths remain unchanged. E1 first runs a smoke
fit, then the full TRM fit. The same invocation reports the raw-similarity
control; a paired conv invocation supplies the conv+Viterbi baseline. All
comparisons use the unchanged strict `trajectory_acc` referee.

### 7.4 E1 orchestration and failure behavior

A thin command composes existing stages:

1. preflight BB10 and required caches;
2. produce or reuse the unlabeled agentic timeline;
3. materialize pseudo-GT and print gate accounting;
4. stop with a distinct starvation result if no usable training mass remains;
5. run smoke training;
6. run full conv and TRM comparisons against BB11.

The command performs no canonical DB writes and never mutates pi-storage.
Missing caches fail with an actionable prerequisite message. A failed stage
preserves prior artifacts so the next run resumes from the last valid boundary.

### 7.5 Verification

Tests cover:

- supervised and unlabeled callers sharing one refinement function without
  changing existing driver output;
- explicit G2/G3 safety configuration on the pseudo-label path;
- manifest-backed `track_id` resolution and rejection accounting;
- deterministic, atomic pseudo-GT YAML materialization;
- YAML round-trip through `TrajectorySpanDataset`;
- train/eval set leakage and empty-dataset rejection;
- CLI smoke orchestration with expensive probes and training stubbed.

The runtime experiment is successful only if it completes the BB10 pseudo-train
→ BB11 GT-eval comparison and prints all same-protocol baselines. Any metric
verdict is recorded in the experiment ledger and canonical status workflow, not
hand-copied into this design.
