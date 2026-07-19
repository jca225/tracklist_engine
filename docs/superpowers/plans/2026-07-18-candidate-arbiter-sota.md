# Candidate Arbiter + Structured Placement SOTA Plan

**Status:** proposed implementation plan  
**Objective:** beat the current agentic placement board without regressing
already-correct spans by learning when to accept a candidate and otherwise
falling back to the frozen baseline.

## Execution checkpoint

Completed on branch `fp-hit-decoder-clean`:

- Phase 0 immutable candidate schema, provenance-stamped JSONL persistence and
  baseline extraction;
- shadow extraction of all probe proposals already serialized in canonical
  timelines;
- evaluation-only candidate oracle and strict two-set oracle race;
- baseline-relative label contract, missingness-aware vectorizer and a
  precision-weighted logistic critic baseline.

The candidate oracle clears the placement gate on both held-out boards, so
existing proposer recall is sufficient for this milestone and a learned
arbiter is earned. The real-only single-set-to-single-set logistic critic does
not transfer, so it must not be integrated into a driver. Current active work
is Phase 1 evidence enrichment plus Phase 2 synthetic hard-negative generation.
The mashup-invariant encoder remains deferred.

This plan does not introduce another probe first. The ridge diagnostic and the
instrumental-FP delta proof showed that existing proposers often contain useful
answers, but their confidence is not calibrated for candidate selection. The
system needs a verifier/ranker before it needs a new representation.

Canonical alignment results and all headline numbers remain exclusively in
[`docs/alignment_status.md`](../../alignment_status.md). Experimental reports
belong under ignored `out/` paths and must not update that document until the
formal regeneration gate passes.

## Load-bearing decisions

1. **Baseline is a first-class candidate.** The model may improve a span or
   return the input placement unchanged. It never has to manufacture a change.
2. **Candidates remain separate.** FP, HuBERT, lyrics, cue, MERT, chroma and
   baseline hypotheses are not collapsed into one `SpanBelief.best()` before
   learning.
3. **Train the critic before the actor.** The first model ranks pinned
   hypotheses; it does not search raw audio end to end.
4. **Synthetic trains; real GT calibrates and evaluates.** Synthetic examples
   provide scale and hard negatives. Leave-one-set-out real GT is the transfer
   gate.
5. **Fail closed.** If the selected candidate does not clear a calibrated
   improvement threshold, retain the baseline byte-for-byte.
6. **Global decoding is second-stage.** Local candidate scores feed a
   tracklist-level decoder; the ranker itself does not learn ordering by
   memorizing set position.
7. **No GT-tuned slot rules.** Direction, displacement or source gates found
   after inspecting a named failure are diagnostic features, never acceptance
   rules.
8. **The mashup-invariant encoder is contingent.** Build it only for cases
   where the frozen candidate bank has no near-GT ridge.

## Target architecture

```text
existing probes
    │
    ▼
top-K PlacementCandidate bank ───── baseline candidate
    │
    ▼
CandidateEvidence materialization
    │
    ├── synthetic exact labels + perturbation negatives
    └── real GT LOSO labels
    │
    ▼
precision-first candidate critic
    │   score = P(candidate improves baseline)
    ▼
calibrated accept / baseline / review
    │
    ▼
tracklist-level structured decoder
    │
    ▼
timeline, strict scorer, regression audit
```

## Public contracts

Create `workspaces/alignment_prototype/candidate_arbiter/`.

### `schema.py`

```python
@dataclass(frozen=True)
class PlacementCandidate:
    set_id: str
    slot_label: str
    recording_id: str
    claimed_stem: str
    source: str
    rank: int
    set_start_s: float
    ref_start_s: float | None
    native_confidence: float | None
    evidence: CandidateEvidence


@dataclass(frozen=True)
class CandidateEvidence:
    baseline_delta_s: float
    ridge_peak: float | None
    ridge_second: float | None
    ridge_margin: float | None
    ridge_prominence: float | None
    ridge_support_s: float | None
    vote_count: int | None
    vote_density: float | None
    runner_up_ratio: float | None
    agreeing_sources: tuple[str, ...]
    independent_groups: int
    cue_delta_s: float | None
    mert_delta_s: float | None
    neighbor_gap_left_s: float | None
    neighbor_gap_right_s: float | None
    active_layer_count: int | None
    repeat_ambiguity: float | None
```

Missing evidence is represented by `None` plus an explicit missingness mask at
tensorization time. Never zero-fill a missing sensor as if it measured zero.

### `labels.py`

```python
@dataclass(frozen=True)
class CandidateLabel:
    candidate_error_s: float
    baseline_error_s: float
    improvement_s: float
    improves_baseline: bool
    within_accept_tolerance: bool
```

Labels are generated only in training/evaluation code. Production inference
cannot import GT loaders.

### `model.py`

```python
class CandidateCritic(nn.Module):
    def forward(self, features, missing_mask, source_id, stem_id) -> Tensor:
        """Uncalibrated logit for P(candidate improves baseline)."""
```

Start with a small MLP or pairwise logistic head. A larger audio encoder is not
part of this phase.

### `select.py`

```python
@dataclass(frozen=True)
class Selection:
    candidate: PlacementCandidate | None  # None means baseline
    probability: float
    reason: str


def select_or_baseline(
    candidates: Sequence[PlacementCandidate],
    probabilities: Sequence[float],
    *,
    accept_threshold: float,
    min_margin: float,
) -> Selection:
    ...
```

Acceptance threshold is fit on training/calibration sets for a precision-first
objective and frozen before held-out evaluation.

## Phase 0 — freeze the board and candidate provenance

**Files**

- Create: `candidate_arbiter/schema.py`
- Create: `candidate_arbiter/io.py`
- Create: `tests/alignment_prototype/test_candidate_arbiter_schema.py`
- Modify: probe result serializers only where provenance is currently lost

**Tasks**

- [ ] Define `PlacementCandidate`, `CandidateEvidence` and JSON round-trip.
- [ ] Give every candidate a stable key:
  `(set_id, normalized_slot, recording_id, source, rank)`.
- [ ] Add a baseline candidate for every span.
- [ ] Validate that candidate materialization never changes a source timeline.
- [ ] Store artifacts under
  `workspaces/alignment_prototype/out/candidates/<set_id>.jsonl`.
- [ ] Add schema-version and producer-SHA metadata.

**Gate**

- Candidate extraction from a canonical timeline is deterministic.
- Round-trip preserves optional evidence and rejects non-finite values.
- Baseline candidate count equals input span count.

**Commit**

`feat(arbiter): add immutable placement candidate contract`

## Phase 1 — expose top-K evidence without changing decisions

**Files**

- Create: `candidate_arbiter/adapters.py`
- Modify: `mix_fp_hits.py`
- Modify: `stem_placement.py`
- Modify: `lyrics_align.py`
- Modify: `refine_ref_offsets.py`
- Test: `tests/alignment_prototype/test_candidate_adapters.py`

**Tasks**

- [ ] Add read-only top-K return types to each proposer. Existing argmax APIs
  remain unchanged.
- [ ] FP adapter records cluster extent, vote count, vote density, runner-up
  ratio and monotonic-decode compatibility.
- [ ] HuBERT/chroma adapters record `neuro.precision.Precision` fields.
- [ ] Lyrics adapter records diagonal support, IDF mass and runner-up margin.
- [ ] Cue/MERT adapters record displacement and agreement features.
- [ ] Build independence-group agreement using the existing grouping in
  `agentic/belief.py`.
- [ ] Materialize candidates in shadow mode during a driver run.

**Gate**

- With shadow output disabled, current timelines are byte-identical.
- Adapter top-1 equals each proposer’s legacy argmax.
- Every candidate can be traced to raw proposer evidence.

**Commit**

`feat(arbiter): materialize top-k probe candidates in shadow mode`

## Phase 2 — construct critic training data

**Files**

- Create: `candidate_arbiter/labels.py`
- Create: `candidate_arbiter/dataset.py`
- Create: `candidate_arbiter/perturb.py`
- Extend: `synthetic_mix/` adapters, not the renderer contract
- Test: `tests/alignment_prototype/test_candidate_dataset.py`

**Real examples**

- Generate candidates from each hand-GT set.
- Match labels using the scorer’s recording-aware GT resolution.
- Split by complete set; never split spans from one set across train and test.
- Add hard negatives:
  - wrong offset near the correct ridge;
  - correct content, wrong repeat instance;
  - sibling recording;
  - wrong stem;
  - wrong warp;
  - neighboring track’s ridge.

**Synthetic examples**

- Reuse `synthetic_mix/generate_v2.py` topology and
  `trajectory/synthetic_adapter.py`.
- Render each clean source under independently sampled:
  unrelated beds, stacked vocals, gain automation, filtering, reverb, noise,
  time stretch, pitch shift, loops and jump cuts.
- Preserve exact transformation labels.
- Generate candidate banks with the same production proposer adapters.
- Include clean and hostile versions of the same source in a grouped split to
  prevent augmentation-family leakage.

**Gate**

- Dataset audit reports counts by set, stem, source, error bucket and
  synthetic/real provenance.
- No candidate derived from held-out GT enters training.
- Perturbation tests prove label sign and repeat-instance handling.

**Commit**

`feat(arbiter): build real and synthetic candidate-ranking corpus`

## Phase 3 — train the baseline-aware critic

**Files**

- Create: `candidate_arbiter/model.py`
- Create: `candidate_arbiter/train.py`
- Create: `candidate_arbiter/calibrate.py`
- Test: `tests/alignment_prototype/test_candidate_model.py`

**Training objective**

For each non-baseline candidate, predict whether it improves the frozen
baseline. Use pairwise ranking plus a calibrated binary loss:

```text
L = BCE(improves_baseline)
  + λ_rank * max(0, margin - score(best) + score(other))
  + λ_abstain * false_accept_penalty
```

False acceptance must cost substantially more than missed improvement. Tune the
weight on training/calibration data only.

**Baselines to race**

- static source precision;
- `SpanBelief.best()`;
- logistic regression over the same features;
- small MLP;
- always retain baseline.

**Calibration**

- Fit temperature or isotonic calibration on a set not used for gradient
  updates.
- Select the acceptance threshold for monotonic precision under abstention.
- Report reliability diagrams and acceptance curves.

**Gate**

- Beats static fusion on held-out candidate ranking.
- Acceptance precision is monotonic as coverage decreases.
- “Always baseline” remains available and is never scored as an error by the
  selector.
- If the small MLP does not beat logistic regression, ship logistic regression.

**Commit**

`feat(arbiter): train calibrated baseline-aware candidate critic`

## Phase 4 — add fail-closed local selection

**Files**

- Create: `candidate_arbiter/select.py`
- Create: `drivers/arbiter.py`
- Modify: `drivers/race.py`
- Test: `tests/alignment_prototype/test_arbiter_driver.py`

**Tasks**

- [ ] Load a frozen critic bundle with feature schema and calibration metadata.
- [ ] Score candidates per span.
- [ ] Require both probability threshold and top-vs-runner-up margin.
- [ ] On missing features, model failure, schema mismatch or low confidence,
  retain the original span byte-for-byte.
- [ ] Copy only placement-owned fields when accepting a candidate; never shift
  ref-content segments.
- [ ] Record candidate provenance and selection reason.
- [ ] Add `arbiter` to the existing driver race.

**Gate**

- A forced abstain produces a byte-identical baseline timeline.
- Candidate acceptance cannot change identity.
- Missing model/cache is a loud baseline fallback, not a partial timeline.
- Strict scorer and contract loader accept the output.

**Commit**

`feat(drivers): add fail-closed candidate arbiter`

## Phase 5 — structured tracklist decode

**Files**

- Create: `candidate_arbiter/structured_decode.py`
- Test: `tests/alignment_prototype/test_structured_candidate_decode.py`

**State**

One selected candidate or baseline per slot, with overlapping slots allowed.
Transitions encode:

- tracklist order without requiring non-overlap;
- plausible section duration;
- excessive backward/forward placement jumps;
- unsupported dense pileups;
- adjacent-candidate agreement;
- explicit loop/multi-segment exemptions.

**Algorithm**

Begin with beam-search/Viterbi over ordered slots. Keep the local critic frozen.
The global objective is:

```text
sum local calibrated log-odds
- order penalties
- implausible-gap penalties
- unsupported-overlap penalties
```

Do not train an end-to-end sequence model until this transparent decoder beats
local selection.

**Gate**

- Synthetic property tests cover overlap, repeated appearances, jumps and
  baseline fallback.
- Global decode never performs worse than local selection on the calibration
  objective.
- Held-out evaluation beats local selection without source-specific exceptions.

**Commit**

`feat(arbiter): decode candidate placements with tracklist constraints`

## Phase 6 — evaluation protocol

**Artifacts**

- Ignored candidate banks and reports under
  `eda/alignment/candidate_arbiter/out/`.
- No canonical status update during iteration.

**Protocol**

1. Freeze canonical baseline timelines by content hash.
2. Train synthetic + one real set; evaluate on the other real set.
3. Reverse held-out set.
4. Freeze model, feature schema, calibration and threshold.
5. Run the same strict scorer used by `drivers/race.py`.
6. Produce per-span baseline-versus-arbiter deltas.
7. Audit all accepted regressions, not only net medians.
8. Require the task’s SOTA criterion on both sets.
9. Run `make check`.
10. Only after a passing frozen run, regenerate
    `docs/alignment_status.md` through its canonical generator.

**Kill criteria**

- Candidate oracle does not beat baseline: proposer recall is the wall; proceed
  to Phase 7.
- Candidate oracle beats baseline but learned critic does not transfer: obtain
  another real GT set or improve synthetic realism; do not tune on held-out
  slots.
- Critic improves ranking but accepted regressions erase the board gain:
  increase abstention or add structured context.
- Structured decode is flat: retain local critic and close the global decoder
  experiment.

## Phase 7 — contingent mashup-invariant encoder

Start only if candidate-oracle analysis identifies representation-wall spans
where no existing candidate is locally correct.

**Architecture**

- Stem-routed encoder with short overlapping windows.
- Contrastive positives pair a clean source with the same source rendered under
  unrelated beds, stacked vocals, filters, noise, reverb, time stretch and
  pitch shift.
- Negatives include same-artist/sibling recordings and nearby repeated
  sections.
- Exact local search first; ANN indexing only after retrieval cost is measured.
- Convert retrieval hits to top-K ridges and feed them through the same critic.

**Gate**

- Improves candidate-oracle recall on representation-wall cases.
- Does not replace existing candidates or bypass the critic.
- Transfers under set-level holdout.

## Recommended execution order

1. Phase 0 candidate schema.
2. Phase 1 shadow candidate extraction.
3. Candidate-oracle measurement.
4. Stop immediately if oracle headroom is absent.
5. Phase 2 dataset.
6. Phase 3 critic + calibration.
7. Phase 4 fail-closed driver.
8. Frozen LOSO race.
9. Phase 5 only if local decisions still conflict globally.
10. Phase 7 only if oracle analysis earns a new encoder.

The first decisive milestone is not model training. It is the candidate-oracle
report: if selecting the best existing candidate per span cannot beat the
baseline, the critic cannot win and the project should move directly to
representation learning.
