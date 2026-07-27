# Multi-Set Co-Train + LOSO (Flywheel gear 1) — design spec

**Date:** 2026-07-11
**Status:** approved design, pre-implementation
**Owner:** John
**Home:** `alignment/` (`train.py`, `mert_model.py`, `records.py`, `dataset.py`)

## North star

SOTA alignment, as close to ground truth as possible, on **all** 1001tracklists
data. The critical-path bottleneck is **ground-truth scale**: today the learned
aligner can train on only **one set at a time**, so labeling more sets does not
improve the model — the flywheel's retrain gear is broken. This spec builds that
gear: **multi-set co-training** + a **leave-one-set-out (LOSO)** eval, so the
moment a new set is labeled it strengthens the model, and so we get the first
honest cross-set generalization number.

This is **flywheel gear 1** of three. Follow-on cycles: an orchestration driver
(`predict → seed_worst_spans → [human correct] → export → retrain`) and batch
selection (pick + pre-seed the next sets to label).

## The gap (verified in code)

- `train.py` `main` loads **one** yaml (`load_set`), splits *slots within that one
  set* for held-out eval, and `_run_mert_eval(gt_set_id, ...)` binds a single set's
  MERT stores into `build_aligner(train, mix, refs, slot_pools)`. Eval is held-out
  *slots*, never held-out *sets*.
- `SpanTarget` (`records.py`) has **no `set_id`** — spans can't be routed by set.
- Only two GT fixtures exist: `bb11_ground_truth.yaml`, `bb12_ground_truth.yaml`.

## Key architectural insight (makes this tractable)

`train_head` / `train_ensemble` (`mert_model.py`) train on
`examples: tuple[MertSpanExample, ...]` — **materialized feature tensors**
(`mix_segment`, `mix_window_vectors`, `ref_segs`), which are **set-agnostic**. So:

- **Co-train** = materialize examples *per set* (each using that set's stores),
  **concatenate** the example lists, train **one** head on the union. No change to
  the head's forward/loss.
- **LOSO eval** = wrap the co-trained head around the **held-out** set's stores
  (`MertLearnedAligner(head, mix, refs, slot_pools, ...)`) and predict.

The only model-core touch is threading `set_id` for bookkeeping; the learning code
is unchanged.

## Cross-set eval subtlety (do not lose this)

`MertLearnedAligner.anchor_sigma_s` — the in-domain decode is pinned by memorized
train spans interleaving eval spans; **on an unseen set the curves carry no
placement signal and the DP collapses to the front of the mix** (documented at
`mert_model.py:239-246`, BB11 2026-06-11). Scraped tracklist cues are aligner
*input*, not GT, so anchoring on them is fair. **The LOSO eval MUST set
`anchor_sigma_s` (scraped-cue anchoring), not None** — otherwise the held-out
number is a floor artifact, not a real generalization measurement.

## Architecture — three changes

The materializer is **already separated**: `build_aligner` calls
`build_examples(train_targets, mix, refs, slot_pools, search_margin_s=...) ->
tuple[MertSpanExample, ...]` then `train_ensemble`. So co-train reuses
`build_examples` per set — no extraction refactor needed.

### 1. `SpanTarget += set_id` (`records.py`, `dataset.py`)
Add `set_id: str` to the `SpanTarget` frozen dataclass. Populate it in
`dataset.track_to_target` (from the enclosing `GroundTruthSet.set_id`) and thread
`gt.set_id` through `load_set`'s `track_to_target` calls. Behavior-preserving for
existing single-set paths (the field is additive; nothing reads it yet).

### 2. Multi-set co-train (`cotrain.py`, new)
`cotrain(train_sets: list[SetStores], *, cfg, device, init=None) ->
MertAlignHead | MertAlignEnsemble` where `SetStores` bundles
`(set_id, train_spans, mix, refs, slot_pools)`. For each set call the existing
`build_examples(...)`, concatenate all examples, `train_ensemble(all_examples)`.
Returns the head only (not wrapped — the caller wraps per held-out set).

### 3. LOSO driver + `train.py --loso`
`run_loso(set_ids: list[str], yamls: dict[str, Path], *, cfg, device) ->
LosoReport`: for each held-out `s`, load stores for all sets, `cotrain` on the rest,
wrap the head around `s`'s stores with `anchor_sigma_s` set, `predict_sequence` on
`s`'s spans, `evaluate`. Print per-held-out set_start median/`<Xs` + identity, and
an aggregate. Wire as `train.py --loso --sets bb11,bb12` (default both fixtures).
Set-store loading reuses `mert_store.load_bb12_mert(set_id)` (cached locally; SSHes
pi on cache miss).

## Data flow

```
bb11.yaml ─load_set─> (gt, targets[set_id=bb11]) ─┐
bb12.yaml ─load_set─> (gt, targets[set_id=bb12]) ─┤
                                                   ▼
   for held_out in {bb11, bb12}:
     train_sets = others                            examples_for_set(each) ─concat─> train_ensemble ─> head
     wrap head + held_out stores (anchor_sigma_s set) ─predict_sequence─> evaluate(held_out)
                                                   ▼
                                            LosoReport (per-set + aggregate)
```

## Testing / validation

- **Unit — set_id threading:** `load_set(bb12.yaml)` yields targets all with
  `set_id == "<bb12 set_id>"`; `SpanTarget` still frozen/hashable.
- **Unit — cotrain concatenation:** `cotrain` on two tiny fake `SetStores` calls
  `build_examples` per set and passes the concatenated list to `train_ensemble`;
  monkeypatch both (`build_examples` → per-set stub lists, `train_ensemble` →
  capture its input length) and assert the captured length equals the sum of
  per-set counts. No GPU / no real stores needed.
- **Unit — single-set cotrain ≡ build_aligner examples:** `cotrain` with one
  `SetStores` feeds `train_ensemble` exactly the `build_examples(...)` list that
  `build_aligner` would (same monkeypatch capture), guarding parity with the
  shipped single-set path.
- **Integration (offline, real MERT cache) — the deliverable:** `train.py --loso
  --sets bb11,bb12` runs end-to-end, printing held-out set_start median for each
  direction. This is the honest generalization number; it needs both sets' MERT
  caches present (or pi reachable). Not a CI test — a `make`-able offline run.

## Risks & honest limitations

1. **n=2 is a floor, not a benchmark.** LOSO on two sets gives one number each
   direction — directional. Its *value* is proving the gear works and estimating
   how much set 3 would help, not a publishable generalization claim.
2. **MERT-cache dependency.** The integration run needs bb11+bb12 MERT stores
   (local cache or pi). Unit/golden tests must NOT depend on pi — mock the stores.
3. **Cross-set anchor prior is load-bearing** — omitting `anchor_sigma_s` yields a
   floor artifact (front-of-mix collapse), not a real number. Enforced by the eval
   path setting it explicitly and documented at the call site.
4. **bb11 GT vintage:** must be the post-2026-06-11 regeneration (pre-regen heads
   learned ~0 ref offsets). The fixtures are current; note it so a stale cache
   isn't silently used.

## Non-goals

- The orchestration driver and batch selection (flywheel gears 2–3).
- New GT labeling (needs John; this gear makes labeling *pay off*, it doesn't
  create labels).
- Changing the head architecture or the channel ensemble.
- The FX-ladder benchmark (separate, demoted diagnostic).

## Open questions for the plan

- `cotrain` home: a new `cotrain.py` vs extending `mert_model.py`. Prefer a new
  small `cotrain.py` (keeps `mert_model.py` focused; `build_aligner` stays the
  single-set convenience wrapper).
- Whether `--loso` replaces or sits beside the existing single-set `--eval` path
  (prefer beside — additive, no regression to the shipped single-set eval).
