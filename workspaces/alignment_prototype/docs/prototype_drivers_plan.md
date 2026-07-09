# Plan: Three interchangeable end-to-end aligner prototypes

Status: **BUILT 2026-07-09** — `workspaces/alignment_prototype/drivers/` + `make race`.
Two scoping decisions locked (see §Decisions).

## BUILT — what shipped

- `drivers/base.py` — `EndToEndDriver` protocol + `SetContext` (cross-set source
  auto-resolves to the other complete GT set) + timeline helpers (`finalize`
  validates through `core.contracts.load_timeline`; `gt_stem_by_slot` re-routes
  the stale materialized stem from GT).
- `drivers/classical.py` — subprocess `infer` → `joint_ref_decode` (the shipped
  baseline).
- `drivers/agentic.py` — `agentic.loop.resolve` over the classical base;
  committed/review/suggest override set_start, **escalate falls back to
  classical**. Owns PLACEMENT only — inherits classical's ref-decode verbatim
  (translating ref_segments double-counts the placement move against the
  GT-anchored scorer; measured, removed).
- `drivers/ml.py` — hybrid C1: MERT identity + classical placement + learned
  `TrajectoryDecoder` (checkpoint `decoder_<source_set>.pt`) for ref_segments;
  rebuilds the exact `(2,Tm,Tr)` grid `trajectory.data` trains on; graceful
  keep-classical fallback for short/no-segment spans.
- `drivers/race.py` + `make race` (SETS=/DRIVERS=/EXTRA=) — runs each driver,
  scores every timeline through the SAME `score_timeline_vs_gt`, prints the board.

**Clean board — fresh per-set infer base + LIVE agentic** (both confounds removed;
`make race`, 2026-07-09; ~44 min, two full infer passes):

| set | driver | id% | place_med | ref_med | head_traj% | acap_traj% |
|---|---|---|---|---|---|---|
| BB11 | classical | 84 | 7.3 | 25.8 | 20 | 14 |
| BB11 | agentic | 84 | **2.5** | 24.3 | 20 | 14 |
| BB11 | ml | 84 | 7.3 | **3.7** | 20 | **17** |
| BB12 | classical | 84 | 4.8 | 10.2 | 26 | 25 |
| BB12 | agentic | 84 | **3.1** | 10.2 | 26 | 25 |
| BB12 | ml | 84 | 4.8 | 9.8 | 19 | 22 |

Verdict (de-confounded):
- **Identity holds 84% everywhere.** The earlier 81/75 id dips were a
  reuse-base + replay artifact (identity is time-overlap-matched; moving
  set_start reshuffles overlaps). Recording_id is never changed.
- **Agentic (live) = clean placement win on BOTH sets** (BB11 7.3→2.5, BB12
  4.8→3.1), no identity/trajectory cost. The earlier BB12 placement regression
  (4.5→9.4) was **replay**, not the real driver. Ship as the placement refiner.
- **ML is a CONDITIONAL lever, not a uniform upgrade.** Big win where classical
  ref-decode is weak (BB11 25.8→3.7s) → wash/regression where it's already strong
  (BB12 10.2→9.8s, **traj 26→19 head**). Cross-set decoder (held-out ~0.37) can't
  beat a tuned in-domain looptrace. Next: gate learned segments on low classical
  path-conf per span — the race board makes that measurable.

An earlier reuse-base + agentic-replay board (BB11 only) showed agentic
placement 2.9s and ml ref 3.3s but spurious id/placement regressions on BB12 —
kept only in git history as the record of why the fresh+live run was necessary.

Reproduce: `make race` (needs pi-storage + warm MERT/aligning audio; live
agentic needs mix MERT). Fast iteration: `--reuse-base SET=path` skips infer.
ML needs a cross-set checkpoint per set: `decoder_1fsnxchk.pt` (align BB11) and
`decoder_2nvzlh2k.pt` (align BB12), both trained 2026-07-09.

### ML confidence gate — turns the conditional lever additive (2026-07-09)

`viterbi_segments(..., return_score=True)` now emits a per-span decode confidence
(mean margin of the decoded path over the frame's average ref emission).
`HybridMlDriver(gate_margin=T)` / `race --ml-gate T` keeps classical segments
unless the learned decode clears `T` — trust ML only where the model is sure.

The confidence cleanly separates the two regimes on its own: BB11 (ML helps)
scores median 5.8 (p10 3.1); BB12 (ML hurts) median 1.1 (p75 2.9). Sweep
(reusing the fresh classical bases):

| gate | set | ref_med | head_traj% | acap_traj% | ml-decoded / gated |
|---|---|---|---|---|---|
| none | BB11 | 3.7 | 20 | 17 | 112 / 0 |
| **3.0** | BB11 | 3.7 | **22** | **18** | 101 / 11 |
| none | BB12 | 9.8 | 19 ← regress | 22 | 129 / 0 |
| **3.0** | BB12 | 13.3 | **26** ← restored | **25** | 32 / 97 |

Gate 3.0 keeps 90% of BB11 (preserves the ref-offset win, +2pp trajectory) and
gates 75% of BB12 back to classical (trajectory regression 26→19 ELIMINATED,
restored to 26/25). Net: ML becomes non-regressing on trajectory (its owned
metric) across both sets. Caveat: BB12 `ref_med` wobbles 9.8→13.3 — a small-n
straight-clip subset; the ungated 9.8-vs-classical-10.2 "win" was noise.

Threshold is **logit-scaled per checkpoint** — `--ml-gate` stays OPT-IN, not a
silent default; 3.0 is validated for the current `decoder_*.pt` pair. Retraining
the decoder means re-sweeping. Next: a per-span learned router (ml vs classical)
over {ml confidence, classical path-conf, span class} — this gate is its
one-feature threshold baseline.

## Original plan (as drafted)

## Framing

We don't need to build three aligners from scratch — ~2.5 already exist, they're
just not wired as interchangeable end-to-end drivers scored on one board:

- **Classical** already runs end-to-end: `infer.py` (MERT identity + fp/stem/lyrics
  placement, fixed gates) → `joint_ref_decode.py` (Viterbi ref-segments) → timeline JSON.
- **Agentic** already runs: `agentic/ --live` (belief fusion + `Ladder` + budget-aware
  worst-belief-first scheduling + escalation) — but emits per-rung precision/recall,
  **not** a full timeline, so it never hits the real scorecard.
- **ML** exists only for the *trajectory* stage: `trajectory/TrajectoryDecoder`
  (.425 strict / .490 fiber held-out BB11). No learned identity/placement → a "full
  ML driver" is genuinely partial.

So the real work is **unification** (one driver interface + one scorecard for all
three) plus **filling the ML gap**, not three greenfield builds.

## Organizing idea

Every aligner solves the same three sub-problems per span — **identity** (which
recording), **placement** (set_start / where in the mix), **trajectory**
(ref_segments / internal warp). Drivers differ only in *how* they decide each stage.
Put them behind one interface, race them on one scorecard.

```
set_id + {tokenized tracklist, track audios, set audio}
    → [DRIVER] → predicted_timeline.json (list[SpanPrediction] w/ ref_segments)
    → score_timeline_vs_gt → scorecard row
```

The scorecard (`score_timeline_vs_gt.py` / `build_span_table.py`) already consumes a
timeline JSON and emits identity%, set_start err, ref_offset err, trajectory_acc, and
impact-weighted failure attribution. **That is the shared judge — nothing new needed
there.** The whole plan hangs off making all three drivers emit that same timeline JSON.

## 1. Common driver interface (the only genuinely new abstraction)

Thin protocol in `harness/`, one level above the existing `Probe`/`AlignmentResult`
contract:

```python
class EndToEndDriver(Protocol):
    name: str
    def align_set(self, ctx: SetContext) -> PredictedTimeline: ...
```

- `SetContext` = existing loader (`dataset.load_set` + aligning-folder `manifest.json`:
  mix.m4a, mix_vocals/instrumental, per-slot candidate audios + stems).
- `PredictedTimeline` = `list[SpanPrediction]` — `records.py` already has the right
  fields (slot_label, recording_id, set_start/end, ref_start/end, ref_segments,
  tempo_ratio, confidence).
- One runner `race.py` (`--driver {classical,agentic,ml} --set-id …`) → writes
  `out/<set>_<driver>_timeline.json` → invokes existing scorer. Plus `make race` to run
  all three on BB11+BB12 and print a side-by-side board.

This is the load-bearing deliverable. Once it exists the three drivers are adapters.

## 2. Driver A — Classical / Deterministic (low effort; mostly wrapping)

Wrap existing `infer.py` → `joint_ref_decode.py` behind `align_set`. Fixed, no
learning at decision time: MERT `predict_sequence` identity, axis-routed placement
(`--fp-placement` regular/instr, `--stem-placement`+`--lyrics-placement` acappella,
90s gate), chroma/HuBERT Viterbi ref_segments.
- Effort: small — refactor CLI-oriented I/O into a callable returning `PredictedTimeline`.
- Expected: reproduces today's headline (identity 84/83%, set_start median 6.3/7.9s,
  acappella traj 21%). **Baseline the other two must beat.**
- Risk: low.

## 3. Driver B — Agentic (medium effort; gap is output, not logic)

Reuse `agentic/loop.resolve()` + `--live` runners + `SpanBelief` + `Ladder` +
budget-aware scheduling + escalation (all built). Missing piece: it doesn't commit a
full timeline. Add adapter: after `resolve()`, each span's dominant belief cluster →
`SpanPrediction` (auto_commit/review/suggest → predictions; **escalate → fall back to
Driver A's prediction for that span** so the timeline is complete — see Decisions).
- Effort: medium — belief→prediction serializer + escalation-fill.
- Uniquely tests: does precision-weighted fusion + abstention beat fixed gates on the
  same board? Ladder gives a free **auto-commit clean%** (pseudo-GT quality) metric
  relevant to the bootstrap flywheel.
- Risk: medium. Live probe costs (Whisper/HuBERT) → slow full-set runs; needs feature
  caches warm.

## 4. Driver C — ML (highest effort; partial today)

Honest scope: identity = existing MERT head, trajectory = existing `TrajectoryDecoder`
+ `viterbi_segments`, placement = weak link (no learned model).

- **C1 (BUILD NOW):** MERT identity → *borrow* classical/agentic placement → learned
  trajectory decoder for ref_segments. "Hybrid ML": learning does the trajectory stage
  (its proven .425/.490 lane), heuristics do placement. Real third contestant, no new GT.
- **C2 (DEFERRED — research, blocked on GT):** small learned placement selector over
  probe evidence (HuBERT diagonal, fp sharpness, fiber ambiguity, lyrics score) — the
  "learned fusion arbiter." Needs a 3rd labeled set (BB10/Murph); n=2 overfits.
- Effort: C1 medium, C2 large + blocked.

## 5. Sequence

1. Interface + `race.py` + `make race` (~½ day).
2. Driver A wrap → confirm it reproduces current scorecard (validates harness). (~½ day)
3. Driver B belief→timeline adapter → first agentic-vs-classical board. (~1 day)
4. Driver C1 hybrid-ML. (~1 day)
5. C2 only after a third GT set exists.

## 6. Payoff

Reproducible three-way scorecard where each stage is swappable — "does the ML
trajectory decoder beat Viterbi in a full pipeline?" and "does agentic abstention beat
fixed gates?" become one `make race` away instead of buried in ad-hoc `out/*.json`.
Turns the aligner from a pile of probes into three nameable systems, improvable
independently — the north-star shape (interchangeable drivers behind the harness contract).

## Decisions (locked 2026-07-09)

- **ML scope = C1 hybrid only.** MERT identity + borrowed placement + learned
  trajectory decoder. Defer C2 (learned placement selector) until a 3rd labeled set
  exists; BB11+BB12 (n=2) would overfit.
- **Agentic escalation → fall back to classical.** Escalated spans (belief quality
  < 0.10) filled by Driver A's prediction so all drivers emit complete, span-for-span
  comparable timelines. Agentic "auto-commit clean%" survives as its own metric.

## Key file paths (from 2026-07-09 code map)

| Item | Path |
|---|---|
| Prototype root | `workspaces/alignment_prototype/` |
| Driver contract (extend here) | `harness/contract.py`, `harness/driver.py`, `harness/merge.py`, `harness/axes.py` |
| Prediction record | `workspaces/alignment_prototype/records.py` (`SpanPrediction`) |
| GT loader | `workspaces/alignment_prototype/dataset.py` (`load_set`) |
| Classical stages | `infer.py`, `joint_ref_decode.py` |
| Agentic | `agentic/loop.py` (`resolve`), `belief.py`, `policy.py` (`Ladder`), `live_runners.py` |
| ML trajectory | `trajectory/model.py` (`TrajectoryDecoder`), `trajectory/decode.py` (`viterbi_segments`), `trajectory/train.py` |
| Scorecard | `score_timeline_vs_gt.py`, `eda/alignment/failure_analysis/build_span_table.py` + `analyze.py`; `make scorecard` |
| GT fixtures | `labeling/fixtures/bb11_ground_truth.yaml` (2nvzlh2k), `labeling/fixtures/bb12_ground_truth.yaml` (1fsnxchk) |
| Existing e2e driver script | `scripts/reinfer_driver.sh` |

GT sets: BB11 `2nvzlh2k` (150 spans), BB12 `1fsnxchk` (166 spans). Others available
(inference-only, no full GT): `1rfb0yl9` Disco Lines, `pwgrrb1` Murph, `w1mgcjt` BB10.
