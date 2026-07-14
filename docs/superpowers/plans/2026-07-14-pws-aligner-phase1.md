# PWS Aligner — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the aligner's hand-tuned probe-fusion (`harness/merge.py` `source_priority`) with a label model that *learns each probe's accuracy from unlabeled sets, without ground truth*, and prove it beats the hand-tuned baseline on BB11/BB12 with zero new GT.

**Architecture:** Existing probes already emit `AlignmentResult` votes (recording_id, offset_s, confidence, abstain, source). Per mix span, collect those votes over a **discrete hypothesis space** `H = {candidate recording_id × offset-bin}` (bin width = the scorecard's 2 s strict window). A **label model** (Dawid–Skene baseline → FABLE upgrade) estimates per-probe confusion and returns a posterior over `H` per span — no GT. The MAP hypothesis feeds decode in place of `source_priority`; GT (BB11/BB12) is validation only. Fully de-risked: a dependency-free Dawid–Skene baseline is the go/no-go kill-gate (Task 5) *before* any heavy FABLE/GP work.

**Tech Stack:** Python 3 (repo `venvs/audio`), numpy for the baseline label model, pytest. FABLE (Task 6) adds a GP + Pólya-Gamma VI stack, gated on Task 5 showing lift. No pi-storage or real-audio dependency for unit tests — the label model is validated with **synthetic-oracle tests** (known accuracies, known feature-dependent noise).

## Global Constraints

- **Home:** all new code under `workspaces/pws_aligner/`. **Import**, never copy, `workspaces/alignment_prototype/` modules. Do not modify `alignment_prototype` until the fork beats it on the scorecard.
- **Sensor freeze respected:** add NO new probes/channels/priors. This effort touches only aggregation/training (the sanctioned learned-model + driver lanes per `alignment_prototype/CLAUDE.md`).
- **No new GT authoring.** GT sets are validation only: **BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`**. The committed baseline is the **corrected GT (`de1ce92`, 2026-07-14 re-export)** — the aligner's ~76% GT-seconds loss against it is *genuine failure concentrated in acappella decode + stem-routing*, not a data artifact. That acappella region is precisely where the label model must move the needle.
- **BB12 (`1fsnxchk`) is the PRIMARY board; BB11 (`2nvzlh2k`) is confirmatory.** BB11 agentic **stalls on the Whisper lyrics probe on Mac MPS** — the stall is backend-specific, so run the Task-5 gate inference on a **Vast.ai CUDA box** (credit added 2026-07-14; API key `~/.config/vastai/vast_api_key`, use the curl API — the CLI is broken on Py3.14; instance recipe per `project_vastai_instance_choice`, PRO 4000-class ≈40.5 s/track; **list-before-create, destroy only your own instance**). On CUDA, BB11 runs with the lyrics probe enabled. Mac-MPS fallback: BB12 only, lyrics probe off for BB11 or skip BB11. Do not block the Task-5 decision on BB11 either way.
- **Success bar (the falsifiable claim):** the learned label model beats hand-tuned `source_priority` fusion on the **BB12 scorecard** (identity + trajectory + set_start), confirmed on BB11 where runnable, adding zero new GT. Secondary: learned per-probe precisions track the measured `probe_precision_transfer` values (fp's ~0.90→0.53 feature-dependent swing).
- **Gates:** Snorkel density/modeling-advantage gate before deploying the model over majority-vote; FABLE `Corr(X, LF)` gate before instance-conditioning any probe.
- **Style:** `from __future__ import annotations`; frozen dataclasses for records; pure functions, I/O at edges; `core/result.py`-style Result in library code, `sys.exit` at CLI edges. Rust-flavoured functional Python.
- **Tests run from repo root:** `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/... -v`.
- **Scorecard command:** `venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt --set-id <id> [--fibers] [--decompose]`.

## File Structure

- `workspaces/pws_aligner/__init__.py` — package marker.
- `workspaces/pws_aligner/CLAUDE.md` — one-screen module guide (freeze note + PWS framing + kill-gate).
- `workspaces/pws_aligner/votes.py` — `Vote` record + `collect_votes()` adapter (probe `AlignmentResult` → `Vote` + feature vector + typed abstain). Pure over given probe outputs.
- `workspaces/pws_aligner/hypotheses.py` — `Hypothesis` record + `build_hypothesis_space()` + `vote_to_hypothesis()` (offset discretization).
- `workspaces/pws_aligner/label_model.py` — `LabelModel` protocol; `DawidSkene` (numpy EM) baseline; `MajorityVote` fallback.
- `workspaces/pws_aligner/density_gate.py` — `label_density()` + `choose_aggregator()` (Snorkel modeling-advantage gate).
- `workspaces/pws_aligner/decode_bridge.py` — turn a per-span posterior over `H` into the timeline JSON placement (replaces `source_priority` for the fork's driver).
- `workspaces/pws_aligner/fable/` — (Task 6, gated) FABLE instance-feature label model behind the same `LabelModel` protocol.
- `workspaces/pws_aligner/verifier.py` — (Task 7) Confident-Learning auditor + GT-calibration report.
- `workspaces/pws_aligner/tests/` — synthetic-oracle unit tests per module.

---

### Task 1: Scaffold package + `Vote` record + vote collection

**Files:**
- Create: `workspaces/pws_aligner/__init__.py`, `workspaces/pws_aligner/CLAUDE.md`
- Create: `workspaces/pws_aligner/votes.py`
- Test: `workspaces/pws_aligner/tests/test_votes.py`

**Interfaces:**
- Consumes: `workspaces.alignment_prototype.harness.contract.AlignmentResult` (fields: `recording_id: str|None`, `offset_s: float`, `confidence: float`, `abstain: bool`, `source: str`).
- Produces:
  - `AbstainReason` enum: `NO_DATA`, `LOW_MARGIN`, `OUT_OF_DOMAIN`, `NONE`.
  - `@dataclass(frozen=True) Vote(probe: str, span_id: str, recording_id: str|None, offset_s: float, confidence: float, abstained: bool, reason: AbstainReason, features: tuple[float, ...])`.
  - `collect_votes(span_id: str, results: Sequence[tuple[AlignmentResult, tuple[float,...], AbstainReason]]) -> tuple[Vote, ...]` — pure adapter; `features` is the instance-feature vector (span embedding + sharpness proxies) supplied by the caller, kept opaque here.

- [ ] **Step 1: Write the failing test**

```python
# workspaces/pws_aligner/tests/test_votes.py
from __future__ import annotations
from workspaces.alignment_prototype.harness.contract import AlignmentResult
from workspaces.pws_aligner.votes import Vote, AbstainReason, collect_votes


def test_collect_maps_result_and_abstain_reason():
    fired = AlignmentResult(recording_id="r1", offset_s=120.0, confidence=0.8, source="fp")
    skipped = AlignmentResult.abstained(source="hubert")
    votes = collect_votes(
        "span0",
        [
            (fired, (0.8, 1.2), AbstainReason.NONE),
            (skipped, (0.0, 0.0), AbstainReason.NO_DATA),
        ],
    )
    assert votes == (
        Vote("fp", "span0", "r1", 120.0, 0.8, False, AbstainReason.NONE, (0.8, 1.2)),
        Vote("hubert", "span0", None, 0.0, 0.0, True, AbstainReason.NO_DATA, (0.0, 0.0)),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_votes.py -v`
Expected: FAIL with `ModuleNotFoundError: workspaces.pws_aligner.votes`.

- [ ] **Step 3: Write minimal implementation**

Create `workspaces/pws_aligner/__init__.py` (empty). Create `workspaces/pws_aligner/votes.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
from workspaces.alignment_prototype.harness.contract import AlignmentResult


class AbstainReason(Enum):
    NONE = "none"
    NO_DATA = "no_data"
    LOW_MARGIN = "low_margin"
    OUT_OF_DOMAIN = "out_of_domain"


@dataclass(frozen=True)
class Vote:
    probe: str
    span_id: str
    recording_id: str | None
    offset_s: float
    confidence: float
    abstained: bool
    reason: AbstainReason
    features: tuple[float, ...]


def collect_votes(
    span_id: str,
    results: Sequence[tuple[AlignmentResult, tuple[float, ...], AbstainReason]],
) -> tuple[Vote, ...]:
    return tuple(
        Vote(
            probe=res.source,
            span_id=span_id,
            recording_id=None if res.abstain else res.recording_id,
            offset_s=res.offset_s,
            confidence=res.confidence,
            abstained=res.abstain,
            reason=reason,
            features=features,
        )
        for res, features, reason in results
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_votes.py -v`
Expected: PASS.

- [ ] **Step 5: Write `CLAUDE.md`**

Create `workspaces/pws_aligner/CLAUDE.md` with: (a) one line that this fork rebuilds the aligner as programmatic weak supervision per `docs/superpowers/specs/2026-07-14-pws-aligner-design.md`; (b) the sensor-freeze note (no new probes — aggregation only); (c) the kill-gate: "if Task 5's Dawid–Skene fusion does not beat `source_priority` on the BB11+BB12 scorecard, stop before FABLE."

- [ ] **Step 6: Commit**

```bash
git add workspaces/pws_aligner/__init__.py workspaces/pws_aligner/votes.py workspaces/pws_aligner/CLAUDE.md workspaces/pws_aligner/tests/test_votes.py
git commit -m "feat(pws): Vote record + probe-result vote adapter"
```

---

### Task 2: Discrete hypothesis space + offset discretization

**Files:**
- Create: `workspaces/pws_aligner/hypotheses.py`
- Test: `workspaces/pws_aligner/tests/test_hypotheses.py`

**Interfaces:**
- Consumes: `Vote` (Task 1).
- Produces:
  - `@dataclass(frozen=True) Hypothesis(recording_id: str|None, offset_bin: int)`.
  - `build_hypothesis_space(votes: Sequence[Vote], bin_s: float = 2.0) -> tuple[Hypothesis, ...]` — the sorted, de-duplicated set of hypotheses any non-abstaining vote lands in (`offset_bin = round(offset_s / bin_s)`), plus a distinguished `Hypothesis(None, 0)` NULL/abstain hypothesis at index 0.
  - `vote_to_hypothesis(vote: Vote, bin_s: float = 2.0) -> Hypothesis` — NULL for abstained votes.

- [ ] **Step 1: Write the failing test**

```python
# workspaces/pws_aligner/tests/test_hypotheses.py
from __future__ import annotations
from workspaces.pws_aligner.votes import Vote, AbstainReason
from workspaces.pws_aligner.hypotheses import (
    Hypothesis, build_hypothesis_space, vote_to_hypothesis,
)

NULL = Hypothesis(None, 0)


def _v(probe, rid, off, ab=False):
    return Vote(probe, "s", rid, off, 0.7, ab, AbstainReason.NONE if not ab else AbstainReason.NO_DATA, ())


def test_offset_binning_and_null():
    assert vote_to_hypothesis(_v("fp", "r1", 121.0), bin_s=2.0) == Hypothesis("r1", 60)  # round(121/2)=60
    assert vote_to_hypothesis(_v("h", None, 0.0, ab=True)) == NULL


def test_space_dedups_and_puts_null_first():
    votes = [_v("fp", "r1", 120.0), _v("chroma", "r1", 121.0), _v("h", "r2", 300.0)]
    space = build_hypothesis_space(votes, bin_s=2.0)
    assert space[0] == NULL
    assert set(space) == {NULL, Hypothesis("r1", 60), Hypothesis("r2", 150)}
    assert len(space) == 3  # r1@120 and r1@121 collapse to bin 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_hypotheses.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# workspaces/pws_aligner/hypotheses.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from workspaces.pws_aligner.votes import Vote

_NULL_FIRST = object()


@dataclass(frozen=True)
class Hypothesis:
    recording_id: str | None
    offset_bin: int


def vote_to_hypothesis(vote: Vote, bin_s: float = 2.0) -> Hypothesis:
    if vote.abstained or vote.recording_id is None:
        return Hypothesis(None, 0)
    return Hypothesis(vote.recording_id, round(vote.offset_s / bin_s))


def build_hypothesis_space(votes: Sequence[Vote], bin_s: float = 2.0) -> tuple[Hypothesis, ...]:
    null = Hypothesis(None, 0)
    others = {
        vote_to_hypothesis(v, bin_s)
        for v in votes
        if not v.abstained and v.recording_id is not None
    }
    ordered = sorted(others, key=lambda h: (h.recording_id or "", h.offset_bin))
    return (null, *ordered)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_hypotheses.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workspaces/pws_aligner/hypotheses.py workspaces/pws_aligner/tests/test_hypotheses.py
git commit -m "feat(pws): discrete hypothesis space + offset binning"
```

---

### Task 3: Dawid–Skene label model (numpy EM, no GT)

**Files:**
- Create: `workspaces/pws_aligner/label_model.py`
- Test: `workspaces/pws_aligner/tests/test_label_model.py`

**Interfaces:**
- Consumes: `Vote` (Task 1), `Hypothesis`, `build_hypothesis_space`, `vote_to_hypothesis` (Task 2).
- Produces:
  - `LabelModel` protocol with `fit(spans: Sequence[Sequence[Vote]]) -> None` and `predict_proba(span_votes: Sequence[Vote]) -> dict[Hypothesis, float]`.
  - `MajorityVote` implementing it (confidence-weighted tally).
  - `DawidSkene(max_iter: int = 50, tol: float = 1e-4)` implementing it: classic EM over per-probe confusion between "voted hypothesis" and latent true hypothesis, estimating each probe's reliability with **no GT**. Exposes `probe_accuracy() -> dict[str, float]` (diagonal mass of each probe's learned confusion = P(correct | voted)).

- [ ] **Step 1: Write the failing test (synthetic-oracle recovery)**

```python
# workspaces/pws_aligner/tests/test_label_model.py
from __future__ import annotations
import random
from workspaces.pws_aligner.votes import Vote, AbstainReason
from workspaces.pws_aligner.label_model import DawidSkene, MajorityVote
from workspaces.pws_aligner.hypotheses import Hypothesis, vote_to_hypothesis


def _synth(n=400, seed=0):
    """3 probes: 'good' 0.95, 'ok' 0.75, 'bad' 0.45. True hyp per span in {r1@10, r2@20}."""
    rng = random.Random(seed)
    accs = {"good": 0.95, "ok": 0.75, "bad": 0.45}
    truths = [("r1", 5) if rng.random() < 0.5 else ("r2", 10) for _ in range(n)]  # bins at bin_s=2
    spans = []
    for (rid, b) in truths:
        votes = []
        for probe, a in accs.items():
            if rng.random() < a:
                off = b * 2.0
                v_rid = rid
            else:  # wrong: flip to the other hypothesis
                (v_rid, wb) = ("r2", 10) if rid == "r1" else ("r1", 5)
                off = wb * 2.0
            votes.append(Vote(probe, "s", v_rid, off, 0.7, False, AbstainReason.NONE, ()))
        spans.append(votes)
    return spans, truths, accs


def test_dawid_skene_recovers_accuracies_and_labels():
    spans, truths, accs = _synth()
    ds = DawidSkene()
    ds.fit(spans)
    est = ds.probe_accuracy()
    # ranking preserved: good > ok > bad, each within 0.1 of truth
    assert est["good"] > est["ok"] > est["bad"]
    for p in accs:
        assert abs(est[p] - accs[p]) < 0.12
    # label accuracy beats the worst probe and beats majority vote is not required,
    # but MAP labels must be >= 0.9 correct on this easy synthetic
    correct = 0
    for votes, (rid, b) in zip(spans, truths):
        proba = ds.predict_proba(votes)
        top = max(proba, key=proba.get)
        correct += (top == Hypothesis(rid, b))
    assert correct / len(spans) >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_label_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Implement `MajorityVote` (confidence-weighted tally over `vote_to_hypothesis`) and `DawidSkene` as classic EM: E-step infers a distribution over the latent hypothesis per span from current per-probe confusion + class priors; M-step re-estimates each probe's confusion (rows = voted hypothesis, cols = latent) and the class priors. Use `numpy`. Represent each probe's confusion as P(voted-is-correct) vs P(voted-is-wrong-spread-uniformly) — a 2-parameter-per-probe reduction is sufficient for the binned space and matches Data Programming's `(α, β)`. `probe_accuracy()` returns the learned diagonal. Initialize accuracies at 0.7, priors uniform, iterate to `tol` or `max_iter`. (Reference: `1605.07723` §3 independent model; `1911.00068` for the confusion-matrix framing.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_label_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workspaces/pws_aligner/label_model.py workspaces/pws_aligner/tests/test_label_model.py
git commit -m "feat(pws): Dawid-Skene label model recovers probe accuracy without GT"
```

---

### Task 4: Snorkel density / modeling-advantage gate

**Files:**
- Create: `workspaces/pws_aligner/density_gate.py`
- Test: `workspaces/pws_aligner/tests/test_density_gate.py`

**Interfaces:**
- Consumes: `Vote` (Task 1), `LabelModel`/`MajorityVote`/`DawidSkene` (Task 3).
- Produces:
  - `label_density(spans: Sequence[Sequence[Vote]]) -> float` — mean non-abstaining votes per span.
  - `choose_aggregator(spans, low: float = 1.0, high: float = 6.0) -> str` — returns `"majority_vote"` when density `< low` or `> high` (Snorkel: MV provably near-optimal at both extremes), else `"label_model"`.

- [ ] **Step 1: Write the failing test**

```python
# workspaces/pws_aligner/tests/test_density_gate.py
from __future__ import annotations
from workspaces.pws_aligner.votes import Vote, AbstainReason
from workspaces.pws_aligner.density_gate import label_density, choose_aggregator


def _span(n_fire, n_abstain):
    fire = [Vote(f"p{i}", "s", "r1", 10.0, 0.7, False, AbstainReason.NONE, ()) for i in range(n_fire)]
    ab = [Vote(f"a{i}", "s", None, 0.0, 0.0, True, AbstainReason.NO_DATA, ()) for i in range(n_abstain)]
    return fire + ab


def test_density_counts_only_fired_votes():
    spans = [_span(3, 2), _span(1, 4)]
    assert label_density(spans) == 2.0  # (3 + 1) / 2


def test_gate_picks_mv_at_low_density_and_model_midband():
    assert choose_aggregator([_span(1, 5) for _ in range(10)], low=1.5) == "majority_vote"
    assert choose_aggregator([_span(3, 1) for _ in range(10)]) == "label_model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_density_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# workspaces/pws_aligner/density_gate.py
from __future__ import annotations
from typing import Sequence
from workspaces.pws_aligner.votes import Vote


def label_density(spans: Sequence[Sequence[Vote]]) -> float:
    if not spans:
        return 0.0
    fired = sum(sum(1 for v in s if not v.abstained) for s in spans)
    return fired / len(spans)


def choose_aggregator(spans: Sequence[Sequence[Vote]], low: float = 1.0, high: float = 6.0) -> str:
    d = label_density(spans)
    return "label_model" if low <= d <= high else "majority_vote"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_density_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add workspaces/pws_aligner/density_gate.py workspaces/pws_aligner/tests/test_density_gate.py
git commit -m "feat(pws): Snorkel density gate (MV vs label model)"
```

---

### Task 5: Decode bridge + scorecard acceptance — **THE KILL-GATE**

**Files:**
- Create: `workspaces/pws_aligner/decode_bridge.py`
- Create: `workspaces/pws_aligner/run_phase1.py` (CLI: build votes for a set's spans, aggregate, write a timeline JSON in the format `score_timeline_vs_gt` reads)
- Test: `workspaces/pws_aligner/tests/test_decode_bridge.py`

**Interfaces:**
- Consumes: per-span posterior `dict[Hypothesis, float]` (Task 3), `choose_aggregator` (Task 4). Existing: `workspaces.alignment_prototype.infer` (to obtain per-span probe `AlignmentResult`s and the feature vectors from `neuro/precision.py`), and the predicted-timeline JSON schema that `score_timeline_vs_gt` consumes (`out/<set_id>_predicted_timeline.json`).
- Produces:
  - `posterior_to_placement(span_id, posterior, bin_s=2.0) -> dict` — one timeline-span dict: `recording_id`, `offset_s` (bin center of MAP hypothesis), `confidence` (MAP posterior mass), `abstain` (True if NULL wins).
  - `run_phase1(set_id: str) -> Path` — end-to-end: reuse `infer` to get probe outputs per span, `collect_votes`, gate, aggregate, write the fork's timeline JSON.

- [ ] **Step 1: Write the failing test (unit — placement extraction)**

```python
# workspaces/pws_aligner/tests/test_decode_bridge.py
from __future__ import annotations
from workspaces.pws_aligner.hypotheses import Hypothesis
from workspaces.pws_aligner.decode_bridge import posterior_to_placement


def test_map_hypothesis_becomes_placement():
    post = {Hypothesis(None, 0): 0.1, Hypothesis("r1", 60): 0.7, Hypothesis("r2", 150): 0.2}
    span = posterior_to_placement("span0", post, bin_s=2.0)
    assert span["recording_id"] == "r1"
    assert span["offset_s"] == 120.0  # bin 60 * 2.0
    assert abs(span["confidence"] - 0.7) < 1e-9
    assert span["abstain"] is False


def test_null_winner_abstains():
    post = {Hypothesis(None, 0): 0.6, Hypothesis("r1", 60): 0.4}
    span = posterior_to_placement("span0", post)
    assert span["abstain"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_decode_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation of `posterior_to_placement`**

```python
# workspaces/pws_aligner/decode_bridge.py
from __future__ import annotations
from workspaces.pws_aligner.hypotheses import Hypothesis


def posterior_to_placement(span_id: str, posterior: dict[Hypothesis, float], bin_s: float = 2.0) -> dict:
    top = max(posterior, key=posterior.get)
    is_null = top.recording_id is None
    return {
        "span_id": span_id,
        "recording_id": None if is_null else top.recording_id,
        "offset_s": 0.0 if is_null else top.offset_bin * bin_s,
        "confidence": float(posterior[top]),
        "abstain": is_null,
    }
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_decode_bridge.py -v`
Expected: PASS.

- [ ] **Step 5: Implement `run_phase1` CLI**

Wire `run_phase1(set_id)`: call the existing `infer` path to get per-span probe `AlignmentResult`s + `neuro/precision.py` feature vectors, `collect_votes`, `choose_aggregator`, fit the chosen aggregator on the set's spans (unlabeled), `predict_proba` per span, `posterior_to_placement`, and write `out/<set_id>_pws_timeline.json` in the exact schema `score_timeline_vs_gt` reads. Fail-fast (`sys.exit`) on missing set audio/features.

- [ ] **Step 6: Run the acceptance gate (records numbers, not an assert)**

```bash
# PRIMARY board = BB12 (1fsnxchk). Hand-tuned baseline (existing pipeline) for reference:
venvs/audio/bin/python -m workspaces.alignment_prototype.infer --set-id 1fsnxchk
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt --set-id 1fsnxchk --fibers --decompose
# PWS fork:
venvs/audio/bin/python -m workspaces.pws_aligner.run_phase1 --set-id 1fsnxchk
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt --set-id 1fsnxchk --timeline out/1fsnxchk_pws_timeline.json --fibers --decompose
# CONFIRMATORY = BB11 (2nvzlh2k), ONLY with the Whisper lyrics probe disabled (it stalls on MPS);
# if it hangs, skip BB11 for the gate — the decision rests on BB12.
```

**Decision gate:** record identity %, set_start median, trajectory (strict + fiber) for baseline vs PWS on **BB12** (and BB11 if it ran) in the commit message. **If PWS does not beat hand-tuned fusion on BB12, STOP — do not build FABLE (Task 6).** A null result here is a real finding (per the Snorkel density gate): report it and route to *more/denser LFs* rather than more machinery. `score_timeline_vs_gt` may need a `--timeline <path>` arg; add it to `alignment_prototype` only if absent (smallest possible change, its own commit).

- [ ] **Step 7: Commit**

```bash
git add workspaces/pws_aligner/decode_bridge.py workspaces/pws_aligner/run_phase1.py workspaces/pws_aligner/tests/test_decode_bridge.py
git commit -m "feat(pws): decode bridge + phase-1 scorecard gate [record baseline vs PWS numbers in body]"
```

---

### Task 6: FABLE instance-feature label model — **GATED on Task 5 lift**

**Files:**
- Create: `workspaces/pws_aligner/fable/__init__.py`, `workspaces/pws_aligner/fable/model.py`
- Test: `workspaces/pws_aligner/tests/test_fable.py`

**Interfaces:**
- Consumes: `Vote` (with non-empty `features`), the `LabelModel` protocol (Task 3).
- Produces: `Fable(...)` implementing `LabelModel` (`fit`, `predict_proba`, `probe_accuracy`) **plus** `probe_accuracy_at(features: tuple[float,...]) -> dict[str, float]` — the instance-conditioned accuracy. Route the EBCC subtype mixture through a GP over `features` (Pólya-Gamma-augmented VI); see `2210.02724`. **Adapt the authors' reference implementation** rather than deriving PG-VI from scratch; keep it behind our protocol so Tasks 3–5 are unchanged.

**Gate before writing code:** for each probe, compute `Corr(features, correctness)` on the Task-3 synthetic + on the real BB spans. Instance-condition **only** probes with non-trivial correlation (FABLE law). If no probe clears it, FABLE cannot help — record that and skip to Task 7 with the Dawid–Skene model as final.

- [ ] **Step 1: Write the failing test (feature-dependent synthetic oracle)**

```python
# workspaces/pws_aligner/tests/test_fable.py
from __future__ import annotations
import random
from workspaces.pws_aligner.votes import Vote, AbstainReason
from workspaces.pws_aligner.fable.model import Fable


def _synth_feature_dependent(n=600, seed=1):
    """fp is accurate (0.95) when feature>0 ('clean drop'), poor (0.4) when feature<0 ('heavy FX').
    A constant-accuracy model cannot capture this; FABLE should."""
    rng = random.Random(seed)
    spans, truths = [], []
    for _ in range(n):
        rid, b = ("r1", 5) if rng.random() < 0.5 else ("r2", 10)
        f = rng.uniform(-1.0, 1.0)
        a_fp = 0.95 if f > 0 else 0.40
        votes = []
        for probe, a in (("fp", a_fp), ("stable", 0.8)):
            if rng.random() < a:
                v_rid, off = rid, b * 2.0
            else:
                (v_rid, wb) = ("r2", 10) if rid == "r1" else ("r1", 5)
                off = wb * 2.0
            votes.append(Vote(probe, "s", v_rid, off, 0.7, False, AbstainReason.NONE, (f,)))
        spans.append(votes); truths.append((rid, b))
    return spans, truths


def test_fable_learns_feature_dependent_accuracy():
    spans, _ = _synth_feature_dependent()
    m = Fable()
    m.fit(spans)
    hi = m.probe_accuracy_at((0.8,))["fp"]   # clean region
    lo = m.probe_accuracy_at((-0.8,))["fp"]  # FX region
    assert hi - lo > 0.25  # captures the swing a constant model misses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_fable.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `Fable` (adapt reference impl behind the protocol)**

Bring in the FABLE reference (EBCC + GP classifier over `features` + Pólya-Gamma VI, Lanczos low-rank per `2210.02724`), wrapped so `fit`/`predict_proba`/`probe_accuracy` match Task 3's protocol and `probe_accuracy_at` exposes the instance-conditioned accuracy. Add any new dependency (e.g. a GP lib) to `workspaces/pws_aligner/requirements.txt`, not the repo root.

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_fable.py -v`
Expected: PASS.

- [ ] **Step 5: Re-run the scorecard gate with FABLE**

Repeat Task 5 Step 6 with `Fable` swapped in for `DawidSkene`. Record BB11/BB12 numbers. FABLE must beat Dawid–Skene to justify its cost; if not, keep Dawid–Skene.

- [ ] **Step 6: Commit**

```bash
git add workspaces/pws_aligner/fable/ workspaces/pws_aligner/tests/test_fable.py workspaces/pws_aligner/requirements.txt
git commit -m "feat(pws): FABLE instance-feature label model [record vs Dawid-Skene in body]"
```

---

### Task 7: Confident-Learning verifier + GT-as-validation calibration

**Files:**
- Create: `workspaces/pws_aligner/verifier.py`
- Test: `workspaces/pws_aligner/tests/test_verifier.py`

**Interfaces:**
- Consumes: per-span soft labels from the label model (Task 3/6), GT loader `workspaces.alignment_prototype.score_timeline_vs_gt` (for the validation-only calibration report on BB11/BB12).
- Produces:
  - `prune_confident_errors(soft_labels, predicted_proba) -> tuple[set[str], "JointEstimate"]` — Confident-Learning joint estimate `Q_{ỹ,y*}` + the span_ids to prune/reweight (`1911.00068`).
  - `calibration_report(label_model, gt) -> dict[str, tuple[float, float]]` — per-probe (learned_accuracy, GT-measured_accuracy) on BB11/BB12; **validation only, never fed back into fit**. Flags probes whose learned precision diverges from the `probe_precision_transfer` measurements (e.g. fp's 0.90→0.53 swing).

- [ ] **Step 1: Write the failing test (error injection)**

```python
# workspaces/pws_aligner/tests/test_verifier.py
from __future__ import annotations
from workspaces.pws_aligner.verifier import prune_confident_errors


def test_prunes_injected_label_errors():
    # 100 spans; soft label = argmax; inject 10 flips the model is confident are wrong
    soft = {f"s{i}": ("A" if i % 2 == 0 else "B") for i in range(100)}
    proba = {sid: ({"A": 0.9, "B": 0.1} if lbl == "A" else {"A": 0.1, "B": 0.9})
             for sid, lbl in soft.items()}
    # flip 10 labels against a confident model prediction
    for i in range(0, 20, 2):
        soft[f"s{i}"] = "B"  # label says B but proba says A strongly
    to_prune, _ = prune_confident_errors(soft, proba)
    assert {f"s{i}" for i in range(0, 20, 2)} <= to_prune
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement Confident-Learning estimator**

Implement the confident-joint per `1911.00068`: per-class thresholds `t_j = mean predicted proba on spans labeled j`, count `C_{ỹ,y*}`, calibrate to marginals, prune off-diagonal by the PBC rule. Add `calibration_report` reading GT via the existing loader (validation only).

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_verifier.py -v`
Expected: PASS.

- [ ] **Step 5: Produce the calibration report on BB11/BB12**

```bash
venvs/audio/bin/python -m workspaces.pws_aligner.verifier --calibrate --sets 2nvzlh2k,1fsnxchk
```
Record the per-probe (learned, GT-measured) accuracy table. This closes the secondary success bar (learned precisions track `probe_precision_transfer`).

- [ ] **Step 6: Commit**

```bash
git add workspaces/pws_aligner/verifier.py workspaces/pws_aligner/tests/test_verifier.py
git commit -m "feat(pws): confident-learning verifier + GT calibration report"
```

---

## Self-Review

**Spec coverage:**
- LF layer (spec §II.2/L1) → Tasks 1–2 (votes + hypothesis space; existing probes reused, contract untouched — respects sensor freeze). Typed abstention → Task 1. *Open-vocab operation-typing is Phase 2, out of scope (spec §II.4).*
- Label model, no GT, replaces hand-tuned fusion (§II.2/L2) → Tasks 3 (Dawid–Skene) + 6 (FABLE), density gate → Task 4, instance-conditioning `Corr(X,LF)` gate → Task 6 pre-gate.
- Verifier + GT-as-validation (§II.2/L3, §II.6 secondary) → Task 7.
- Success bar / kill-gate (§II.6 primary) → Task 5.
- §II.10 decode-vs-end-model decision → resolved (soft labels → decode as priors) in Task 5.
- Risks: conditional-independence/correlation → Dawid–Skene handles per-probe reliability; explicit dependency **structure learning** is deferred to a Phase-1.5 note (flagged here as a known gap — the baseline treats probes as independent; if calibration (Task 7) shows fp+chroma double-counting, add Snorkel structure learning before trusting the posterior). Thin-density → Task 4 gate. FABLE cost → Task 6 gated on Task 5 lift.

**Placeholder scan:** no TBD/TODO; every code step shows code; FABLE (Task 6) intentionally adapts a reference implementation rather than hand-writing PG-VI — the *interface* and *acceptance test* are fully specified, which is the correct fidelity for a heavy numerical component.

**Type consistency:** `Vote`, `Hypothesis`, `AbstainReason`, `LabelModel.fit/predict_proba/probe_accuracy`, `probe_accuracy_at`, `posterior_to_placement` signatures are consistent across Tasks 1→7.

**Known deferred gap (not a spec miss):** Snorkel dependency **structure learning** between correlated probes (fp+chroma both "content") is not in Phase 1; the plan flags it as the first Phase-1.5 item, conditioned on the Task-7 calibration report showing over-confidence. This is the honest scope line — the baseline is sound to *test the falsifiable claim*, and structure learning is only worth its complexity if the data shows correlation is inflating confidence.
