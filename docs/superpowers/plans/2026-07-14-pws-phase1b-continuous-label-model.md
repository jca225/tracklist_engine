# PWS Phase-1b: Continuous Label Model + Operation LFs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the continuous label model (EM over per-probe Gaussian offset noise = *learned* inverse-variance fusion) that answers the Phase-1 kill-gate refutation, plus the first categorical operation LF (key-lock vs varispeed), and re-run the gate on BB12.

**Architecture:** Identity stays categorical (recordings ARE categorical — the sound part of Dawid–Skene); offsets get a continuous noise model: per probe, offset ~ inlier·N(μ_span, σ_probe²) + (1−inlier)·Uniform(±240 s), with per-probe identity accuracy, σ, and inlier rate learned by EM over unlabeled spans (no GT). Learned 1/σ² IS the `neuro/` inverse-variance fusion, made self-supervised. Operation LFs are a separate categorical lane where DS-style aggregation is well-matched.

**Tech Stack:** Python (stdlib `math`/`dataclasses` for the model — no numpy needed in the EM), pytest, existing `workspaces/pws_aligner` infrastructure (Vote, capture_votes, verifier, run_phase1), `librosa` for the operation LF only.

## Global Constraints

- Python: `venvs/audio/bin/python` from repo root; tests via `venvs/audio/bin/python -m pytest`.
- Style: `from __future__ import annotations`, full type hints, frozen dataclasses, pure functions, errors-as-values in core / fail-fast in CLIs (repo CLAUDE.md).
- **Offset frame invariant (LOAD-BEARING):** all `Vote.offset_s` values are RELATIVE (`ref_start_s − set_start_s`). `capture_votes.py` normalizes absolute-frame probes (`chroma`, `continuity`, `hubert`) at capture (commit 561aa7d). Never re-convert downstream.
- **No changes to `workspaces/alignment_prototype/`** (fork must beat it before promotion — `workspaces_dir` convention).
- **No new GT.** BB11 (`2nvzlh2k`) / BB12 (`1fsnxchk`) GT is validation/grading only.
- **No hand-typed headline numbers** outside `docs/alignment_status.md` machinery (SSOT rule). Gate verdicts go in `workspaces/pws_aligner/CLAUDE.md`.
- Commit after every task (project overrides "only commit when asked"). Branch: `pws-phase1b-continuous` (worktree off `pws-alignment-reframe`).
- Baseline numbers to beat (BB12, from the v2b gate — for reference in acceptance checks, do not re-type elsewhere): hand-tuned `source_priority` identity 84%, ref-offset median 14.0 s, strict trajectory 42%; refuted DS scored 32% / 33.6 s / 33%.

---

### Task 0: Unblock commits — reconcile the guardrails ratchet

The pre-commit hook fails at HEAD on `pws-alignment-reframe` (pre-existing from the 416558a merge): `raw_manifest_read` 99 > baseline 95, `parents_depth` 138 > 131. Every code commit in this plan will be blocked until reconciled.

**Files:**
- Modify: `scripts/guardrails_ratchet.json`
- (Possibly modify: whichever files introduced the new occurrences, if trivial)

**Interfaces:**
- Consumes: `scripts/guardrails.py` (the checker), `git log`/`git diff` archaeology.
- Produces: a passing `make check` baseline for all later tasks.

- [ ] **Step 1: Locate the new occurrences**

```bash
cd /path/to/repo
grep -rn --include='*.py' -E 'manifest\.json' workspaces/pws_aligner workspaces/streaming_mir scripts analysis ingest | wc -l
git diff 80d3140..416558a --stat | head -30
git diff 80d3140..416558a -S 'manifest.json' --name-only
git diff 80d3140..416558a -G 'parents\[[0-9]\]' --name-only
```
Expected: the offending files are in the phase-1 merge (likely `workspaces/pws_aligner/` and/or `workspaces/streaming_mir/` code).

- [ ] **Step 2: Decide fix vs ratchet per occurrence**

Rule: if an occurrence is a `Path(__file__).parents[N]` that would break under the refactor-safety conventions, fix it by anchoring on a module-level `_REPO_ROOT` constant already used in that package (follow the file's existing pattern). If the occurrences are deliberate (e.g., a workflow script that genuinely reads a manifest.json), raise the baseline instead.

- [ ] **Step 3: If raising baseline, edit `scripts/guardrails_ratchet.json`**

Set `raw_manifest_read` baseline to the current count (99) and `parents_depth` to (138), each with a `"justification"` field (the ratchet file's existing entries show the schema — mirror it):
```json
"raw_manifest_read": {"baseline": 99, "justification": "2026-07-14 pws-phase1 merge added deliberate manifest reads in workspaces/pws_aligner capture tooling; reviewed, not drive-by"}
```
(Adjust key names to match the file's actual schema — read it first.)

- [ ] **Step 4: Verify guardrails pass**

Run: `venvs/audio/bin/python scripts/guardrails.py`
Expected: `guardrails: 0 violation(s)` (exit 0).

- [ ] **Step 5: Commit**

```bash
git add scripts/guardrails_ratchet.json <any fixed files>
git commit -m "chore: reconcile guardrails ratchet after pws-phase1 merge"
```

---

### Task 1: `ContinuousLabelModel` — EM over (recording, offset)

**Files:**
- Create: `workspaces/pws_aligner/continuous_model.py`
- Test: `workspaces/pws_aligner/tests/test_continuous_model.py`

**Interfaces:**
- Consumes: `Vote`, `AbstainReason` from `workspaces/pws_aligner/votes.py` (fields: `probe: str`, `span_id: str`, `recording_id: str | None`, `offset_s: float` (RELATIVE), `confidence: float`, `abstained: bool`, `reason: AbstainReason`, `features: tuple[float, ...]`).
- Produces (used by Tasks 2–4):
  - `ProbeNoise(accuracy: float, sigma_s: float, inlier: float)` (frozen dataclass)
  - `FusedSpan(recording_id: str | None, offset_s: float, confidence: float, n_votes: int)` (frozen dataclass)
  - `class ContinuousLabelModel:` with `fit(spans: Sequence[Sequence[Vote]]) -> None`, `predict(span_votes: Sequence[Vote]) -> FusedSpan`, `probe_noise() -> dict[str, ProbeNoise]`. `predict` works pre-fit using documented init priors (so single-span smoke tests don't need a corpus).

- [ ] **Step 1: Write the failing tests**

```python
# workspaces/pws_aligner/tests/test_continuous_model.py
from __future__ import annotations

import random

from workspaces.pws_aligner.continuous_model import ContinuousLabelModel, FusedSpan
from workspaces.pws_aligner.votes import AbstainReason, Vote


def _vote(probe: str, span_id: str, rec: str | None, off: float,
          abstained: bool = False) -> Vote:
    return Vote(probe=probe, span_id=span_id, recording_id=rec, offset_s=off,
                confidence=0.8, abstained=abstained,
                reason=AbstainReason.NO_DATA if abstained else AbstainReason.NONE,
                features=())


# (accuracy, sigma_s, inlier) — heterogeneous by design: this is the regime
# that killed Dawid-Skene (fp ~0.2s vs chroma ~seconds across 2s bins).
_PROBES = {"sharp": (0.90, 0.3, 0.90), "mid": (0.80, 2.0, 0.85), "blurry": (0.60, 6.0, 0.80)}
_RECS = ("r1", "r2", "r3")


def _synthetic(n: int = 300, seed: int = 7):
    rng = random.Random(seed)
    spans, truths = [], []
    for i in range(n):
        true_r = rng.choice(_RECS)
        true_mu = rng.uniform(-120.0, 120.0)
        votes = []
        for name, (acc, sig, eta) in _PROBES.items():
            if rng.random() < 0.15:
                votes.append(_vote(name, f"s{i}", None, 0.0, abstained=True))
                continue
            if rng.random() < acc:
                off = rng.gauss(true_mu, sig) if rng.random() < eta else rng.uniform(-240, 240)
                votes.append(_vote(name, f"s{i}", true_r, off))
            else:
                wrong = rng.choice([r for r in _RECS if r != true_r])
                votes.append(_vote(name, f"s{i}", wrong, rng.uniform(-240, 240)))
        spans.append(tuple(votes))
        truths.append((true_r, true_mu))
    return spans, truths


def test_oracle_learns_sigma_ordering_and_identity():
    spans, truths = _synthetic()
    m = ContinuousLabelModel()
    m.fit(spans)
    noise = m.probe_noise()
    assert noise["sharp"].sigma_s < noise["mid"].sigma_s < noise["blurry"].sigma_s
    assert noise["sharp"].sigma_s < 1.0
    for name, (acc, _sig, _eta) in _PROBES.items():
        assert abs(noise[name].accuracy - acc) < 0.15
    correct = 0
    errs = []
    for votes, (true_r, true_mu) in zip(spans, truths):
        fused = m.predict(votes)
        if fused.recording_id == true_r:
            correct += 1
            errs.append(abs(fused.offset_s - true_mu))
    assert correct / len(spans) >= 0.85
    errs.sort()
    assert errs[len(errs) // 2] < 0.8  # median fused error ~ sharp-probe scale


def test_heterogeneous_precision_no_null_collapse():
    """THE regression for the v2b kill-gate: three probes agreeing on the
    recording but spread across different 2s bins must fuse, not go NULL."""
    votes = (
        _vote("fp", "s0", "rX", 100.1),
        _vote("hubert", "s0", "rX", 102.5),
        _vote("chroma", "s0", "rX", 104.5),
    )
    fused = ContinuousLabelModel().predict(votes)  # pre-fit: init priors
    assert fused.recording_id == "rX"
    assert abs(fused.offset_s - 100.1) < 1.5  # pulled toward the sharp probe


def test_all_abstain_is_null():
    votes = (_vote("fp", "s0", None, 0.0, abstained=True),)
    fused = ContinuousLabelModel().predict(votes)
    assert fused.recording_id is None
    assert fused.n_votes == 0


def test_single_vote_beats_null():
    fused = ContinuousLabelModel().predict((_vote("fp", "s0", "rY", 42.0),))
    assert fused.recording_id == "rY"
    assert abs(fused.offset_s - 42.0) < 1e-6


def test_unseen_probe_uses_default_priors():
    spans, _ = _synthetic(n=50)
    m = ContinuousLabelModel()
    m.fit(spans)
    fused = m.predict((_vote("brand_new_probe", "s0", "rZ", 10.0),))
    assert fused.recording_id == "rZ"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_continuous_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workspaces.pws_aligner.continuous_model'`

- [ ] **Step 3: Implement `continuous_model.py`**

```python
# workspaces/pws_aligner/continuous_model.py
"""Continuous label model: EM over (categorical recording, continuous offset).

Answer to the Phase-1 kill-gate (see CLAUDE.md): Dawid-Skene over 2s offset
bins was refuted because bin-agreement is the wrong granularity for continuous
offsets with heterogeneous probe precisions (fp ~0.2s vs chroma ~seconds) —
genuinely-right probes land in different bins, DS reads pervasive
disagreement, floors every accuracy, NULL wins.

Here identity stays categorical (recordings ARE categorical — that part of DS
was sound). Offsets are continuous: given an identity-correct vote,
    offset ~ inlier * N(mu_span, sigma_probe^2) + (1-inlier) * U(+-W).
EM learns per-probe (accuracy, sigma_s, inlier) with NO ground truth.
Learned 1/sigma^2 IS the neuro/ inverse-variance fusion, made self-supervised.

Frame invariant: all offsets are RELATIVE (ref_start_s - set_start_s), per
capture_votes.py normalization. Never re-convert here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .votes import Vote

_NULL_PRIOR_WEIGHT = 1.05          # mirror label_model.py
_OUTLIER_HALF_WIDTH_S = 240.0
_LOG_UNIFORM = -math.log(2.0 * _OUTLIER_HALF_WIDTH_S)
_UNIFORM = math.exp(_LOG_UNIFORM)
_SIGMA_FLOOR_S = 0.05
_CLAMP_LO, _CLAMP_HI = 0.02, 0.98
_SIGMA_INIT_S = {"fp": 0.5, "hubert": 3.0, "chroma": 8.0, "continuity": 8.0}
_SIGMA_DEFAULT_INIT_S = 5.0
_ACC_INIT = 0.7
_INLIER_INIT = 0.8
_MU_REFINE_ITERS = 3


@dataclass(frozen=True)
class ProbeNoise:
    accuracy: float   # P(recording vote correct | fires, truth != NULL)
    sigma_s: float    # offset noise std given identity-correct inlier
    inlier: float     # P(inlier | identity-correct)


@dataclass(frozen=True)
class FusedSpan:
    recording_id: str | None  # None = NULL
    offset_s: float           # fused RELATIVE offset (0.0 when NULL)
    confidence: float         # posterior mass on the MAP recording
    n_votes: int              # non-abstaining votes seen


def _log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _clamp(v: float) -> float:
    return min(_CLAMP_HI, max(_CLAMP_LO, v))


class ContinuousLabelModel:
    def __init__(self, max_iter: int = 50, tol: float = 1e-4) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._noise: dict[str, ProbeNoise] = {}
        self._fitted = False

    # -- parameters ---------------------------------------------------------

    def probe_noise(self) -> dict[str, ProbeNoise]:
        return dict(self._noise)

    def _noise_for(self, probe: str) -> ProbeNoise:
        got = self._noise.get(probe)
        if got is not None:
            return got
        return ProbeNoise(
            accuracy=_ACC_INIT,
            sigma_s=_SIGMA_INIT_S.get(probe, _SIGMA_DEFAULT_INIT_S),
            inlier=_INLIER_INIT,
        )

    # -- per-span inference -------------------------------------------------

    def _fused_mu(self, match: list[Vote]) -> tuple[float, list[float]]:
        """Robust precision-weighted mean over identity-matching votes.

        Returns (mu, per-vote inlier responsibilities gamma)."""
        weights = [1.0 / self._noise_for(v.probe).sigma_s ** 2 for v in match]
        mu = sum(w * v.offset_s for w, v in zip(weights, match)) / sum(weights)
        gammas = [1.0] * len(match)
        for _ in range(_MU_REFINE_ITERS):
            gammas = []
            for v in match:
                n = self._noise_for(v.probe)
                dens_in = n.inlier * math.exp(_log_normal_pdf(v.offset_s, mu, n.sigma_s))
                gammas.append(dens_in / (dens_in + (1.0 - n.inlier) * _UNIFORM))
            denom = sum(g * w for g, w in zip(gammas, weights))
            if denom <= 0.0:
                break
            mu = sum(g * w * v.offset_s for g, w, v in zip(gammas, weights, match)) / denom
        return mu, gammas

    def _span_eval(
        self, span_votes: Sequence[Vote]
    ) -> tuple[dict[str | None, float], dict[str, float], dict[str, list[float]]]:
        """Posterior over recordings (incl. NULL) + fused mu + inlier gammas."""
        fired = [v for v in span_votes if not v.abstained and v.recording_id]
        if not fired:
            return {None: 1.0}, {}, {}
        cands = list(dict.fromkeys(v.recording_id for v in fired))  # type: ignore[arg-type]
        k = len(cands)
        logpost: dict[str | None, float] = {}
        mus: dict[str, float] = {}
        gam: dict[str, list[float]] = {}

        lp_null = math.log(_NULL_PRIOR_WEIGHT)
        for v in fired:
            n = self._noise_for(v.probe)
            lp_null += math.log(1.0 - n.accuracy) - math.log(k) + _LOG_UNIFORM
        logpost[None] = lp_null

        for rec in cands:
            match = [v for v in fired if v.recording_id == rec]
            mu, gammas = self._fused_mu(match)
            mus[rec], gam[rec] = mu, gammas
            lp = 0.0
            for v in fired:
                n = self._noise_for(v.probe)
                if v.recording_id == rec:
                    dens = (n.inlier * math.exp(_log_normal_pdf(v.offset_s, mu, n.sigma_s))
                            + (1.0 - n.inlier) * _UNIFORM)
                    lp += math.log(n.accuracy) + math.log(max(dens, 1e-300))
                else:
                    lp += (math.log(1.0 - n.accuracy)
                           - math.log(max(k - 1, 1)) + _LOG_UNIFORM)
            logpost[rec] = lp

        peak = max(logpost.values())
        raw = {r: math.exp(lp - peak) for r, lp in logpost.items()}
        z = sum(raw.values())
        return {r: p / z for r, p in raw.items()}, mus, gam

    # -- EM -----------------------------------------------------------------

    def fit(self, spans: Sequence[Sequence[Vote]]) -> None:
        probes = {v.probe for span in spans for v in span if not v.abstained}
        self._noise = {p: self._noise_for(p) for p in probes}
        for _ in range(self.max_iter):
            num_correct: dict[str, float] = {p: 0.0 for p in probes}
            denom_fire: dict[str, float] = {p: 0.0 for p in probes}
            sq: dict[str, float] = {p: 0.0 for p in probes}
            gsum: dict[str, float] = {p: 0.0 for p in probes}
            msum: dict[str, float] = {p: 0.0 for p in probes}
            for span in spans:
                q, mus, gam = self._span_eval(span)
                fired = [v for v in span if not v.abstained and v.recording_id]
                non_null = 1.0 - q.get(None, 0.0)
                for v in fired:
                    denom_fire[v.probe] += non_null
                    q_r = q.get(v.recording_id, 0.0)
                    num_correct[v.probe] += q_r
                    if q_r > 0.0 and v.recording_id in mus:
                        match = [m for m in fired if m.recording_id == v.recording_id]
                        g = gam[v.recording_id][match.index(v)]
                        resid = v.offset_s - mus[v.recording_id]
                        sq[v.probe] += q_r * g * resid * resid
                        gsum[v.probe] += q_r * g
                        msum[v.probe] += q_r
            delta = 0.0
            for p in probes:
                old = self._noise[p]
                acc = _clamp(num_correct[p] / denom_fire[p]) if denom_fire[p] > 0 else old.accuracy
                sigma = (max(_SIGMA_FLOOR_S, math.sqrt(sq[p] / gsum[p]))
                         if gsum[p] > 0 else old.sigma_s)
                inl = _clamp(gsum[p] / msum[p]) if msum[p] > 0 else old.inlier
                delta = max(delta, abs(acc - old.accuracy),
                            abs(sigma - old.sigma_s), abs(inl - old.inlier))
                self._noise[p] = ProbeNoise(accuracy=acc, sigma_s=sigma, inlier=inl)
            if delta < self.tol:
                break
        self._fitted = True

    # -- prediction ----------------------------------------------------------

    def predict(self, span_votes: Sequence[Vote]) -> FusedSpan:
        q, mus, _gam = self._span_eval(span_votes)
        n_votes = sum(1 for v in span_votes if not v.abstained and v.recording_id)
        best = max(q, key=lambda r: q[r])
        if best is None:
            return FusedSpan(None, 0.0, q.get(None, 1.0), n_votes)
        return FusedSpan(best, mus[best], q[best], n_votes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_continuous_model.py -v`
Expected: 5 PASS. If `test_oracle_learns_sigma_ordering_and_identity` is flaky on tolerances, tighten the synthetic (n=500) rather than loosening asserts.

- [ ] **Step 5: Run the whole pws_aligner suite (no regressions)**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/ -q`
Expected: all pass (46 existing + 5 new).

- [ ] **Step 6: Commit**

```bash
git add workspaces/pws_aligner/continuous_model.py workspaces/pws_aligner/tests/test_continuous_model.py
git commit -m "feat(pws): continuous label model — EM over per-probe Gaussian offset noise"
```

---

### Task 2: Fused-placement bridge + `run_phase1 --model continuous`

**Files:**
- Modify: `workspaces/pws_aligner/decode_bridge.py` (add `fused_to_placement`)
- Modify: `workspaces/pws_aligner/run_phase1.py` (add `--model continuous` branch)
- Test: `workspaces/pws_aligner/tests/test_decode_bridge.py` (extend)

**Interfaces:**
- Consumes: `FusedSpan`, `ContinuousLabelModel` from Task 1; existing `posterior_to_placement()` in `decode_bridge.py`.
- Produces: `fused_to_placement(span_id: str, fused: FusedSpan) -> dict` emitting the **identical dict schema** as `posterior_to_placement` (same keys — read that function first and mirror it exactly), except `offset_s` carries the un-binned continuous value; plus a `<set_id>_pws_probe_noise.json` sidecar `{probe: {"accuracy":…, "sigma_s":…, "inlier":…}}`.

- [ ] **Step 1: Read the existing schema** — open `workspaces/pws_aligner/decode_bridge.py::posterior_to_placement` and `run_phase1.py`'s timeline-writing code; note the exact output keys. The new function must produce the same keys so `score_timeline_vs_gt.py` consumes it unchanged.

- [ ] **Step 2: Write the failing test** (adapt key names to what Step 1 found — the assertions below name the semantic content):

```python
# append to workspaces/pws_aligner/tests/test_decode_bridge.py
from workspaces.pws_aligner.continuous_model import FusedSpan
from workspaces.pws_aligner.decode_bridge import fused_to_placement


def test_fused_to_placement_keeps_unbinned_offset():
    placed = fused_to_placement("span_07", FusedSpan("rec_a", 101.37, 0.93, 3))
    assert placed["recording_id"] == "rec_a"
    assert abs(placed["offset_s"] - 101.37) < 1e-6   # NOT quantized to a 2s bin
    assert placed["confidence"] == 0.93
    assert not placed["abstained"]


def test_fused_to_placement_null_abstains():
    placed = fused_to_placement("span_07", FusedSpan(None, 0.0, 0.88, 0))
    assert placed["abstained"]
    assert placed["recording_id"] is None
```

- [ ] **Step 3: Run to verify failure**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_decode_bridge.py -v`
Expected: FAIL — `ImportError: cannot import name 'fused_to_placement'`

- [ ] **Step 4: Implement `fused_to_placement` in `decode_bridge.py`** (mirror `posterior_to_placement`'s exact keys; core shape):

```python
def fused_to_placement(span_id: str, fused: FusedSpan) -> dict:
    """Continuous-model analog of posterior_to_placement: same output schema,
    but offset_s is the un-binned fused value (the whole point of the
    continuous model is not to quantize)."""
    abstained = fused.recording_id is None
    return {
        # ... same keys posterior_to_placement emits, populated from fused ...
        "span_id": span_id,
        "recording_id": fused.recording_id,
        "offset_s": None if abstained else round(fused.offset_s, 3),
        "confidence": round(fused.confidence, 4),
        "abstained": abstained,
    }
```

- [ ] **Step 5: Wire `--model continuous` into `run_phase1.py`** — extend the existing model-selection argument (currently DS/MV via `choose_aggregator`); the continuous branch: `model = ContinuousLabelModel(); model.fit(all_votes)`; per span `fused_to_placement(span_id, model.predict(votes))`; write the usual `<set_id>_pws_timeline.json` plus `<set_id>_pws_probe_noise.json` from `model.probe_noise()`. The continuous model handles all densities natively (single-vote spans beat NULL structurally), so the density gate is bypassed for this branch — note that in a comment.

- [ ] **Step 6: Run the full suite**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add workspaces/pws_aligner/decode_bridge.py workspaces/pws_aligner/run_phase1.py workspaces/pws_aligner/tests/test_decode_bridge.py
git commit -m "feat(pws): fused-placement bridge + run_phase1 --model continuous"
```

---

### Task 3: Continuous calibration report (the standing tripwire)

The calibration bar caught both DS failures (learned fp .038 vs GT-measured .474). The continuous model needs the same tripwire: learned σ must track GT-measured residuals.

**Files:**
- Modify: `workspaces/pws_aligner/verifier.py` (add continuous report beside the existing `calibration_report`)
- Test: `workspaces/pws_aligner/tests/test_verifier.py` (extend)

**Interfaces:**
- Consumes: `ProbeNoise` from Task 1; GT placements in the same form the existing `calibration_report` consumes (read `verifier.py` first and reuse its GT-loading path — pass GT as `dict[span_id, tuple[recording_id, gt_offset_s]]` if that is what it uses, otherwise adapt to its actual type).
- Produces:
  - `ProbeCalibration(probe: str, learned_sigma_s: float, measured_mad_s: float | None, learned_accuracy: float, measured_accuracy: float | None, n_scored: int)` (frozen dataclass)
  - `continuous_calibration_report(noise: dict[str, ProbeNoise], spans: Sequence[Sequence[Vote]], gt: dict[str, tuple[str, float]]) -> list[ProbeCalibration]`
  - `sigma_rank_inversions(report: list[ProbeCalibration], min_n: int = 10) -> list[tuple[str, str]]` — pairs where learned σ ordering contradicts measured MAD ordering. Non-empty ⇒ the model is trusting the wrong probes (the DS failure signature).

- [ ] **Step 1: Write the failing test**

```python
# append to workspaces/pws_aligner/tests/test_verifier.py
from workspaces.pws_aligner.continuous_model import ProbeNoise
from workspaces.pws_aligner.verifier import (
    continuous_calibration_report, sigma_rank_inversions,
)


def _spans_with_known_residuals():
    # probe "tight": residuals ~0.1s; probe "loose": residuals ~5s; 12 spans each
    spans, gt = [], {}
    for i in range(12):
        sid = f"s{i}"
        gt[sid] = ("rec", 100.0)
        spans.append((
            _vote("tight", sid, "rec", 100.0 + (0.1 if i % 2 else -0.1)),
            _vote("loose", sid, "rec", 100.0 + (5.0 if i % 2 else -5.0)),
        ))
    return spans, gt


def test_report_measures_mad_per_probe():
    spans, gt = _spans_with_known_residuals()
    noise = {"tight": ProbeNoise(0.9, 0.2, 0.9), "loose": ProbeNoise(0.7, 6.0, 0.8)}
    report = {r.probe: r for r in continuous_calibration_report(noise, spans, gt)}
    assert abs(report["tight"].measured_mad_s - 0.1) < 1e-6
    assert abs(report["loose"].measured_mad_s - 5.0) < 1e-6
    assert report["tight"].n_scored == 12


def test_sigma_rank_inversion_tripwire():
    spans, gt = _spans_with_known_residuals()
    # Learned sigmas INVERTED vs measured: model trusts the loose probe more.
    noise = {"tight": ProbeNoise(0.9, 6.0, 0.9), "loose": ProbeNoise(0.7, 0.2, 0.8)}
    report = continuous_calibration_report(noise, spans, gt)
    assert sigma_rank_inversions(report) == [("tight", "loose")] or \
           sigma_rank_inversions(report) == [("loose", "tight")]
```

(Reuse the `_vote` helper from `test_continuous_model.py` — import it or duplicate the 6-line helper locally, matching the file's existing conventions.)

- [ ] **Step 2: Run to verify failure**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_verifier.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement in `verifier.py`**

```python
@dataclass(frozen=True)
class ProbeCalibration:
    probe: str
    learned_sigma_s: float
    measured_mad_s: float | None
    learned_accuracy: float
    measured_accuracy: float | None
    n_scored: int


def continuous_calibration_report(
    noise: dict[str, ProbeNoise],
    spans: Sequence[Sequence[Vote]],
    gt: dict[str, tuple[str, float]],
) -> list[ProbeCalibration]:
    residuals: dict[str, list[float]] = defaultdict(list)
    id_hits: dict[str, list[bool]] = defaultdict(list)
    for span in spans:
        for v in span:
            if v.abstained or v.recording_id is None or v.span_id not in gt:
                continue
            gt_rec, gt_off = gt[v.span_id]
            hit = v.recording_id == gt_rec
            id_hits[v.probe].append(hit)
            if hit:
                residuals[v.probe].append(abs(v.offset_s - gt_off))
    out = []
    for probe, n in sorted(noise.items()):
        res = sorted(residuals.get(probe, []))
        hits = id_hits.get(probe, [])
        out.append(ProbeCalibration(
            probe=probe,
            learned_sigma_s=n.sigma_s,
            measured_mad_s=res[len(res) // 2] if res else None,
            learned_accuracy=n.accuracy,
            measured_accuracy=(sum(hits) / len(hits)) if hits else None,
            n_scored=len(res),
        ))
    return out


def sigma_rank_inversions(
    report: list[ProbeCalibration], min_n: int = 10
) -> list[tuple[str, str]]:
    scored = [r for r in report if r.measured_mad_s is not None and r.n_scored >= min_n]
    bad = []
    for i, a in enumerate(scored):
        for b in scored[i + 1:]:
            learned = a.learned_sigma_s - b.learned_sigma_s
            measured = a.measured_mad_s - b.measured_mad_s  # type: ignore[operator]
            if learned * measured < 0:
                bad.append((a.probe, b.probe))
    return bad
```

- [ ] **Step 4: Run tests, full suite, commit**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/ -q` → all pass.

```bash
git add workspaces/pws_aligner/verifier.py workspaces/pws_aligner/tests/test_verifier.py
git commit -m "feat(pws): continuous calibration report + sigma rank-inversion tripwire"
```

---

### Task 4: Gate v3 — continuous model vs hand-tuned fusion on BB12

**Files:**
- Modify: `workspaces/pws_aligner/CLAUDE.md` (verdict)
- Create: gate artifacts (`1fsnxchk_pws_timeline.json`, `1fsnxchk_pws_probe_noise.json` — wherever run_phase1 writes them today)

**Interfaces:**
- Consumes: everything above; the genuine-votes file from the v2b gate (find it: `ls workspaces/pws_aligner/*votes*.json` or the path recorded in the v2b worktree ledger; if absent, regenerate with `capture_votes.py` — it runs the real harness probes per span; BB12 only, do NOT run BB11 lyrics/Whisper paths).
- Produces: gate verdict written into `workspaces/pws_aligner/CLAUDE.md`.

- [ ] **Step 1: Locate or regenerate genuine BB12 votes**

```bash
ls workspaces/pws_aligner/ | grep -i vote
# if absent:
venvs/audio/bin/python -m workspaces.pws_aligner.capture_votes --set-id 1fsnxchk  # check its --help for exact args
```

- [ ] **Step 2: Run the continuous model end-to-end**

```bash
venvs/audio/bin/python -m workspaces.pws_aligner.run_phase1 --model continuous <votes-file args per its --help>
```
Expected: timeline JSON + probe-noise sidecar written; zero spans crash.

- [ ] **Step 3: Score against GT**

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt --set-id 1fsnxchk <timeline arg per its --help>
```
Record: identity %, ref-offset median (s), strict trajectory %, NULL/abstain span count.

- [ ] **Step 4: Calibration tripwire**

Run `continuous_calibration_report` + `sigma_rank_inversions` on the fitted noise vs BB12 GT (small runner inline in `run_phase1.py --calibrate` or a 20-line script beside it). Expected: **zero rank inversions**; learned σ ordering must match measured MAD ordering (fp tightest). If inversions fire, the gate FAILS regardless of scorecard numbers (right-answer-for-wrong-reasons guard).

- [ ] **Step 5: Verdict — three explicit outcomes** (write into `workspaces/pws_aligner/CLAUDE.md`, dated, alongside the v1/v2b verdicts; update memory `project_pws_gate_verdict` accordingly):

  - **PASS:** ≥ hand-tuned fusion on ≥2 of {identity, ref-offset median, strict traj} AND no calibration inversions AND NULL-abstain rate well below v2b's ~62%. → next step is BB11 held-out confirmation, then promotion discussion.
  - **PARTIAL:** beats DS decisively, approaches hand fusion, calibration clean. → iterate (per-axis σ? instance-conditioning per FABLE gate) — still Phase-1b, document levers.
  - **FAIL:** loses broadly or calibration inverted. → STOP again, document honestly; the informative-null path (density/independence diagnosis) is the deliverable.

- [ ] **Step 6: Commit** (artifacts per repo convention — timelines are usually gitignored; commit the CLAUDE.md verdict + any runner code):

```bash
git add workspaces/pws_aligner/CLAUDE.md workspaces/pws_aligner/run_phase1.py
git commit -m "eval(pws): gate v3 — continuous label model vs hand fusion on BB12 (verdict inside)"
```

---

### Task 5: First categorical operation LF — key-lock vs varispeed

The pivotal discriminator from the spec (§I.4 keystone) and the ontology research: varispeed ⇒ pitch shift = 12·log₂(r) coupled to tempo ratio r; key-lock ⇒ r ≠ 1 with pitch preserved. Genuinely categorical ⇒ DS-style aggregation is well-matched here (this lane, unlike offsets, keeps the categorical machinery).

**Files:**
- Create: `workspaces/pws_aligner/operations.py`
- Test: `workspaces/pws_aligner/tests/test_operations.py`

**Interfaces:**
- Consumes: `AbstainReason` from `votes.py`; `librosa`, `numpy` (already in `venvs/audio`).
- Produces:
  - `class TempoPitchLabel(Enum): KEYLOCK / VARISPEED / NO_TEMPO_CHANGE`
  - `OperationVote(lf: str, span_id: str, label: str, confidence: float, abstained: bool, reason: AbstainReason)` (frozen dataclass)
  - `estimate_pitch_shift_semitones(mix: np.ndarray, ref: np.ndarray, sr: int) -> float` (CQT-profile cross-correlation, 1/3-semitone resolution)
  - `keylock_vs_varispeed(span_id: str, mix: np.ndarray, ref: np.ndarray, sr: int, tempo_ratio: float) -> OperationVote`

- [ ] **Step 1: Write the failing test** (synthetic — librosa provides both transforms, no repo audio needed):

```python
# workspaces/pws_aligner/tests/test_operations.py
from __future__ import annotations

import librosa
import numpy as np

from workspaces.pws_aligner.operations import (
    TempoPitchLabel, keylock_vs_varispeed,
)

_SR = 22050


def _harmonic_tone(seconds: float = 6.0, f0: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * _SR)) / _SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    # amplitude modulation so time-stretch has structure to preserve
    return (y * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t))).astype(np.float32)


def test_varispeed_detected():
    ref = _harmonic_tone()
    r = 1.06
    mix = librosa.resample(ref, orig_sr=_SR, target_sr=int(_SR / r))  # play fast: pitch+tempo up
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=r)
    assert vote.label == TempoPitchLabel.VARISPEED.value
    assert not vote.abstained


def test_keylock_detected():
    ref = _harmonic_tone()
    r = 1.06
    mix = librosa.effects.time_stretch(ref, rate=r)  # tempo up, pitch preserved
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=r)
    assert vote.label == TempoPitchLabel.KEYLOCK.value


def test_no_tempo_change():
    ref = _harmonic_tone()
    vote = keylock_vs_varispeed("s0", ref, ref, _SR, tempo_ratio=1.0)
    assert vote.label == TempoPitchLabel.NO_TEMPO_CHANGE.value


def test_keyshift_on_top_abstains():
    # pitch moved but matches NEITHER pattern (keylock + deliberate key-shift):
    ref = _harmonic_tone()
    mix = librosa.effects.pitch_shift(ref, sr=_SR, n_steps=3.0)
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=1.06)
    assert vote.abstained
```

- [ ] **Step 2: Run to verify failure**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_operations.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `operations.py`**

```python
# workspaces/pws_aligner/operations.py
"""Categorical operation LFs (lane 2 of Phase-1b).

Unlike offsets (continuous — see continuous_model.py), operation-type labels
are genuinely categorical, so DS-style aggregation IS well-matched here.
First LF: the keystone key-lock vs varispeed discriminator —
  varispeed: pitch shift == 12*log2(tempo_ratio)  (pitch/tempo coupled)
  key-lock:  tempo_ratio != 1 but pitch preserved (Master Tempo, 2001+)
Detector signatures grounded in docs/dj_operation_ontology_research.md §2.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import librosa
import numpy as np

from .votes import AbstainReason

_BINS_PER_OCTAVE = 36  # 1/3-semitone CQT resolution
_N_BINS = 6 * _BINS_PER_OCTAVE
_FMIN = 55.0
_TEMPO_EPS = 0.02          # |r-1| below this = no meaningful tempo change
_PITCH_TOL_ST = 0.5        # tolerance on pitch-shift match, semitones
_MAX_SHIFT_ST = 12.0


class TempoPitchLabel(Enum):
    KEYLOCK = "keylock"
    VARISPEED = "varispeed"
    NO_TEMPO_CHANGE = "no_tempo_change"


@dataclass(frozen=True)
class OperationVote:
    lf: str
    span_id: str
    label: str
    confidence: float
    abstained: bool
    reason: AbstainReason


def _cqt_profile(y: np.ndarray, sr: int) -> np.ndarray:
    c = np.abs(librosa.cqt(y, sr=sr, fmin=_FMIN,
                           n_bins=_N_BINS, bins_per_octave=_BINS_PER_OCTAVE))
    prof = np.log1p(c).mean(axis=1)
    prof -= prof.mean()
    return prof


def estimate_pitch_shift_semitones(mix: np.ndarray, ref: np.ndarray, sr: int) -> float:
    """Pitch shift of mix relative to ref via CQT-profile cross-correlation."""
    pm, pr = _cqt_profile(mix, sr), _cqt_profile(ref, sr)
    max_lag = int(_MAX_SHIFT_ST * _BINS_PER_OCTAVE / 12)
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = pm[lag:], pr[: len(pr) - lag]
        else:
            a, b = pm[:lag], pr[-lag:]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag * 12.0 / _BINS_PER_OCTAVE


def keylock_vs_varispeed(
    span_id: str, mix: np.ndarray, ref: np.ndarray, sr: int, tempo_ratio: float
) -> OperationVote:
    lf = "keylock_vs_varispeed"
    if abs(tempo_ratio - 1.0) < _TEMPO_EPS:
        return OperationVote(lf, span_id, TempoPitchLabel.NO_TEMPO_CHANGE.value,
                             0.9, False, AbstainReason.NONE)
    shift = estimate_pitch_shift_semitones(mix, ref, sr)
    expected = 12.0 * math.log2(tempo_ratio)
    if abs(shift - expected) < max(_PITCH_TOL_ST, 0.25 * abs(expected)):
        return OperationVote(lf, span_id, TempoPitchLabel.VARISPEED.value,
                             0.8, False, AbstainReason.NONE)
    if abs(shift) < _PITCH_TOL_ST:
        return OperationVote(lf, span_id, TempoPitchLabel.KEYLOCK.value,
                             0.8, False, AbstainReason.NONE)
    # pitch moved but matches neither pattern (e.g. key-lock + key-shift):
    return OperationVote(lf, span_id, TempoPitchLabel.KEYLOCK.value,
                         0.0, True, AbstainReason.LOW_MARGIN)
```

Note: for small r (1.06 ⇒ 1.01 st) the varispeed check needs the 1/3-semitone
CQT resolution — that is why `_BINS_PER_OCTAVE = 36`. If `test_varispeed_detected`
fails marginally, raise to 60 bins/octave before touching tolerances.

- [ ] **Step 4: Run tests, full suite**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_operations.py -v` → 4 PASS.
Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add workspaces/pws_aligner/operations.py workspaces/pws_aligner/tests/test_operations.py
git commit -m "feat(pws): keylock-vs-varispeed operation LF (first categorical lane LF)"
```

---

### Task 6: Operation-LF runner on BB12 (analysis, no gate)

**Files:**
- Create: `workspaces/pws_aligner/run_operations.py`

**Interfaces:**
- Consumes: `keylock_vs_varispeed` from Task 5; span enumeration + audio loading from `capture_votes.py` (reuse its loaders — read that file and import its span/audio helpers rather than re-implementing).
- Produces: `<set_id>_operation_votes.json` — list of OperationVote dicts per GT span, plus a printed label histogram. **Analysis only — no accuracy claim** (GT-derived operation labels are future work; per-clip GT stretch extraction is not yet plumbed).

- [ ] **Step 1: Implement the runner** — mirror `capture_votes.py`'s CLI shape (`--set-id`, same span source). For each GT span with a placed ref: load mix-span audio + time-aligned ref segment via capture_votes' helpers, get tempo_ratio from the harness result if it carries one (`AlignmentResult.tempo_ratio`) else 1.0, call `keylock_vs_varispeed`, collect votes, dump JSON, print histogram of labels + abstain rate.

- [ ] **Step 2: Run on BB12**

```bash
venvs/audio/bin/python -m workspaces.pws_aligner.run_operations --set-id 1fsnxchk
```
Expected: JSON written; histogram printed; no crashes. Sanity expectations (not asserts): mashup-heavy BB12 should show a real KEYLOCK/VARISPEED mixture, not 100% one label (per the 31%-repitched-acappellas memory the varispeed/repitch class must be non-empty).

- [ ] **Step 3: Record the histogram** in `workspaces/pws_aligner/CLAUDE.md` under a "lane 2 — operations" heading (counts only, no accuracy claims).

- [ ] **Step 4: Commit**

```bash
git add workspaces/pws_aligner/run_operations.py workspaces/pws_aligner/CLAUDE.md
git commit -m "feat(pws): operation-LF runner — keylock/varispeed histogram on BB12 spans"
```

---

### Task 7: File the two aligner bug tickets (independent of PWS lane)

From the failure tables (`docs/archive/failure_tables_bb11_bb12_20260714.md`) — worth tickets regardless of how the gate goes.

**Files:** none (GitHub issues via `gh`).

- [ ] **Step 1: File ticket 1**

```bash
gh issue create \
  --title "aligner: 'ref≈0:00 intro-grab' decode collapse" \
  --body "Biggest failure pattern in docs/archive/failure_tables_bb11_bb12_20260714.md: decode locks onto the ref intro (offset ≈ 0:00) regardless of the true position. Hypothesis: low-information/self-similar intro frames let matched filters peak at 0. Candidate fixes: (a) penalize offset≈0 unless corroborated by a second channel; (b) exclude low-novelty ref intro region from peak search; (c) require fp corroboration for offsets < 10s. Acceptance: re-score the failure-table rows tagged intro-grab on BB11/BB12."
```

- [ ] **Step 2: File ticket 2**

```bash
gh issue create \
  --title "aligner: repeated-track instance disambiguation (BB12 'Slide' −746s)" \
  --body "From docs/archive/failure_tables_bb11_bb12_20260714.md: when a track (or its self-similar fiber) appears multiple times, decode attributes all spans to ONE anchor — BB12 'Slide' produced four spans from one anchor with a −746s error. Needs instance-aware assignment: per-span independent offset hypotheses, fiber-aware anchor splitting (fibers/ already classifies self-repeats), or a decode-level constraint that distinct mix spans may map to distinct ref instances. Acceptance: BB12 Slide spans resolve to their own anchors."
```

- [ ] **Step 3: Note the issue numbers** in `docs/archive/failure_tables_bb11_bb12_20260714.md` (one line at top: "Tracked: #N, #M") and commit that one-liner.

```bash
git add docs/archive/failure_tables_bb11_bb12_20260714.md
git commit -m "docs: link failure tables to filed aligner tickets"
```

---

## Out of scope (YAGNI — per spec §II.9 and the gate verdict)

- FABLE instance-conditioning (correctly gated off until `Corr(X, LF-correctness)` justifies it; the continuous σ is feature-flat in this cycle).
- Span-embedding cache (Phase-2 prerequisite; not needed for the continuous model).
- Phase-2 open-vocab discovery loop.
- More operation LFs beyond keylock/varispeed (loop-flag via fibers etc. — next cycle, after the lane-2 plumbing exists).
- Any change to `alignment_prototype` or new GT.

## Self-review notes

- Spec coverage: gate-verdict lever (continuous EM σ) → Tasks 1–4; user's lane 2 (categorical ontology LFs) → Tasks 5–6; the "two bug tickets" → Task 7; calibration-as-tripwire (survivor of v2b) → Task 3; density gate → subsumed (continuous model handles single-vote spans natively; documented in Task 2 Step 5).
- Types consistent: `ProbeNoise`/`FusedSpan` defined Task 1, consumed Tasks 2–4 with matching signatures; `OperationVote` defined Task 5, consumed Task 6.
- Known unknowns made explicit rather than hidden: exact `posterior_to_placement` keys (Task 2 Step 1 reads first), `capture_votes`/`run_phase1` CLI args (Task 4 checks `--help`), ratchet JSON schema (Task 0 Step 3 reads first).
