# André-Absorption Phase 2 — Resample-Ratio Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the fingerprint front-end a resample-ratio search so it can follow the pitch+tempo-coupled `resample` transform (the one axis where the specialist NMF currently leads), and land the two correctness carries from Phase 0's final review.

**Architecture:** UnmixDB's `resample` shifts pitch AND tempo by one ratio. The existing `fp_offset` only *time-stretches* the ref (pitch-preserving), so it is structurally blind to resample. Add `fp_offset_resample` to `landmark_fp.py` that resamples the ref by a ratio grid using fast `librosa.resample` (interpolation, NOT phase-vocoder — cheap), which physically models the transform. Wire it as a new `eval_bench` method and measure it against plain `fused` on the resample stratum. First, fix the two Phase-0 carries so identity can be measured honestly alongside.

**Tech Stack:** Python 3, numpy, librosa, pandas, `venvs/audio/bin/python`.

## Global Constraints

- Run from repo root `/Users/johnnycabrahams/Desktop/tracklist_engine` with `venvs/audio/bin/python`.
- Test command form: `venvs/audio/bin/python -m pytest <path> -v`
- Branch first: `git checkout -b andre-absorption-phase2` before Task 1.
- Sensor-phase freeze is LIFTED for this work (John, 2026-07-11) — new channels are allowed.
- `landmark_fp` constants: `SR=22050`, `FHOP=512`. `fp_offset` convention: recovered offset seconds = `off_frames * FHOP / SR * scale`. Mirror this exactly for the resample scale factor.
- Abstain sentinel is a `Pred` with `set_start_s = float("nan")` (from Phase 0).
- Tests live in `alignment/external/tests/test_eval_bench.py` (eval_bench changes) and a new `alignment/tests/test_landmark_fp_resample.py` (landmark_fp change — create dir + `__init__.py` in Task 3).
- Every task keeps `venvs/audio/bin/python -m alignment.external.eval_bench --synthetic` green.
- Numbers written into any findings doc MUST come from a real run's stdout — never fabricated.

---

### Task 1: Fix `method_dtw` confidence — path-average, not best-cell

**Files:**
- Modify: `alignment/external/eval_bench.py` (`method_dtw`)
- Test: `alignment/external/tests/test_eval_bench.py`

**Interfaces:**
- Consumes: existing `method_dtw`, `Pred`, `Sample`, `GTSpan`.
- Produces: `method_dtw` whose `Pred.score` is `1 - mean(cost along the warping path)` instead of `1 - C.min()`. Signature unchanged.

**Why:** Phase-0 final review flagged `1 - C.min()` as the single best-matching cell anywhere in the cost matrix — path-length-invariant and optimistic. It would mislead if DTW enters identity ranking. Path-average cost is the honest confidence.

- [ ] **Step 1: Write the failing test**

```python
# append to test_eval_bench.py
def test_method_dtw_score_is_path_average_not_best_cell():
    # A track that matches the mix over its planted region but has a few
    # perfect-match frames should NOT report score ~1.0 (best-cell); it should
    # reflect the AVERAGE path cost, which is < 1.0 given imperfect frames.
    rng = np.random.default_rng(1)
    D, tlen, Tm = 12, 120, 800
    tf = rng.random((D, tlen)).astype(np.float32)
    mix = rng.random((D, Tm)).astype(np.float32) * 0.05
    start_f = 300
    mix[:, start_f : start_f + tlen] += tf
    # make 3 mix frames identical to their track frame (perfect cells), rest noisy
    s = Sample("s", mix, {0: tf}, [GTSpan(0, start_f * HOP / SR, 1.0)])
    preds = method_dtw(s)
    # best-cell would be ~1.0; path-average over a noisy-background match is lower
    assert preds[0].score < 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest alignment/external/tests/test_eval_bench.py::test_method_dtw_score_is_path_average_not_best_cell -v`
Expected: FAIL — current score is `1 - C.min()` ≈ 1.0, so `< 0.95` fails.

- [ ] **Step 3: Implement the path-average score**

In `method_dtw`, replace the final `out[idx] = Pred(...)` line. The warping path is already computed as `wp` (reversed to ascending, columns `track_f = wp[:,0]`, `mix_f = wp[:,1]`). Compute the mean cost along the path:

```python
        path_cost = float(C[wp[:, 0], wp[:, 1]].mean())
        out[idx] = Pred(max(0.0, set_start), tempo, 1.0 - path_cost)
```

(Delete the old `1.0 - float(C.min())`.)

- [ ] **Step 4: Run tests to verify pass**

Run: `venvs/audio/bin/python -m pytest alignment/external/tests/test_eval_bench.py -v`
Expected: PASS — all tests including the two prior DTW tests (planted-offset recovery, tiny-track abstain) still pass and the new path-average test passes.

- [ ] **Step 5: Commit**

```bash
git add alignment/external/eval_bench.py \
        alignment/external/tests/test_eval_bench.py
git commit -m "fix(eval_bench): method_dtw score = 1 - path-average cost (was best-cell)"
```

---

### Task 2: Fix identity-under-abstention — don't count a decline as wrong

**Files:**
- Modify: `alignment/external/eval_bench.py` (`score_sample` identity block)
- Test: `alignment/external/tests/test_eval_bench.py`

**Interfaces:**
- Consumes: `score_sample`, `is_abstain`, `Pred`, `Sample`, `GTSpan`.
- Produces: `score_sample`'s identity accuracy (`id_ok`) computed over **committed (non-abstained) GT spans only**. An abstained true-track span is excluded from the identity denominator, not scored as a miss.

**Why:** Phase-0 final review: under open-mode with distractors, abstained spans would silently depress `identity_acc` — conflating "declined" with "mis-identified." Identity should be measured only where the method committed.

- [ ] **Step 1: Write the failing test**

```python
# append to test_eval_bench.py
def test_identity_excludes_abstained_spans():
    # one committed correct span + one abstained span, with a distractor the
    # committed track out-scores. Identity should be 1.0 (1/1 committed), not
    # 0.5 (1/2 counting the abstention as a miss).
    mix = np.ones((12, 400), dtype=np.float32)
    tfa = np.ones((12, 40), dtype=np.float32)
    tfb = np.ones((12, 40), dtype=np.float32)
    s = Sample(
        "m", mix, {0: tfa, 1: tfb},
        [GTSpan(0, 1.0, 1.0), GTSpan(1, 2.0, 1.0)],
        distractor_feats={"d0": np.zeros((12, 40), np.float32)},
    )
    preds = {
        0: Pred(1.0, 1.0, 100.0),                 # committed, high score
        1: Pred(float("nan"), float("nan"), 3.0), # abstained
    }
    _, id_ok = score_sample(s, preds)
    assert id_ok == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest alignment/external/tests/test_eval_bench.py::test_identity_excludes_abstained_spans -v`
Expected: FAIL — current code counts the abstained span in the denominator (`hits / len(sample.gt)` = 1/2 = 0.5), and also compares `p.score > best_dist` on the abstained pred.

- [ ] **Step 3: Implement committed-only identity**

In `score_sample`, the identity block currently reads (inside `if sample.distractor_feats:`):

```python
        hits = sum(
            int((p := preds.get(sp.track_idx)) is not None and p.score > best_dist)
            for sp in sample.gt
        )
        id_ok = hits / max(1, len(sample.gt))
```

Replace it with a committed-only count:

```python
        committed_gt = [
            sp for sp in sample.gt
            if (p := preds.get(sp.track_idx)) is not None and not is_abstain(p)
        ]
        hits = sum(
            int(preds[sp.track_idx].score > best_dist) for sp in committed_gt
        )
        id_ok = hits / max(1, len(committed_gt))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `venvs/audio/bin/python -m pytest alignment/external/tests/test_eval_bench.py -v`
Expected: PASS — the new test plus all prior tests.

Also smoke: `venvs/audio/bin/python -m alignment.external.eval_bench --synthetic`
Expected: table prints, no traceback.

- [ ] **Step 5: Commit**

```bash
git add alignment/external/eval_bench.py \
        alignment/external/tests/test_eval_bench.py
git commit -m "fix(eval_bench): identity over committed spans only (abstain != mis-id)"
```

---

### Task 3: `fp_offset_resample` — resample-ratio fingerprint search

**Files:**
- Modify: `alignment/landmark_fp.py` (add function)
- Create: `alignment/tests/__init__.py` (empty)
- Test: `alignment/tests/test_landmark_fp_resample.py`

**Interfaces:**
- Consumes: `constellation`, `hashes`, `_vote_histogram`, `vote_sharpness`, `SR`, `FHOP` (all in `landmark_fp`).
- Produces: `fp_offset_resample(mix_y, ref_y, *, ratios=...) -> tuple[float, int, float, float]` returning `(ref_start_s, votes, ratio, sharpness)`. Resamples the ref by each ratio `r` with `librosa.resample(ref_y, orig_sr=SR, target_sr=int(round(SR / r)))` (fast interpolation, not phase-vocoder), fingerprints it, votes against the mix, keeps the ratio with the most votes. Offset seconds = `off_frames * FHOP / SR * r` (mirrors `fp_offset`'s stretch-scale convention).

**Note on direction (implementer must verify):** the offset/ratio sign convention here is validated behaviorally by the synthetic test below (which plants a known ratio) and again on real GT in Task 5. If the planted-ratio test cannot pass with `target_sr = SR / r`, try `target_sr = SR * r` and the reciprocal offset scale — pick whichever recovers the planted ratio, and note which in your report.

- [ ] **Step 1: Write the failing test**

```python
# alignment/tests/test_landmark_fp_resample.py
import numpy as np
import librosa
from alignment.landmark_fp import SR, fp_offset_resample


def _tone_complex(dur_s: float, freqs, sr: int = SR) -> np.ndarray:
    """A signal with stable spectral-peak landmarks: sustained tones gated by
    0.25 s onsets (gives both frequency and time structure for the constellation)."""
    t = np.arange(int(dur_s * sr)) / sr
    y = np.zeros_like(t)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t)
    env = ((t * 4.0) % 1.0 < 0.5).astype(np.float64)  # 4 Hz on/off gate
    return (y * env).astype(np.float32)


def test_fp_offset_resample_recovers_planted_ratio_and_offset():
    ref = _tone_complex(20.0, [400, 900, 1700, 3100, 5200])
    ratio = 1.10  # mix track is sped up 10%: pitch+tempo x1.10
    # simulate the resampled track as it appears in the mix
    mix_track = librosa.resample(ref, orig_sr=SR, target_sr=int(round(SR / ratio)))
    lead = int(3.0 * SR)  # planted 3 s into the mix
    mix = np.zeros(lead + len(mix_track) + int(2.0 * SR), dtype=np.float32)
    mix[lead : lead + len(mix_track)] += mix_track
    ratios = tuple(round(1.0 + 0.02 * k, 3) for k in range(-13, 17))  # ~0.74..1.32
    ref_start_s, votes, r_hat, sharp = fp_offset_resample(mix, ref, ratios=ratios)
    assert votes > 0
    assert abs(r_hat - ratio) <= 0.03          # within one grid step
    assert abs(ref_start_s - 3.0) < 1.0        # planted lead recovered
```

**Fixture latitude:** if the constellation does not fire enough votes on this tone-complex to recover the ratio, enrich the fixture (add more partials, a click/impulse train for sharper time landmarks, or lengthen it) — but keep the three behavioral assertions (`votes > 0`, ratio within one step, offset within 1 s) exactly. Note any fixture change in your report.

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest alignment/tests/test_landmark_fp_resample.py -v`
Expected: FAIL with `ImportError: cannot import name 'fp_offset_resample'`

- [ ] **Step 3: Implement `fp_offset_resample`**

Add to `landmark_fp.py` after `fp_offset`:

```python
def fp_offset_resample(
    mix_y: np.ndarray,
    ref_y: np.ndarray,
    *,
    ratios: tuple[float, ...] = tuple(round(1.0 + 0.02 * k, 3) for k in range(-13, 17)),
) -> tuple[float, int, float, float]:
    """Resample-ratio fingerprint offset — for the RESAMPLE transform (pitch AND
    tempo coupled by one ratio). Unlike `fp_offset` (time-stretch, pitch-preserving),
    this resamples the ref by each ratio `r` with fast interpolation (not a
    phase-vocoder), so a pitch-shifted alignment diagonal re-registers.

    Returns (ref_start_s, votes, ratio, sharpness). A mix track resampled to speed r
    has pitch+tempo x r; resampling the ref to target_sr = SR / r reproduces that
    speed-r playback when read back at SR. Recovered offset scales by r, mirroring
    `fp_offset`'s stretch-scale convention.
    """
    import librosa

    hm = hashes(*constellation(mix_y))
    best = (0.0, 0, 1.0, 0.0)
    for r in ratios:
        if abs(r - 1.0) < 1e-3:
            ry = ref_y
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ry = librosa.resample(ref_y, orig_sr=SR, target_sr=int(round(SR / r)))
        votes = _vote_histogram(hm, hashes(*constellation(ry)))
        if not votes:
            continue
        off, v = max(votes.items(), key=lambda kv: kv[1])
        if v > best[1]:
            best = (off * FHOP / SR * r, v, r, vote_sharpness(votes))
    return best
```

- [ ] **Step 4: Run test to verify pass**

Run: `venvs/audio/bin/python -m pytest alignment/tests/test_landmark_fp_resample.py -v`
Expected: PASS (recovers ratio within one grid step and offset within 1 s). If it fails only on fixture weakness, enrich per the latitude note and re-run.

- [ ] **Step 5: Commit**

```bash
git add alignment/landmark_fp.py \
        alignment/tests/__init__.py \
        alignment/tests/test_landmark_fp_resample.py
git commit -m "feat(landmark_fp): fp_offset_resample — resample-ratio search for pitch+tempo shift"
```

---

### Task 4: `method_fused_resample` in eval_bench

**Files:**
- Modify: `alignment/external/eval_bench.py`
- Test: `alignment/external/tests/test_eval_bench.py`

**Interfaces:**
- Consumes: `fp_offset_resample` (Task 3), `Sample`, `Pred`, `SR`.
- Produces: `method_fused_resample(sample) -> dict[int, Pred]` registered in `METHODS` as `"fused_resample"`. For each candidate: `off, votes, ratio, sharp = fp_offset_resample(mix_y, track_y)`; `Pred(set_start_s = max(0, -off), tempo_ratio = ratio, score = float(votes))`. Audio-only (returns `{}` in feature-only synthetic). Mirrors `method_fused`'s audio-loading shape.

- [ ] **Step 1: Write the failing test**

```python
# append to test_eval_bench.py
def test_method_fused_resample_returns_empty_without_audio():
    from alignment.external.eval_bench import method_fused_resample
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 40), np.float32)}, [GTSpan(0, 1.0, 1.0)])
    assert method_fused_resample(s) == {}  # no mix_path -> audio method no-ops
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest alignment/external/tests/test_eval_bench.py::test_method_fused_resample_returns_empty_without_audio -v`
Expected: FAIL with `ImportError: cannot import name 'method_fused_resample'`

- [ ] **Step 3: Implement `method_fused_resample`**

Add to `eval_bench.py` after `method_dtw`:

```python
def method_fused_resample(sample: Sample) -> dict[int, Pred]:
    """Placement via the resample-ratio fp search (for the resample transform:
    pitch+tempo coupled). tempo_ratio = the recovered resample ratio. Audio only."""
    if sample.mix_path is None or not sample.track_paths:
        return {}
    import librosa
    from alignment.landmark_fp import fp_offset_resample

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        my, _ = librosa.load(str(sample.mix_path), sr=SR, mono=True)
    out: dict[int, Pred] = {}
    for k, tp in sample.track_paths.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ty, _ = librosa.load(str(tp), sr=SR, mono=True)
        off, votes, ratio, _sharp = fp_offset_resample(my, ty)
        out[k] = Pred(max(0.0, -off), ratio, float(votes))
    return out
```

Register in `METHODS`:

```python
METHODS: dict[str, Method] = {
    "grid_mf": method_grid_mf,
    "no_warp": method_grid_locked,
    "nmf": method_nmf,
    "fused": method_fused,
    "dtw": method_dtw,
    "fused_resample": method_fused_resample,
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `venvs/audio/bin/python -m pytest alignment/external/tests/test_eval_bench.py -v`
Expected: PASS (all tests).

Smoke: `venvs/audio/bin/python -m alignment.external.eval_bench --synthetic --methods grid_mf,fused_resample`
Expected: table prints (fused_resample shows n=0 in feature-only synthetic — audio method), no traceback.

- [ ] **Step 5: Commit**

```bash
git add alignment/external/eval_bench.py \
        alignment/external/tests/test_eval_bench.py
git commit -m "feat(eval_bench): method_fused_resample (resample-ratio placement)"
```

---

### Task 5: Run the resample arm + record the honest delta

**Files:**
- Modify: `alignment/external/out/reduction_table.txt` (append, gitignored — local only)
- Modify: `alignment/external/unmixdb_findings.md` (append a resample-arm subsection)

**Interfaces:**
- Consumes: `method_fused`, `method_fused_resample`, the `--stratified` CLI (Phase 0).

- [ ] **Step 1: Timing probe**

Run (small, to estimate `fused_resample` per-mix cost — it fingerprints the ref ~30× per candidate):

```bash
venvs/audio/bin/python -m alignment.external.eval_bench \
  --unmixdb-root ~/data/unmixdb-v1.1 --max-mixes 12 --feature chroma \
  --methods fused,fused_resample --stratified
```

Expected: a stratified table; note wall-clock. Extrapolate to pick the sample size for Step 2 (target ≤ ~30 min).

- [ ] **Step 2: Run fused vs fused_resample, stratified**

Use the largest `--max-mixes` feasible in ~30 min from the probe (the shipped Phase-0 headline was 240 → 220 loaded; if `fused_resample` is too slow, drop to 120 or 60). Record the ACTUAL value used.

```bash
venvs/audio/bin/python -m alignment.external.eval_bench \
  --unmixdb-root ~/data/unmixdb-v1.1 --max-mixes <N> --feature chroma \
  --methods fused,fused_resample --stratified \
  2>&1 | tee -a alignment/external/out/reduction_table.txt
```

Expected: stratified table with both methods; the comparison of interest is the three `resample` rows (bass/compressor/distortion/none) — `fused_resample` set_start MAE/med vs `fused`.

- [ ] **Step 3: Append the honest resample-arm subsection to findings**

Add a `### Resample arm (Phase 2)` subsection under the reduction-table section of `unmixdb_findings.md` with the REAL numbers from Step 2: the `resample`-stratum set_start MAE/med for `fused` vs `fused_resample`, the actual sample size N, and an honest read:
- If `fused_resample` improves the resample rows → by how much, and note this narrows the specialist's last advantage (the absorption thesis's open axis).
- If it does not (or regresses on non-resample strata) → say so plainly; the resample-ratio search may add spurious votes on non-resampled content. Record it either way. Do NOT invent numbers — paste from the artifact.

Also note the direction convention `fp_offset_resample` ended up using (from Task 3's report) so the result is reproducible.

- [ ] **Step 4: Commit**

```bash
git add alignment/external/unmixdb_findings.md
git commit -m "docs(findings): resample arm result — fp_offset_resample vs fused on resample stratum"
```

---

## Self-Review

- **Spec coverage:** Phase 2 resample arm (spec §Phase 2) = Tasks 3-5; the spec's "pitch-search grid over landmark_fp, not CQT" decision is realized as the resample-ratio grid (a pitch-search that also fixes tempo, matching the coupled transform). The two Phase-0 carries (DTW score, identity-under-abstention) = Tasks 1-2. Gain/superset/cull/paper remain out of scope (separate plans). ✓
- **Placeholder scan:** every code step shows complete code; Task 3's fixture-latitude and Task 5's sample-size latitude are bounded judgment calls with fixed behavioral assertions / mandatory recording, not placeholders. ✓
- **Type consistency:** `fp_offset_resample` returns `(ref_start_s, votes, ratio, sharpness)` consumed identically in `method_fused_resample`; `Pred(set_start_s, tempo_ratio, score)` used consistently; `method_fused_resample` name matches its `METHODS` registration and its test import. ✓
