# André-Absorption Phase 0 — Warp Reduction Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the load-bearing reduction table — our aligner in "André mode" vs the reproduced NMF and DTW baselines, on UnmixDB, on André's warp-error units, stratified by warp×effect, with abstain-rate reported as a first-class capability column.

**Architecture:** Extend the existing `external/eval_bench.py` harness (do not fork it). Add a stratum parser, an abstain-aware scorer with a no-abstain/open toggle, a stratified summary, and a `method_dtw` baseline. Then run the table on the shipped 240-mix stratified sample and commit the artifact + a findings update.

**Tech Stack:** Python 3, numpy, pandas, librosa, `venvs/audio/bin/python`. Run from repo root as a module.

## Global Constraints

- Run everything from repo root with `venvs/audio/bin/python`. Module path: `workspaces.alignment_prototype.external.eval_bench`.
- UnmixDB present locally at `~/data/unmixdb-v1.1` (8 GB).
- `eval_bench --synthetic` (feature-space, no audio) MUST stay green after every task — it is the fast smoke test.
- Sampling for the headline run: seed 0, 240-mix stratified (20 per warp×effect stratum), matching the shipped `external/out/unmixdb_bench.txt`, so numbers are comparable.
- Abstain sentinel: a `Pred` with `set_start_s = float("nan")`. Non-abstain preds always have finite `set_start_s`.
- No C++/Rust. Hot loops are already native (scipy/numpy). Python only.
- Tests live in `workspaces/alignment_prototype/external/tests/` (create the dir + `__init__.py` in Task 1).

---

### Task 1: Stratum parser

**Files:**
- Modify: `workspaces/alignment_prototype/external/eval_bench.py`
- Create: `workspaces/alignment_prototype/external/tests/__init__.py` (empty)
- Test: `workspaces/alignment_prototype/external/tests/test_eval_bench.py`

**Interfaces:**
- Produces: `stratum(mix_id: str) -> tuple[str, str]` returning `(warp, effect)`. UnmixDB mix ids look like `set042mix3-resample-bass-07`; warp ∈ {none,resample,stretch}, effect ∈ {none,bass,compressor,distortion}. Unknown/unparseable → `("unknown", "unknown")`.

- [ ] **Step 1: Write the failing test**

```python
# workspaces/alignment_prototype/external/tests/test_eval_bench.py
from workspaces.alignment_prototype.external.eval_bench import stratum


def test_stratum_parses_warp_and_effect():
    assert stratum("set042mix3-resample-bass-07") == ("resample", "bass")
    assert stratum("set001mix3-none-none-00") == ("none", "none")
    assert stratum("set099mix3-stretch-distortion-19") == ("stretch", "distortion")


def test_stratum_unknown_on_garbage():
    assert stratum("not-a-real-id") == ("unknown", "unknown")
    assert stratum("set042mix3-warpX-effectY-01") == ("unknown", "unknown")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py -v`
Expected: FAIL with `ImportError: cannot import name 'stratum'`

- [ ] **Step 3: Write minimal implementation**

Add to `eval_bench.py`, just below the `Method = Callable[...]` line (after the types block):

```python
_WARPS = {"none", "resample", "stretch"}
_EFFECTS = {"none", "bass", "compressor", "distortion"}


def stratum(mix_id: str) -> tuple[str, str]:
    """(warp, effect) from an UnmixDB mix id `set<NNN>mix3-<warp>-<effect>-<NN>`.
    Unparseable -> ('unknown','unknown')."""
    parts = mix_id.split("-")
    if len(parts) < 4:
        return ("unknown", "unknown")
    warp, effect = parts[1], parts[2]
    if warp not in _WARPS or effect not in _EFFECTS:
        return ("unknown", "unknown")
    return (warp, effect)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/external/eval_bench.py \
        workspaces/alignment_prototype/external/tests/__init__.py \
        workspaces/alignment_prototype/external/tests/test_eval_bench.py
git commit -m "feat(eval_bench): stratum(mix_id) warp×effect parser"
```

---

### Task 2: Abstain-aware scorer + no-abstain / open toggle

**Files:**
- Modify: `workspaces/alignment_prototype/external/eval_bench.py`
- Test: `workspaces/alignment_prototype/external/tests/test_eval_bench.py`

**Interfaces:**
- Consumes: `Sample`, `GTSpan`, `Pred` (existing).
- Produces:
  - `is_abstain(p: Pred) -> bool` — true iff `math.isnan(p.set_start_s)`.
  - `make_fused(min_votes: float = 0.0) -> Method` — factory; the returned method is `method_fused` with an abstention floor. `min_votes == 0.0` = André-mode (always commit). `min_votes > 0` = open-mode (abstain when fp vote count `< min_votes`, emitting `Pred(nan, nan, votes)`).
  - `score_sample` return type unchanged shape `(rows, id_ok)` but each row gains `abstained: bool`; abstained rows carry `set_start_err=nan, tempo_err=nan, tempo_pct=nan`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_eval_bench.py
import math
import numpy as np
from workspaces.alignment_prototype.external.eval_bench import (
    GTSpan, Pred, Sample, is_abstain, make_fused, score_sample,
)


def test_is_abstain():
    assert is_abstain(Pred(float("nan"), float("nan"), 3.0)) is True
    assert is_abstain(Pred(12.0, 1.0, 50.0)) is False


def test_score_sample_marks_abstain_and_excludes_from_error():
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 40), np.float32)},
               [GTSpan(0, 10.0, 1.0)])
    rows, _ = score_sample(s, {0: Pred(float("nan"), float("nan"), 1.0)})
    assert len(rows) == 1
    assert rows[0]["abstained"] is True
    assert math.isnan(rows[0]["set_start_err"])


def test_make_fused_is_a_method_factory():
    m0 = make_fused(0.0)      # André-mode
    m1 = make_fused(100.0)    # open-mode
    assert callable(m0) and callable(m1)
    # feature-only sample -> method_fused returns {} (no audio), both modes
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 40), np.float32)}, [GTSpan(0, 1.0, 1.0)])
    assert m0(s) == {} and m1(s) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_abstain'`

- [ ] **Step 3: Write minimal implementation**

Add `import math` to the top imports of `eval_bench.py`. Add after the `stratum` function:

```python
def is_abstain(p: Pred) -> bool:
    return math.isnan(p.set_start_s)
```

Replace the body of `method_fused` (lines ~112-150) so it takes a floor. Rename the existing function to `_fused_impl` with a `min_votes` argument and add the factory:

```python
def _fused_impl(sample: Sample, min_votes: float) -> dict[int, Pred]:
    if sample.mix_path is None or not sample.track_paths:
        return {}
    import librosa
    from workspaces.alignment_prototype.landmark_fp import fp_offset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        my, _ = librosa.load(str(sample.mix_path), sr=SR, mono=True)
    out = {}
    for k, tp in sample.track_paths.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ty, _ = librosa.load(str(tp), sr=SR, mono=True)
        off, votes, fp_st, _sharp = fp_offset(
            my, ty, stretches=(0.9, 0.95, 1.0, 1.05, 1.1)
        )
        if min_votes > 0.0 and float(votes) < min_votes:
            out[k] = Pred(float("nan"), float("nan"), float(votes))  # abstain
            continue
        fp_start = max(0.0, -off)
        tf = sample.track_feats.get(k)
        if tf is not None and tf.shape[1] >= 8 and sample.mix_feat.shape[1] > tf.shape[1]:
            mf_start, _, mf_tempo = detect_offset(tf, sample.mix_feat)
        else:
            mf_start, mf_tempo = fp_start, fp_st
        set_start = mf_start if abs(mf_start - fp_start) <= 8.0 else fp_start
        out[k] = Pred(set_start, mf_tempo, float(votes))
    return out


def make_fused(min_votes: float = 0.0) -> Method:
    """André-mode (min_votes=0, always commit) or open-mode (min_votes>0, abstain
    on weak fp votes)."""
    def method(sample: Sample) -> dict[int, Pred]:
        return _fused_impl(sample, min_votes)
    return method


def method_fused(sample: Sample) -> dict[int, Pred]:
    return _fused_impl(sample, 0.0)
```

In `score_sample`, replace the row-append block (inside the `for sp in sample.gt` loop) with abstain-aware logic:

```python
    for sp in sample.gt:
        p = preds.get(sp.track_idx)
        if p is None:
            continue
        if is_abstain(p):
            rows.append(dict(
                mix_id=sample.mix_id, track=sp.track_idx,
                set_start_err=float("nan"), tempo_err=float("nan"),
                tempo_pct=float("nan"), peak=p.score, abstained=True,
            ))
            continue
        rows.append(dict(
            mix_id=sample.mix_id, track=sp.track_idx,
            set_start_err=abs(p.set_start_s - sp.set_start_s),
            tempo_err=abs(p.tempo_ratio - sp.tempo_ratio),
            tempo_pct=abs(p.tempo_ratio - sp.tempo_ratio) / max(1e-6, sp.tempo_ratio) * 100.0,
            peak=p.score, abstained=False,
        ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py -v`
Expected: PASS (5 tests)

Also verify the synthetic smoke still works:
Run: `venvs/audio/bin/python -m workspaces.alignment_prototype.external.eval_bench --synthetic`
Expected: prints the placement/warp table with no traceback.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/external/eval_bench.py \
        workspaces/alignment_prototype/external/tests/test_eval_bench.py
git commit -m "feat(eval_bench): abstain-aware scorer + make_fused(min_votes) mode toggle"
```

---

### Task 3: Stratified summary with abstain-rate column

**Files:**
- Modify: `workspaces/alignment_prototype/external/eval_bench.py`
- Test: `workspaces/alignment_prototype/external/tests/test_eval_bench.py`

**Interfaces:**
- Consumes: `stratum`, per-row `abstained` field.
- Produces: `summary_by_stratum(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame` — one row per (method, warp, effect) plus an ALL row per method. Columns: `method, warp, effect, n, abstain_pct, set_start_MAE_s, set_start_med_s, tempo_MAE`. Error stats computed over **committed (non-abstained) rows only**; `abstain_pct = 100 * mean(abstained)` over all rows in the group.
- Also: extend the existing `summary` to add an `abstain_pct` column (over-all, non-stratified) so the flat table stays consistent. Error means use `df[~df.abstained]`.

- [ ] **Step 1: Write the failing test**

```python
# append to test_eval_bench.py
import pandas as pd
from workspaces.alignment_prototype.external.eval_bench import summary_by_stratum


def test_summary_by_stratum_groups_and_reports_abstain():
    df = pd.DataFrame([
        dict(mix_id="set1mix3-none-none-00", track=0, set_start_err=1.0,
             tempo_err=0.01, tempo_pct=1.0, peak=50, abstained=False),
        dict(mix_id="set1mix3-none-none-00", track=1, set_start_err=float("nan"),
             tempo_err=float("nan"), tempo_pct=float("nan"), peak=2, abstained=True),
        dict(mix_id="set2mix3-resample-bass-01", track=0, set_start_err=3.0,
             tempo_err=0.02, tempo_pct=2.0, peak=40, abstained=False),
    ])
    df.attrs["identity_acc"] = float("nan")
    out = summary_by_stratum({"fused": df})
    none = out[(out.method == "fused") & (out.warp == "none") & (out.effect == "none")].iloc[0]
    assert none.n == 2
    assert none.abstain_pct == 50.0
    assert none.set_start_MAE_s == 1.0  # committed-only mean
    allrow = out[(out.method == "fused") & (out.warp == "ALL")].iloc[0]
    assert allrow.n == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py::test_summary_by_stratum_groups_and_reports_abstain -v`
Expected: FAIL with `ImportError: cannot import name 'summary_by_stratum'`

- [ ] **Step 3: Write minimal implementation**

Add to `eval_bench.py` after the existing `summary`:

```python
def _grp_row(method: str, warp: str, effect: str, g: pd.DataFrame) -> dict:
    committed = g[~g.abstained]
    return dict(
        method=method, warp=warp, effect=effect, n=len(g),
        abstain_pct=round(100 * g.abstained.mean(), 1),
        set_start_MAE_s=round(committed.set_start_err.mean(), 2) if len(committed) else float("nan"),
        set_start_med_s=round(committed.set_start_err.median(), 2) if len(committed) else float("nan"),
        tempo_MAE=round(committed.tempo_err.mean(), 4) if len(committed) else float("nan"),
    )


def summary_by_stratum(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for method, df in dfs.items():
        if len(df) == 0:
            out.append(dict(method=method, warp="ALL", effect="ALL", n=0))
            continue
        strat = df.mix_id.map(stratum)
        df = df.assign(warp=[s[0] for s in strat], effect=[s[1] for s in strat])
        for (w, e), g in df.groupby(["warp", "effect"], sort=True):
            out.append(_grp_row(method, w, e, g))
        out.append(_grp_row(method, "ALL", "ALL", df))
    return pd.DataFrame(out)
```

Also update the flat `summary` — add one line inside its per-method dict, and switch error means to committed-only. Change the `out.append(dict(...))` in `summary` to:

```python
        committed = df[~df.abstained] if "abstained" in df else df
        out.append(
            dict(
                method=label,
                n=len(df),
                abstain_pct=round(100 * df.abstained.mean(), 1) if "abstained" in df else 0.0,
                set_start_MAE_s=round(committed.set_start_err.mean(), 2),
                set_start_med_s=round(committed.set_start_err.median(), 2),
                set_start_exact2s_pct=round(100 * (committed.set_start_err < 2).mean(), 0),
                tempo_MAE=round(committed.tempo_err.mean(), 4),
                tempo_pct=round(committed.tempo_pct.median(), 2),
                identity_acc=round(df.attrs.get("identity_acc", float("nan")), 3),
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py -v`
Expected: PASS (6 tests)

Run: `venvs/audio/bin/python -m workspaces.alignment_prototype.external.eval_bench --synthetic`
Expected: table prints, now with an `abstain_pct` column, no traceback.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/external/eval_bench.py \
        workspaces/alignment_prototype/external/tests/test_eval_bench.py
git commit -m "feat(eval_bench): summary_by_stratum + abstain_pct column"
```

---

### Task 4: DTW baseline method

**Files:**
- Modify: `workspaces/alignment_prototype/external/eval_bench.py`
- Test: `workspaces/alignment_prototype/external/tests/test_eval_bench.py`

**Interfaces:**
- Consumes: `Sample`, `Pred`, `librosa.sequence.dtw`, existing `detect_offset` for the tempo estimate.
- Produces: `method_dtw(sample: Sample) -> dict[int, Pred]` registered in `METHODS` as `"dtw"`. Runs in **feature space** (chroma features already on the `Sample`), so it works on both synthetic and UnmixDB samples with no extra audio load. For each candidate: DTW-align the track feature sequence against the mix feature sequence, read `set_start_s` from the first mix frame of the warping path, and `tempo_ratio` from the path's average mix-frames-per-track-frame slope. Abstain (`Pred(nan,nan,-1)`) if the track is too short (`< 8` frames) or longer than the mix.

- [ ] **Step 1: Write the failing test**

```python
# append to test_eval_bench.py
from workspaces.alignment_prototype.external.eval_bench import method_dtw, HOP, SR


def test_method_dtw_recovers_planted_offset():
    rng = np.random.default_rng(0)
    D, tlen = 12, 120
    tf = rng.random((D, tlen)).astype(np.float32)
    Tm = 800
    mix = (rng.random((D, Tm)).astype(np.float32) * 0.05)
    start_f = 300
    mix[:, start_f:start_f + tlen] += tf  # planted, tempo 1.0, no stretch
    s = Sample("synthDTW", mix, {0: tf}, [GTSpan(0, start_f * HOP / SR, 1.0)])
    preds = method_dtw(s)
    assert 0 in preds
    # within ~1s of the planted start
    assert abs(preds[0].set_start_s - start_f * HOP / SR) < 1.0
    assert abs(preds[0].tempo_ratio - 1.0) < 0.15


def test_method_dtw_abstains_on_tiny_track():
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 4), np.float32)}, [GTSpan(0, 1.0, 1.0)])
    preds = method_dtw(s)
    assert math.isnan(preds[0].set_start_s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py::test_method_dtw_recovers_planted_offset -v`
Expected: FAIL with `ImportError: cannot import name 'method_dtw'`

- [ ] **Step 3: Write minimal implementation**

Add to `eval_bench.py` after `method_fused`:

```python
def method_dtw(sample: Sample) -> dict[int, Pred]:
    """André's DTW baseline, feature space: subsequence-DTW each candidate track
    against the mix; set_start = mix time at path start, tempo = path slope
    (mix-frames per track-frame). Works on synthetic + UnmixDB."""
    import librosa

    out: dict[int, Pred] = {}
    M = sample.mix_feat
    for idx, tf in sample.track_feats.items():
        if tf.shape[1] < 8 or M.shape[1] <= tf.shape[1]:
            out[idx] = Pred(float("nan"), float("nan"), -1.0)
            continue
        # cost = 1 - cosine similarity between track frames (rows) and mix frames (cols)
        tfn = tf / (np.linalg.norm(tf, axis=0, keepdims=True) + 1e-8)
        Mn = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-8)
        C = 1.0 - (tfn.T @ Mn)  # (Tk, Tm)
        _, wp = librosa.sequence.dtw(C=C, subseq=True, backtrack=True)
        wp = wp[::-1]  # ascending
        track_f = wp[:, 0]
        mix_f = wp[:, 1]
        set_start = float(mix_f[0]) * HOP / SR
        span_track = max(1, track_f[-1] - track_f[0])
        span_mix = mix_f[-1] - mix_f[0]
        tempo = float(span_mix) / float(span_track)  # mix frames per track frame
        out[idx] = Pred(max(0.0, set_start), tempo, 1.0 - float(C.min()))
    return out
```

Register it in `METHODS`:

```python
METHODS: dict[str, Method] = {
    "grid_mf": method_grid_mf,
    "no_warp": method_grid_locked,
    "nmf": method_nmf,
    "fused": method_fused,
    "dtw": method_dtw,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/external/tests/test_eval_bench.py -v`
Expected: PASS (8 tests)

Run: `venvs/audio/bin/python -m workspaces.alignment_prototype.external.eval_bench --synthetic --methods grid_mf,dtw`
Expected: table with both methods, no traceback.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/external/eval_bench.py \
        workspaces/alignment_prototype/external/tests/test_eval_bench.py
git commit -m "feat(eval_bench): method_dtw (André DTW baseline, feature space)"
```

---

### Task 5: Wire the reduction CLI + run the headline table

**Files:**
- Modify: `workspaces/alignment_prototype/external/eval_bench.py` (CLI)
- Create: `workspaces/alignment_prototype/external/out/reduction_table.txt` (artifact)
- Modify: `workspaces/alignment_prototype/external/unmixdb_findings.md` (append reduction section)

**Interfaces:**
- Consumes: everything above.
- Produces: CLI flags `--stratified` (print `summary_by_stratum`) and `--min-votes FLOAT` (open-mode floor for the `fused` method, default 0.0 = André-mode). When `--min-votes > 0`, the `fused` entry in the method loop is built with `make_fused(min_votes)`.

- [ ] **Step 1: Add the CLI flags and stratified output**

In `main`, add after the `--identity` argument:

```python
    p.add_argument("--stratified", action="store_true",
                   help="print the warp×effect stratified table")
    p.add_argument("--min-votes", type=float, default=0.0,
                   help="open-mode fp abstain floor for 'fused' (0 = André-mode)")
```

Change the method-loop so `fused` honors `--min-votes`:

```python
    dfs = {}
    for name in args.methods.split(","):
        name = name.strip()
        if name == "fused" and args.min_votes > 0.0:
            dfs[name] = run(samples, make_fused(args.min_votes), name)
            continue
        if name not in METHODS:
            print(f"unknown method {name}")
            continue
        dfs[name] = run(samples, METHODS[name], name)
```

Add after the flat-summary print:

```python
    if args.stratified:
        print("\n=== stratified (warp × effect) ===")
        print(summary_by_stratum(dfs).to_string(index=False))
```

- [ ] **Step 2: Smoke-test the new flags on synthetic**

Run: `venvs/audio/bin/python -m workspaces.alignment_prototype.external.eval_bench --synthetic --methods grid_mf,dtw --stratified`
Expected: flat table AND a stratified table print (synthetic ids are `synth0…` → stratum `unknown/unknown`), no traceback.

- [ ] **Step 3: Run the headline reduction table on UnmixDB**

Run (this is the real run — audio, ~240 mixes; expect minutes-to-tens-of-minutes):

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.external.eval_bench \
  --unmixdb-root ~/data/unmixdb-v1.1 --max-mixes 240 --feature chroma \
  --methods nmf,dtw,fused --stratified --identity \
  | tee workspaces/alignment_prototype/external/out/reduction_table.txt
```

Expected: an artifact file with the flat table (nmf/dtw/fused, incl. `abstain_pct`), the warp×effect stratified table, and the identity block. No traceback; non-zero rows for each method.

- [ ] **Step 4: Run André-mode vs open-mode contrast for `fused`**

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.external.eval_bench \
  --unmixdb-root ~/data/unmixdb-v1.1 --max-mixes 240 --feature chroma \
  --methods fused --min-votes 20 --stratified \
  | tee -a workspaces/alignment_prototype/external/out/reduction_table.txt
```

Expected: the `fused` row now shows `abstain_pct > 0` and (hypothesis) tighter committed MAE than André-mode — the abstention-as-capability contrast.

- [ ] **Step 5: Append the honest reduction section to findings**

Add a new `## Reduction table (André mode)` section to `external/unmixdb_findings.md` summarizing: (a) the flat nmf-vs-dtw-vs-fused table with real numbers pasted from `reduction_table.txt`; (b) the warp×effect breakdown; (c) the André-mode vs open-mode abstain contrast; (d) the honest read — where fused's committed warp error beats the NMF/DTW baselines and where (resample) it abstains/loses. Do not invent numbers; paste from the artifact.

- [ ] **Step 6: Commit**

```bash
git add workspaces/alignment_prototype/external/eval_bench.py \
        workspaces/alignment_prototype/external/out/reduction_table.txt \
        workspaces/alignment_prototype/external/unmixdb_findings.md
git commit -m "feat(eval_bench): reduction table run — nmf/dtw/fused stratified, André vs open mode"
```

---

## What Phase 0 deliberately does NOT cover (follow-on plans)

- **Gain MAE** — needs `UnmixTrackSpan.gain_envelope` + `NmfPred` curve; NMF-only, noisy; separate plan.
- **Phase 1 superset** — `path_decode` non-affine trajectory scoring (needs BB non-affine GT; UnmixDB is affine-only) + open-set identity write-up.
- **Phase 2 resample arm** — pitch-search grid over `landmark_fp`; its own plan once Phase 0 quantifies the resample gap.
- **Phase 3 cull** — move confirmed orphans (`infer_fused`, `enhance_vocal`, `fp_probe`) to `attic/`.
- **Phase 4 paper** — findings report around the measured reduction.

## Self-Review

- **Spec coverage:** Phase 0 spine (abstain toggle, stratification, DTW, reduction run) all have tasks. Gain explicitly deferred with rationale. Superset/resample/cull/paper explicitly out of scope with follow-on note. ✓
- **Placeholder scan:** every code step shows complete code; the only "summarize real numbers" step (Task 5 Step 5) is a data-entry step that forbids inventing numbers, not a code placeholder. ✓
- **Type consistency:** `Pred(set_start_s, tempo_ratio, score)` used consistently; abstain sentinel `set_start_s=nan` consistent across `is_abstain`, `_fused_impl`, `method_dtw`, `score_sample`. `make_fused`/`method_fused`/`method_dtw` names match `METHODS` registration. ✓
