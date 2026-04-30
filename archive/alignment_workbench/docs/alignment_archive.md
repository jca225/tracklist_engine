# Archived alignment experiments

**Purpose**: record dropped approaches so future contributors (human or LLM)
don't re-try them without fresh evidence. The canonical SOTA pipeline lives
in [`sota.py`](../audio_pipeline/alignment/sota.py); it imports Viterbi primitives from
[`indicators_debug.py`](../audio_pipeline/alignment/indicators_debug.py). Each entry below was
evaluated against `tests/fixtures/bigbootie11_ground_truth.yaml` with
`mean_mix_IoU` as the scoring metric.

**Current SOTA** (BB11 baseline to beat): **mean IoU = 0.891** (raw 0.872,
snap-via-argmax 0.751).

---

## 1. MACD crossover transition bonuses (Phase 2)

**Idea**: Use bullish MACD-histogram sign-flips as `SILENCE → ref_i`
transition bonuses and bearish flips as `ref_i → SILENCE` bonuses in the
per-universe Viterbi. The intuition was that MACD crossovers align with
cue-in / cue-out events.

**Result**: ~neutral (Δ = +0.013 over raw Phase 1). The Viterbi was already
cue-anchored via emission cue-gating; transition bonuses were redundant or
drowned out by emission gradient. Complexity cost (time-varying transition
tensor) not worth it.

**Verdict**: DROPPED. Do not re-add unless a case is found where emission
gradient is weak at the true cue boundary and the bonus can dominate.

---

## 2. Wilder ADXR/DMI trust gate + entry/exit locks (Phase 2′)

**Idea**: Use Wilder's Average Directional Movement Rating (ADXR) with
the canonical 20/25 trust thresholds to gate emissions, and use sustained
`+DI > -DI AND ADXR > threshold` / `-DI > +DI AND ...` as Wilder's
trend-reversal confirmation to hard-lock entry/exit boundaries.

**Result**:
- Full trust gate on emission wiped out steady-state plays (Bastille
  mid-play IoU collapsed to 0.0) because MERT-sim is flat-high during
  real play, so ADX stays low.
- Entry-lock pushed Bastille entry 25s → 57s because ADXR takes ~14
  measures to build after cue-in.
- Exit-lock was zero-sum within a universe: locking Fray at 99s let CRJ
  claim 101-131s as a phantom.

**Root cause**: Wilder's thresholds were calibrated on daily OHLC price
data with ADX range ~0-100 and multi-day trend timescales. MERT-sim on
DJ-length measures (~2s) produces ADX peaking at ~15-28, never clearing
Wilder's 25 reliably, and confirming trends takes many more measures
than a typical cue-in (2-4 measures).

**Verdict**: DROPPED. ADXR is a trend-regime filter; our problem is a
discrete-event detection problem. Wrong fit. Do not re-add without
fundamentally different calibration for short timescales.

---

## 3. Per-ref BPM matching penalty (Phase 7)

**Idea**: Compute per-ref and per-mix BPM; penalise a ref's emission by
`|log2(mix_bpm / ref_bpm)|` folded onto `[0, 0.5]` so 2× and 0.5× matches
are free. A ref playing at a wildly different tempo from the mix is
unlikely to be the currently-active ref.

**Result**: Gnash IoU collapsed 0.857 → 0.381. Root cause: DJs routinely
tempo-shift refs to beat-match the mix. Gnash's detected ref BPM was 92.3
but the mix played it at 126.3 BPM (37% speed-up) — the ref IS playing
but the penalty punishes it for being at its original tempo. Acapellas
also have unreliable beat-tracking which compounds.

**Verdict**: DROPPED. Matching per-ref BPM to mix BPM is the wrong model
for DJ sets. A potential useful BPM signal is mix-internal discontinuity
(sudden Δ > 3 BPM between consecutive mix measures as a transition event),
but that's a transition prior, not an emission penalty, and not yet tried.

---

## 4. Argmax-based ref-position inference

**Idea**: For each mix measure m, take `argmax_n cos(ref[n], mix[m])` as
the "where in the ref are we currently" signal, used to bracket play
spans by canonical cue points.

**Result**: Argmax is non-monotonic. It picks the most-similar ref frame
independently per mix frame, so chorus repetitions, production elements,
or any spectrally distinctive ref moment can win regardless of actual
playback position. Concrete BB11 failures:
- Bastille at detected-start=26s → argmax ref_t=38s (the drop, not the
  intro) → cue-snap shifted mix-start too far back
- CRJ argmax gave a descending bracket `[117-60s]` (start > end, impossible)
- Gnash similarly gave `[63-41s]`
- Snap-via-argmax mean IoU: **0.751** (vs Viterbi-snap 0.891, raw 0.872)

**Verdict**: DROPPED. Replaced by `ref_position_viterbi()` in
[`indicators_debug.py`](../audio_pipeline/alignment/indicators_debug.py) — a lightweight monotonic
Viterbi over ref-measure states (subsequence-DTW family), imported by
[`sota.py`](../audio_pipeline/alignment/sota.py). Argmax is retained in the output ONLY for
side-by-side comparison to demonstrate the IoU gap.

---

## Rule of thumb for future changes

1. Propose → prototype in `indicators_debug.py` alongside the current SOTA.
2. Evaluate against `tests/fixtures/*_ground_truth.yaml` (run `audio_pipeline/alignment/eval.py`).
3. If the change **does not beat current SOTA on mean IoU across all
   fixtures**, it goes in this archive with:
   - The idea
   - The measured deltas
   - The root cause of failure
   - A verdict
4. If it beats SOTA: update the header docstring in `indicators_debug.py`
   and call it out in [`ROADMAP.md`](ROADMAP.md).
