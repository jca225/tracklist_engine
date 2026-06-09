# Mix structure analysis — findings

## BB12 (`1fsnxchk`) — information-dynamics probe (2026-06-08)

**Setup:** 2054 bar-sync MERT-330M vectors (layer 6) on local `mix.m4a`; VQ
k=24 tokens; adaptive first-order Markov chain; scored against
`labeling/fixtures/bb12_ground_truth.yaml` (154 unique section-start boundaries,
±2 bars).

**Reproduce:**
```bash
venvs/audio/bin/python -m eda.alignment.mix_structure_probe \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out data/analysis/1fsnxchk_structure_probe.json --persist
```

### Boundary detection (model-information-rate peaks)

| Metric | Value |
|--------|-------|
| Precision | 0.38 |
| Recall | 0.53 |
| F1 | **0.45** |
| Predicted peaks | 214 |
| GT boundaries | 154 |

The cheap Markov-over-VQ probe finds **about half** of human-labeled section
starts without seeing GT. Useful as a **boundary proposer**, not as the aligner.

### Is the mix "information-optimal"?

Using Abdallah & Plumbley (2008) readouts:

**Predictive information rate (PIR — inverted-U "interestingness"):**
~**78%** of bars sit in the mid-|PIR| band (between 0.25× and 2.5× the median).
By the paper's Wundt curve, the mix spends most of its runtime at *intermediate*
predictive information — not in low-information grooves nor only at shock spikes.
That is consistent with a deliberately varied mashup set, though we have no
corpus baseline yet to call it "optimal" vs other DJ sets.

**Bayesian surprise (model-information-rate) at GT boundaries:**
- GT section starts sit at the **56th** percentile of MIR vs **41st** for
  interior bars (+15 percentile-point lift).
- **12.3%** of GT boundaries fall in the top decile of MIR vs **~10%** random
  baseline — modest enrichment (~1.2×), not a smoking gun.

**PIR at GT boundaries:** no lift (49th percentile) — transitions are better
tagged by *surprise / MIR* than by PIR rate alone for this tokenization.

### Notable surprises (probe-only, not in GT)

Highest MIR clusters **late in the set** (~49–58 min): bars 1652–1941. Worth
listening at **~48:38, 49:58, 51:28** — may be outro chaos, double-drops, or
boundaries the hand pass did not mark as new section starts.

Early mix: bar 1 dominates (cold-start of empty Markov state) — ignore for
interpretation.

### Takeaways for aligner design

1. **Segment-then-label is validated** — bar-wise MIR carries section-start signal;
   dense per-bar song classification is unnecessary at first pass.
2. **Tokenization matters** — VQ k=24 + 75th-percentile peaks beat defaults; chroma
   side-stream still worth adding (plan Phase 1b).
3. **Not "solved"** — F1 0.45 and 1.2× top-decile enrichment mean the probe
   characterizes structure but does not replace manual GT.
4. **One bad MERT bar** (index 36, inf overflow) — now sanitized; root cause in
   float16 accumulate should be monitored on re-embed.

### Artifacts

| File | Description |
|------|-------------|
| `data/analysis/1fsnxchk_measure_times.json` | beat_this downbeats |
| `data/analysis/1fsnxchk_mix_mert.npz` | per-bar layer-6 MERT |
| `data/analysis/1fsnxchk_structure_probe.json` | v1 probe output |
| `data/analysis/1fsnxchk_structure_probe_v2.json` | **v2 dual-stream + local peaks** |
| `data/analysis/2nvzlh2k_*` | BB11 baseline (no GT) |
| `data/analysis/aux.db` | `analysis_results` rows |

---

## v2 — dual-stream (MERT-VQ + chroma) + local MIR peaks (2026-06-08)

**Motivation:** test whether chroma side-stream and locally-normalized surprise
peaks improve boundary detection and whether BB12 is uniquely information-dense
vs BB11.

### BB12 boundary F1 (±2 bars, 154 GT starts)

| Stream | Peak mode | F1 | Notes |
|--------|-----------|-----|-------|
| MERT-VQ | global | **0.446** | still best |
| MERT-VQ | local (32-bar z) | 0.391 | local norm did not help |
| Chroma-VQ | global | 0.390 | worse; GT lift **−6.6** (anti-aligned) |
| Chroma-VQ | local | 0.361 | |
| Combined | local union | 0.407 | more peaks, lower precision |

**Takeaway:** pitch/chroma tokens do not track mashup *section starts* — MERT
(shift-agnostic) is the right stream. Local MIR did not beat global thresholding
on BB12; late-set clustering was not the main bottleneck.

### Corpus baseline — mid-|PIR| band fraction (“interestingness”)

| Set | Bars | MERT mid-PIR | Chroma mid-PIR |
|-----|------|--------------|----------------|
| BB12 `1fsnxchk` | 2054 | **0.740** | 0.674 |
| BB11 `2nvzlh2k` | 1919 | 0.709 | 0.715 |

BB12 is slightly higher on MERT-PIR but **not dramatically** more
“information-optimal” than BB11 — both sit in the ~70–74% mid-band regime.
Cannot claim BB12 is special without more sets and event-level GT.

**Reproduce v2:**
```bash
venvs/audio/bin/python -m eda.alignment.mix_structure_probe \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --audio ~/aligning/1fsnxchk__Two\ Friends\ -\ Big\ Bootie\ Mix\ Volume\ 12/mix.m4a \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out data/analysis/1fsnxchk_structure_probe_v2.json
```
