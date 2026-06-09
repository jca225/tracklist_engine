# BB12 Information-Dynamics Probe — lab-set checklist

**Status:** execution moved to [aligner_attention_design.md](aligner_attention_design.md)
(mix structure **analysis phase**) + [eda/alignment/](../eda/alignment/). This file
keeps BB12-specific data readiness only.

**Owner intent:** "Run the model on BB12 and see anything interesting. In a perfect
world we capture everything in the past (the manual labels) with this model."

## Source idea

Abdallah & Plumbley, *Information Dynamics: Patterns of expectation and surprise in
the perception of music* (Connection Science 2008). An **observer** maintains an
**adaptive first-order Markov chain** over a discrete symbol sequence and, per event,
traces five time-varying information measures. Applied to two Philip Glass pieces, the
measures recover the score's sectional boundaries and an expert's "most surprising
moments."

### The five measures and what they become for a DJ mix

| Paper quantity | Formula (Markov) | In BB12 | Spike means |
|---|---|---|---|
| Model-information-rate | KL(Dirichlet posterior ‖ prior) on params (Bayesian surprise) | paper's **best** boundary detector | **track transition / new track loaded** |
| Surprisingness `L(x\|z)` | `−log a_ij` | locally unexpected bar | drop, sudden vocal/acappella, cut |
| Predictive uncertainty `H(X\|Z=z)` | entropy of predictive column | tension/ambiguity in a section | buildup vs groove |
| Predictive information `I(i\|j)` | `Σ_k a_ki log(a_ki/[a²]_kj)` (eq. A11) | info the bar carries about the future | onset/foreground bars |
| Predictive information **rate** | `Ḣ(a²) − Ḣ(a)` | inverted-U / Wundt "interestingness" | mix sits at *intermediate* PIR |

**Core bet:** a track transition is by definition a change in generative statistics →
exactly what model-information-rate catches. The hand labels give a quantitative
scoreboard.

## BB12 data readiness (verified on canonical pi-storage DB, 2026-06-05)

`set_id = 1fsnxchk` ("Two Friends - Big Bootie Mix Volume 12")

- ✅ 165 track slots, 161 distinct recordings, 155 with downloaded audio
- ✅ **167 ground-truth rows, all timestamped** (`set_ground_truth.set_start_s/_end_s`,
  `is_loop`, overlapping mashup slots) — the validation gold (being polished now)
- ✅ 236 reference `track_audio` rows have all-layer MERT measure embeddings
- ✅ 1 `set_audio` row (mix file exists)
- ❌ **mix has no beat grid** (`set_measures`=0, `set_analysis`=0) and no MERT yet

## Decisions (locked)

- **Features:** MERT-330M on the **mix** (rich), per-bar pooled.
- **Tokens — both, compared:**
  - (a) **learned VQ codebook** over mix MERT bars, N≈16 (primary, data-driven).
  - (b) **interpretable** chroma chord/key + energy token (cheap CPU side-stream;
    every surprise spike human-readable). *Open option:* instead use labeled MERT
    clusters for the interpretable stream.
- **Unit:** bar-synchronous (~1,800 bars over ~70 min).

## Pipeline

Implemented in `eda/alignment/` — see [aligner_attention_design.md](aligner_attention_design.md)
and [eda/alignment/README.md](../eda/alignment/README.md).

**Phase 0 (BB12 blocker):** beat-grid + MERT the mix → `data/analysis/1fsnxchk_mix_mert.npz`.

**Phase 1+:** `venvs/audio/bin/python -m eda.alignment.mix_structure_probe` with that artifact
and `labeling/fixtures/bb12_ground_truth.yaml`.

Chroma side-stream (interpretable tokens) still TODO in probe CLI.

## Caveats / known limits

- **First-order Markov "can't count"** (paper §5.2.1): will nail transitions and drops,
  miss phrase-length games (e.g. an 8-bar loop extending to 16). Natural upgrade: HMM /
  variable-order / Dirichlet-process alphabet (paper §7).
- DJ pitch-shift rotates chroma but is constant within a track; boundary detection cares
  about *changes*, so it's robust. (MERT stream is shift-agnostic anyway.)

## First executable step

Phase 0: beat-grid the BB12 mix and run `eda.alignment.prepare_mix_artifact`. Synthetic
self-test works today: `python -m eda.alignment.mix_structure_probe --synthetic`.
