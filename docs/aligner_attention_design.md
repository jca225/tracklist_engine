# Mix structure analysis — sections, events, and MERT probes (IN PROGRESS)

**Status:** data-analysis phase — **not** the aligner, **not** a committed model
architecture. Created 2026-06-07; reframed 2026-06-08.

**Sibling docs:** [alignment_objective.md](alignment_objective.md) (aligner north star),
[alignment_program_plan.md](alignment_program_plan.md) (GT export + aligner execution),
[bb12_information_dynamics_plan.md](bb12_information_dynamics_plan.md) (BB12 lab-set
checklist). **Code home:** [eda/alignment/](../eda/alignment/).

## What this phase is

Exploratory analysis: use mix-side MERT (and cheap probes on top) to ask whether DJ
mixes decompose into **stable sections** separated by **detectable structural events**.

We are **not** building Deliverable B yet. We are characterizing the data so later
aligner design is informed by evidence, not architecture fashion.

**Deliverables:** plots, boundary scores against manual GT where available, an event
taxonomy draft, and a short findings write-up (`eda/alignment/findings.md`). **Not**
`measure_alignment` rows, not a training pipeline, not Ableton round-trip.

## Vocabulary

| Term | Meaning in this phase | Later (aligner) |
|------|----------------------|-----------------|
| **Atom (section)** | A stable span where mix MERT is well explained by a fixed multi-layer source set | `set_ground_truth` span row(s) |
| **Event** | A detected change at a boundary — hypothesis until GT confirms | Supervised edit type if we label one |
| **Layer** | One active source in a section: `(recording, claimed_stem, ref segment, warp, key)` | Same — one GT row |
| **Attention / retrieval** | Diagnostic only: “which catalog refs explain this segment?” | Maybe part of the model; TBD |

### Event types (exploratory taxonomy)

Hypothesis classes to cluster boundaries into — not a fixed enum yet:

- **track_change** — full handoff to a new primary source  
- **layer_add** — acappella (or vocal stem) appears over an existing bed  
- **layer_swap** — same role, different ref (instrumental swap, better acappella)  
- **new_mashup** — multi-layer state appears (2+ simultaneous explainers)  
- **layer_drop** — a layer disappears; bed continues  
- **intra_section_edit** — warp/key/FX shift without identity change (may be continuous, not a hard event)

BB12 is the **first lab set** with dense GT (~167 span rows). Questions are **corpus-wide**;
BB12 is a scoreboard, not the problem definition.

## Pipeline (this phase)

```mermaid
flowchart LR
  A[Phase 0: mix beat grid + per-bar MERT] --> B[Tokenize bars VQ / side-stream]
  B --> C[Adaptive Markov info-dynamics]
  C --> D[Boundary peaks = event candidates]
  D --> E[Segment retrieval vs set tracklist]
  E --> F[Score vs GT + findings]
```

| Phase | Work | Output |
|-------|------|--------|
| **0** | `beat_this` on mix → `set_measures`; MERT-330M per-bar pool → local `.npz` | `data/analysis/<set_id>_mix_mert.npz` |
| **1** | VQ/k-means tokens over mix MERT bars (N≈16); optional chroma side-stream | symbol sequence(s) |
| **2** | `AdaptiveMarkovChain` — surprisingness, model-info-rate, PIR | per-bar traces |
| **3** | Peak-pick boundaries; overlay GT `set_start_s`; precision/recall ±N bars | boundary scores |
| **4** | Per-segment retrieval against set tracklist MERT; count layers (1 vs 2+) | mashup hypotheses |
| **5** | Write-up | `eda/alignment/findings.md` |

⚠ Tokens must come from **mix audio only** — never from GT timestamps (circular).

### Run (once Phase 0 artifact exists)

```bash
# Synthetic sanity check (no audio / DB)
venvs/audio/bin/python -m eda.alignment.mix_structure_probe --synthetic

# BB12 lab set
venvs/audio/bin/python -m eda.alignment.mix_structure_probe \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --gt labeling/fixtures/bb12_ground_truth.yaml
```

Phase 0 prep (mix audio on Mac or pi-storage): see [eda/alignment/README.md](../eda/alignment/README.md).

## Non-goals

- Training the production aligner or writing `measure_alignment`  
- BB12-only conclusions without checking at least one other set  
- Generative “language model of DJing” (downstream codebase)  
- Synthetic mashup **pretraining** (aligner-scale; defer until parsing probe succeeds)  
- Per-bar song softmax over the full ~20k catalog  

## Success criteria (analysis)

- Model-info-rate / surprise peaks align with GT section starts at useful precision/recall (±1–2 bars TBD).  
- Mashup blocks show **multi-ref retrieval** within one segment more often than single-track blocks.  
- We can name **failure modes** (similar back-to-back tracks, long crossfades, subtle stem swaps).  
- Findings change or confirm aligner priorities (segment-then-label vs dense grid).

## Future aligner hypotheses (appendix — not commitments)

If the probe shows section+event structure is real in embedding space, a later aligner
*may* use:

- **Retrieval** over catalog MERT for identity (not a song softmax)  
- **Set prediction** per section for mashup layers (not next-token LM)  
- **Two-clock PE** on the beat grid (mix time + per-source ref phase)  
- **Synthetic pretrain → GT fine-tune** for data hunger  

Those belong in [alignment_program_plan.md](alignment_program_plan.md) P5+ once this
analysis phase reports go/no-go. Attention over mix→catalog is a plausible inductive bias;
this doc does **not** decide the final architecture.

## Prerequisites

- **Re-sourcing** bad catalog audio (2026-05-30 review) — retrieval diagnostics are
  only meaningful against clean ref MERT.  
- **Mix-side MERT** for lab sets — ref-side 330M backfill exists; mix-side does not yet
  for BB12 (`set_measures`=0 as of 2026-06-05).
