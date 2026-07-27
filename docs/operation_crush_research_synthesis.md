# Operation Crush — research synthesis

> Living document: links papers found in iCloud Drive and ACL 2026 / ISMIR 2025 to the Tracklist Engine alignment/mashup pipeline. Update this when a paper is folded in or rejected.

## Objective

Operation Crush is the data-integrity + model-infrastructure project for the August 1 alignment north star. The field has shifted from single global embeddings (CLAP/cosine) to **language-centered, fine-grained, task-specific representations**. This doc maps that shift to concrete code changes.

---

## Papers already aligned with current work

### MERT — *MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training* (ICLR 2024)
- Source: `Music Papers 2026/MERT.pdf`
- Already the backbone of the identity channel (`analysis/adapters/mert_adapter.py`, `alignment/mert_*`).
- **Actionable:** upgrade to MERT 330M with a **learned weighted sum over all hidden states** instead of the single layer-6 pick. The paper shows layers 4–7 encode acoustic/tempo, 8–13 pitch/harmony, 14–19 timbre/instrumentation, 20–24 semantics.
- **Actionable:** use **frame-level MERT features** for precise ref-offset localization, not only pooled embeddings.

### HuBERT — *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* (2021)
- Source: `HuBERT.pdf`
- Basis for the acappella/vocal channel and the MERT teacher design.
- **Actionable:** use iterative k-means teacher pseudo-labeling as the recipe for the pseudo-GT flywheel: generate labels with the current best model, retrain, repeat.

### HT Demucs — *Hybrid Transformers for Music Source Separation* (ICASSP 2023)
- Source: `Music Analysis/Archive/Hybrid Transformers for Music Source Separation.pdf`
- 4-stem separation lineage. RoFormer is the current preferred backend, but HT Demucs remains the reliable fallback.
- **Actionable:** record the **stem backend** (roformer / demucs / uvr) in `track_stems` / `gt_clip_provenance` so the pipeline can flag known backend failure modes.

---

## Papers to fold in immediately for Operation Crush

### 1. DJtransGAN — *Automatic DJ Transitions with Differentiable Audio Effects and GANs* (ICASSP 2022)
- Source: `Music Analysis/Archive/Automatic DJ Transitions with Differentiable Audio Effects and GANs.pdf`
- Models transitions as **parameterized EQ + fader curves** applied to overlapping tracks, with explicit cue-out / cue-in points.
- **Why it is the highest-leverage paper:** the `GroundTruthTrack` schema already captures `gain_curve`, `audible_start_s`, `audible_end_s`, and `audible_frac`. DJtransGAN provides the physical model that generates those values.
- **Code changes:**
  - Add `alignment/transition_model.py` that, given two consecutive spans and their audio, fits a differentiable fade/EQ curve to the overlap.
  - Use the fitted curve to estimate `audible_start_s` / `audible_end_s` and score transition realism.
  - Use the paper's cue-out/cue-in definition (last/first diagonal DTW segment) to refine GT boundaries.
- **Issue:** DJtransGAN transition model task under Operation Crush.

### 2. FIGMA — *Towards FIne-Grained Music retrievAl* (ACL 2026)
- Source: ACL Anthology 2026.acl-long.2197; also referenced in `docs/research_scan_2026_07_11.md`.
- Replaces global CLAP embeddings with **token-level and frame-level audio-text alignment**.
- **Code changes:**
  - Add an audio→text verification channel to the acquisition gate (`scripts/acquire_variant.py`, `analysis/identity_learned.py`).
  - Verify that downloaded audio supports the tracklist's claimed version/artist/remixer text, localized in time.
  - Use frame-token alignment to flag version/variant mismatches that global cosine misses.
- **Issue:** fold into #41 (agentic acquisition gate).

### 3. MULTI-SCORE — *Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations* (ACL 2026)
- Source: ACL Anthology 2026.acl-long.930.
- Two-stage retrieval: cheap filtering → expensive reasoning-based reranking.
- **Code changes:**
  - For candidate acquisition: fingerprint/metadata search (stage 1) → MERT/HuBERT/FIGMA verification (stage 2).
  - For transition search: chroma/fingerprint candidate offsets (stage 1) → learned scorer + local alignment (stage 2).
- **Issue:** fold into #41.

### 4. Amazon Music — *Why One Size Doesn't Fit All: Improving Music Discovery and Familiar Listening with Specialized Models* (WSDM 2025)
- Source: DOI 10.1145/3779211.3793165.
- Use **separate embedding spaces for separate tasks** instead of one universal latent space.
- **Code changes:** formalize the split already implicit in the three-axis design:
  - identity embedding (MERT)
  - placement embedding (chroma/fingerprint over time)
  - transition embedding (DJtransGAN fade/EQ realism)
  - mashup compatibility embedding (FIGMA + human preference)
- **Issue:** fold into #40 (taxonomy/identity) and #41 (agentic gate).

---

## Papers for the agentic / reasoning layer

### 5. Audio reasoning survey — *A Survey of Audio Reasoning in Multimodal Foundation Models* (2026)
- Source: `Music Analysis/2605.21008v1.pdf`.
- Advocates **chain-of-thought / intermediate reasoning** over audio, with explicit verification.
- **Code changes:**
  - The agentic aligner should emit a **reasoning trace** per placement: which channel (fp, HuBERT, chroma) supported the decision, and why.
  - Use the trace in the human review queue and for training-data curation.
- **Issue:** fold into #41.

### 6. Trustworthy audio LALMs — *A Survey of Large Audio Language Models: Generalization, Trustworthiness, and Outlook* (2026)
- Source: `Music Analysis/2605.20266v1.pdf`.
- Taxonomy: hallucination, robustness, safety, privacy, fairness, authentication.
- **Code changes:**
  - Add **authentication** to the acquisition gate: is the downloaded audio actually the claimed track?
  - Add **hallucination check**: does the model's placement have evidence in the audio channels, or is it hallucinating?
- **Issue:** fold into #41.

---

## Papers for the north-north star (personalization / mashup)

### 7. MusicSem — *MusicSem: A Semantically Rich Language–Audio Dataset of Natural Music Descriptions* (2026)
- Source: arXiv 2602.17769.
- Organic Reddit descriptions ("late-night driving") instead of synthetic captions.
- **Use:** `personalization/` cohort modeling and natural-language mashup search.

### 8. CultureMERT — continual pretraining across cultures (ISMIR 2025)
- Source: ISMIR 2025 program.
- **Use:** initialize any domain-adaptation fine-tuning; add domain detection to the aligner for non-EDM sets.

### 9. Universal Music Representations — *Are foundation models actually universal?* (ISMIR 2025)
- Source: ISMIR 2025 program.
- **Use:** justification for explicit abstention / domain-gating instead of assuming zero-shot generalization.

### 10. Music-QA Benchmarks (ISMIR 2025)
- Source: ISMIR 2025 program.
- **Use:** evaluate the aligner with reasoning questions: "Does this span contain the claimed track?" "Is the claimed stem consistent with the audio?"

### 11. Human-preference similarity learning — *Music Similarity Representation Learning Focusing on Individual Instruments with Source Separation and Human Preference* (2025)
- Source: arXiv 2503.18486.
- **Use:** mashup compatibility and transition-quality scoring, replacing spectral cosine with learned human preference.

### 12. Abdallah & Plumbley — *Information Dynamics: Patterns of Expectation and Surprise in the Perception of Music* (2008)
- Source: `Music Papers 2026/Information Dynamics - Patterns of Expectation and Suprise in the Perception of Music.pdf`.
- **Use:** expectation/surprise as a structure feature for transition detection and alignment.

---

## Open whitespace

None of the above papers combine all of:
- audio retrieval
- DJ transitions
- sequential planning
- reinforcement learning / dynamic programming
- weak supervision
- crowd/user behavior
- personalization

That integrated "music intelligence engine" remains the long-term whitespace for `lab/` and the mashup compiler.

---

## Tracking

- **Operative plan: [operation_crush_assault_plan.md](operation_crush_assault_plan.md)** (2026-07-20) — phased plan + discrepancy register; this doc stays the paper-mapping layer.
- Milestone: [Operation Crush](https://github.com/jca225/tracklist_engine/milestone/1)
- Issue #40: data integrity / identity / GT capture
- Issue #41: agentic acquisition / verification gate
- DJtransGAN task: dedicated issue under Operation Crush
