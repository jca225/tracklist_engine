# Corpus rigor & alignment — related work bibliography

> Curated literature for the **48k real DJ-set corpus** program: weak supervision
> at scale, phenomenon discovery without listening to everything, stratified anchor
> GT, and external calibration. Includes Spotify Research inventory.
>
> **Not canonical metrics** — headline alignment numbers stay in
> [alignment_status.md](alignment_status.md).

---

## How to use this doc

| Tag | Meaning |
|-----|---------|
| **LOAD-BEARING** | Directly informs paper design or engineering |
| **USEFUL** | Methods, baselines, or datasets to cite or borrow |
| **CONTEXT** | Background / adjacent field |
| **LOW FIT** | Known but poor match for alignment gate |

---

## Spotify Research — what helps us

Portal: [research.atspotify.com](https://research.atspotify.com/)  
GitHub org: [github.com/spotify-research](https://github.com/spotify-research)  
Audio Intelligence Lab: [research.atspotify.com/audio-intelligence](https://research.atspotify.com/audio-intelligence)

### High relevance

| Resource | Link | Why it matters for us |
|----------|------|----------------------|
| **Automatic Playlist Sequencing and Transitions** (ISMIR 2017) | [Paper PDF](https://archives.ismir.net/ismir2017/paper/000086.pdf) · [Spotify Research](https://research.atspotify.com/publications/automatic-playlist-sequencing-and-transitions) | DJ-style crossfades, cue-point selection, harmonic/tempo constraints — closest Spotify work to *transition* modeling (not full mix alignment). A/B: +1.4pp playlist return with DJ-curated transitions. |
| **The skipping behavior of users… and musical structure** (Spotify, 2019) | [arXiv:1903.06008](https://arxiv.org/pdf/1903.06008) | **Weak supervision at scale**: skip timestamps correlate with musical structure; used to train structure models without hand labels. Direct analog to SC comment `mix_position_ms` + taste prior. |
| **Basic Pitch** (ICASSP 2022) | [arXiv:2203.09893](https://arxiv.org/abs/2203.09893) · [github.com/spotify/basic-pitch](https://github.com/spotify/basic-pitch) | Polyphonic transcription with **pitch-bend detection** — relevant to BB11 fine-cents vs semitone-only priors. Open source (Apache 2.0). |
| **When the Music Stops: Tip-of-the-Tongue Retrieval** (SIGIR 2023) | [arXiv:2305.14072](https://arxiv.org/abs/2305.14072) · [github.com/spotify-research/tot](https://github.com/spotify-research/tot) | Multi-modal identity search (fingerprint + lyrics + text). Schema for hard identification cases. |
| **LLark** (ICML 2024) | [arXiv:2310.07160](https://arxiv.org/abs/2310.07160) · [github.com/spotify-research/llark](https://github.com/spotify-research/llark) | Instruction-following music understanding; training code + dataset fusion patterns (not released weights). |

### Medium relevance (taste / corpus / metadata)

| Resource | Link | Why |
|----------|------|-----|
| **Music Streaming Sessions Dataset (MSSD)** | [arXiv:1901.09851](https://arxiv.org/abs/1901.09851) · [Spotify Research](https://research.atspotify.com/publications/the-music-streaming-sessions-dataset-short-paper) | 160M sessions, 3.7M tracks with audio features — session sequential behavior for `personalization/` / taste prior (north-north star). |
| **Million Playlist Dataset (remastered)** | [AICrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) · [blog](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered) | 1M playlists, 2M tracks — playlist continuation; not DJ mixes but sequential curation prior. |
| **OpenMIC-2018** | [Spotify Research](https://research.atspotify.com/publications/openmic-2018-an-open-dataset-for-multiple-instrument-recognition) | 20K excerpts, 20 instrument classes — stem/instrument tagging at scale. |
| **WASABI** (Deezer/Inria; used in Spotify ToT) | [Zenodo](https://zenodo.org/records/5603369) · [GitHub](https://github.com/micbuffa/WasabiDataset) | 2M songs, chords/tempo/lyrics metadata — **ISRC-linked** crosswalk to catalog (ToT paper uses `wasabi_isrc`). Useful for chart/popularity joins on `lab/corpus_empirics/`. |

### Low fit for alignment gate (cite as contrast)

| Resource | Why not central |
|----------|-----------------|
| Semantic IDs / LLM personalization (NeurIPS 2025 blog posts) | Recommendation, not mix alignment |
| Podcast corpus / ads research | Wrong modality |
| **Deprecated Spotify Web API** `GET /audio-analysis`, `GET /audio-features` (removed ~Nov 2024) | Was Echo Nest beats/bars/chroma — **do not build new pipeline on this**. We run Essentia + beat_this locally instead. |
| ISMIR 2021 late-breaking “Visualizing structure using Spotify data” | Historical; API deprecated |

### Spotify ↔ our stack (already connected)

- **Essentia** models cache on pi-storage trace to **Echo Nest / Spotify analyzer lineage** (key, BPM, chroma) — same family as deprecated API, but we own the inference.
- **`lab/corpus_empirics/`** already joins **Spotify charts** + release dates — WASABI/ISRC is an optional enrichment arm, not a dependency.
- **Basic Pitch** is a candidate **corpus-wide pitch-incidence probe** (Layer 1) alongside `pitch_detune.py` — test on acappella stems before adopting.

---

## Tier 1 — DJ mix alignment & datasets (LOAD-BEARING)

| Paper | Link | Role |
|-------|------|------|
| **Kim et al. 2020** — Computational Analysis of Real-World DJ Mixes (1001Tracklists) | [ISMIR PDF](https://archives.ismir.net/ismir2020/paper/000187.pdf) · [arXiv:2008.10267](https://arxiv.org/abs/2008.10267) · [Code](https://github.com/mir-aidj/djmix-analysis) | **Direct predecessor**: 1,557 mixes, co-author Evan Sacks (1001TL). DTW alignment → tempo/key/cue stats. Key transpose rare (2.5%). Extend to 48k + phenomenon tails. |
| **Raveform** (Kim et al., TISMIR 2026) | [Paper](https://transactions.ismir.net/articles/10.5334/tismir.288) · [Site](https://mir-aidj.github.io/raveform/) | 4,902 mix links, 56,873 tracks, 1,423 structure-annotated — large weak + sparse gold pattern. |
| **André et al. 2024** — DJ Mix Transcription (Multi-Pass NMF) | [arXiv:2410.04198](https://arxiv.org/abs/2410.04198) | External SOTA; UnmixDB eval — we already bench in `external/unmixdb_findings.md`. |
| **Schwarz & Fourer** — UnmixDB & reverse-engineering methods | [HAL](https://hal.archives-ouvertes.fr/hal-02010431) · [Creation repo](https://github.com/Ircam-RnD/unmixdb-creation) | Synthetic GT wind tunnel (2,460 mixes). |
| **Sonnleitner et al. 2016** — Landmark Fingerprinting for DJ Mix Monitoring | [ISMIR PDF](https://archives.ismir.net/ismir2016/paper/000187.pdf) · [Mixotic data](https://www.cp.jku.at/datasets/fingerprinting/) | Real club mixes; pitch/tempo/crossfade; fp lineage → our `landmark_fp`. |
| **Werthen-Brabants 2018** — GT extraction & transition analysis | [GitHub](https://github.com/werthen/dj-mix-ground-truth-extractor) | Automated reverse-engineering when sources known. |
| **Cue-DETR / EDM-CUE** | [arXiv:2407.06823](https://arxiv.org/html/2407.06823v1) | 21k cue points — point-level supervision (cheaper than Ableton). |

---

## Tier 2 — Scale without full GT (LOAD-BEARING)

| Paper | Link | Role |
|-------|------|------|
| **Jiang et al. 2022 — Rare Example Mining (REM)** | [arXiv:2210.08375](https://arxiv.org/abs/2210.08375) | **Selection policy**: rareness in feature space ≠ difficulty. Mine long-tail phenomena for listening budget. |
| **Northcutt et al. 2021 — Confident Learning** | [arXiv:1911.00068](https://arxiv.org/abs/1911.00068) · [cleanlab](https://github.com/cleanlab/cleanlab) | Tracklist scrape claims = noisy labels; find mis-routes before training. |
| **Swayamdipta et al. 2020 — Dataset Cartography** | [arXiv:2009.10795](https://arxiv.org/abs/2009.10795) · [Code](https://github.com/allenai/cartography) | Easy / ambiguous / mislabeled map from training dynamics. |
| **Kim et al. 2019 — Point-labeled SED** | [WASPAA PDF](https://www.bongjunkim.com/pages/files/papers/waspaa_2019_kim.pdf) | Spacebar timestamps ≈ SC playhead comments — cheap temporal weak labels. |
| **FPSL — Pseudo strong labels** (2025) | [arXiv:2501.03740](https://arxiv.org/abs/2501.03740) | Agentic pseudo-GT flywheel: frame predictions → pseudo boundaries. |
| **Parvaneh et al. 2022 — ALFA-Mix** | [arXiv:2203.07034](https://arxiv.org/abs/2203.07034) | Active learning via representation mixing — find spans with features model hasn't seen. |
| **DiVa** — labels from user comments | [arXiv:2308.04805](https://arxiv.org/pdf/2308.04805) | Iterative weak-label harvest (music tags from comments). |

### Selective prediction & long-tail coverage

| Paper | Link | Role |
|-------|------|------|
| **Conformal Prediction for Long-Tailed Classification** | [arXiv:2507.06867](https://arxiv.org/abs/2507.06867) | Class-conditional coverage for rare phenomena (fine cents, oddratio). |
| **Tail-Aware Conformal Prediction** | [arXiv:2508.11345](https://arxiv.org/html/2508.11345v1) | Head/tail coverage balance in prediction sets. |
| **Selective Conformal Risk Control** | [arXiv:2512.12844](https://arxiv.org/abs/2512.12844) | Abstain + risk guarantees — formalizes our open-mode abstention story. |

---

## Tier 3 — Data-engine framing (USEFUL)

| Paper | Link | Role |
|-------|------|------|
| **Li et al. 2024 — Data-Centric Evolution in AD** | [arXiv:2401.12888](https://arxiv.org/abs/2401.12888) · [Awesome list](https://github.com/LincanLi98/Awesome-Data-Centric-Autonomous-Driving) | Closed-loop taxonomy (acquire → select → label → train → deploy). |
| **Mcity Data Engine 2025** | [arXiv:2504.21614](https://arxiv.org/html/2504.21614v1) · [GitHub](https://github.com/mcity/mcity_data_engine) | Open data-engine reference; cites REM. Vision-only — borrow loop, not code. |
| **AIDE ML (wecoai)** — code-space search | [GitHub](https://github.com/wecoai/aideml) · [arXiv:2502.13138](https://arxiv.org/abs/2502.13138) | Autonomous ML R&D agent — optional tabular sweeps only; not core stack. |

---

## Tier 4 — Adjacent MIR (CONTEXT)

| Paper | Link | Role |
|-------|------|------|
| **DanceMusicSegmentation** (ecsplendid) | [GitHub](https://github.com/ecsplendid/DanceMusicSegmentation) | Unsupervised mix segmentation from tracklist + self-similarity (cuenation-style). |
| **All-In-One structure on demixed audio** | [arXiv:2307.16425](https://arxiv.org/abs/2307.16425) | Beats + segments jointly — complements looptrace. |
| **Discovering concepts in generative music models** | [arXiv:2505.18186](https://arxiv.org/html/2505.18186v3) | Automated phenomenon naming (SAE + LLM labels) — north-north `lab/` pattern. |

---

## Evaluation tier contract (for the paper)

```
Tier A — External calibration     UnmixDB / André regime (synthetic, known GT)
Tier B — Corpus weak (~40k mixes)  Tracklist + automated probes (incidence, abstention)
Tier C — Anchor gold (sparse)      Stratified Ableton GT + triaged clips
```

**Do not** claim Tier B numbers are Tier C accuracy.

---

## Suggested citation sentence

> We evaluate on the largest scraped DJ-tracklist corpus with mix audio (N≈40k),
> following the weak-supervision program of large-scale 1001Tracklists analysis
> (Kim et al., ISMIR 2020) and listener-behavior weak labels (McFee et al., 2019),
> with rare-phenomenon selection (Jiang et al., ECCV 2022), confident learning on
> noisy tracklist claims (Northcutt et al., JAIR 2021), and external calibration
> on UnmixDB (André et al., 2024).

---

## Reading order (10 papers)

1. Kim+Sacks ISMIR 2020 — 1001TL predecessor
2. Jiang REM arXiv:2210.08375 — what to listen to
3. Northcutt Confident Learning — noisy tracklists
4. Kim Point Labels WASPAA 2019 — cheap temporal labels
5. Raveform TISMIR 2026 — large corpus + sparse GT
6. McFee skipping arXiv:1903.06008 — Spotify weak structure labels
7. Swayamdipta Cartography — easy/ambiguous/mislabeled
8. André 2024 — external bench
9. Sonnleitner 2016 — fingerprint + DJ transforms
10. Li Data-Centric AD survey — paper framing

---

## Open questions to research next

- [ ] Run Basic Pitch pitch-bend vs `pitch_detune` log-freq on BB11 acappella clips
- [ ] WASABI ISRC join rate on our `recording` / chart tables
- [ ] Replicate Kim 2020 key-transpose histogram on **full 40k** (automated) vs **mashup stratum**
- [ ] Spotify playlist transition paper: extract cue-point prior for `set_start` baseline

---

*Last updated: 2026-07-13. Add papers via PR; do not hand-type alignment metrics here.*
