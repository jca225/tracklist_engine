# DJ Performance-Operation Ontology for PWS Labeling Functions

**Date:** 2026-07-14 · **Branch:** `pws-alignment-reframe`
**Provenance:** deep-research workflow (109 agents, 26 sources fetched, 126 claims
extracted, top 25 adversarially verified by 3-voter panels → 22 confirmed, 3 refuted).
Companion to [2026-07-14-pws-aligner-design.md](superpowers/specs/2026-07-14-pws-aligner-design.md).

**Verification legend used throughout:**
- ✅ **verified** — survived 3-voter adversarial verification (3-0)
- ◻ **surfaced** — extracted from a primary source but budget-dropped before
  verification (not refuted; treat as a lead, re-verify before load-bearing use)
- ✗ **refuted** — killed 0-3; listed in §7 so nobody re-cites it
- ⚠ **hypothesis** — authored here from domain knowledge; no external verification

Per the SSOT rule, nothing in this doc is an alignment status number. All
manual-derived detector signatures are predictions **not yet validated against
recorded mixes** (see §8, Open Question 4).

---

## 1. The headline result: the operation vocabulary is mixer-model-specific

The single most load-bearing verified finding: **there is no one "Pioneer FX list."
LFs must be keyed by mixer model and generation**, and conditioned on set date —
this is exactly a FABLE instance-feature (condition per-LF accuracy on rig context).

| Fact | Status | Source |
|---|---|---|
| DJM-A9 Beat FX is a **closed 14-item list**: DELAY, ECHO, PING PONG, SPIRAL, HELIX, REVERB, FLANGER, PHASER, FILTER, TRIPLET FILTER, TRANS, ROLL, TRIPLET ROLL, **MOBIUS**. It does **not** have SLIP ROLL / VINYL BRAKE / PITCH (those are 900NXS2/V10-generation). | ✅ 3-0 | [A9 manual (B&H lit PDF)](https://www.bhphotovideo.com/lit_files/963732.pdf), [ManualsLib mirror](https://www.manualslib.com/manual/3001872/Pioneer-Dj-Djm-A9.html) |
| MOBIUS = beat-synced endlessly rising/falling oscillator (Shepard-tone-like), 1/16–64-beat cycles — an **A9-specific riser class** that simply cannot occur in pre-A9 sets. | ✅ 3-0 | A9 manual pp. 52–56 |
| DJM-V10 has **no Sound Color FX bank at all** (no Noise/Sweep/Crush/Space). Instead: per-channel HPF/LPF filter with resonance + a **send section** (Short Delay, Long Delay, Dub Echo, Reverb). Its 14 Beat FX include SHIMMER, PITCH, VINYL BRAKE. | ✅ 3-0 | [V10 manual](https://www.manualslib.com/manual/1824795/Pioneer-Dj-Djm-V10.html), [Pioneer DJ Store spec](https://pioneerdjstore.com/products/djm-v10) |
| V10 EQ is **4-band** (HI, HI MID, LOW MID, LOW); mids floor at **−26 dB — cannot full-kill** — vs −inf on HI/LOW and vs the 3-band (−inf-capable) EQ/isolator of the 900NXS2 and A9. EQ-kill detectors must model band topology per mixer. | ✅ 3-0 | V10 manual; [Pioneer 4-band-EQ feature page](https://pioneerdj.com/en/product/features/mixer/djm-v10-4-band-eq-and-compressor/) |
| 900NXS2-era SWEEP (counterclockwise) is documented as a **gate** effect; A9-era SWEEP is a moving **band-stop/band-pass**. Same knob name, different signal signature across generations. | ✅ 3-0 | A9 manual vs [900NXS2 manual](https://novelty.fr/wp-content/uploads/downloaded/downloads/materiel_manuels/pioneer_djm-900nxs2_manual_EN.pdf) |

Citation caveat: the B&H `lit_files` and HAL PDFs intermittently 403 to bots; content
was verified verbatim against ManualsLib / Wayback / French-edition mirrors.

## 2. Verified detector signatures (the physics each LF keys on)

All from manufacturer manuals (✅ 3-0 unless noted); each carries the verifier
panel's caveats inline.

1. **Echo family** — ECHO outputs repeats "while attenuating it according to the
   beat," BEAT 1/16–16 beats, TIME 1–4000 ms. Detector: decaying repeats at lags
   that are rational fractions/multiples of the local beat period. *Caveats:* the
   TIME knob permits arbitrary non-beat lags (beat-rational is a default, not a
   guarantee); "exponential decay" is an inference from "attenuating."
   **PING PONG** adds distinct L/R delay times — a stereo cue separating it from ECHO.
2. **Echo-out transition** — explicitly documented affordance: cut the channel
   fader to 0 and "only the effect sound remains." Detector: abrupt disappearance
   of the dry source while a beat-synced decaying tail continues. *Caveat:*
   tail-persists-on-fader-cut also fires for ROLL-family and REVERB —
   **necessary but not unique**; disambiguate by tail structure (discrete repeats
   vs dense reverb vs looped grain).
3. **NOISE (A9 Sound Color FX)** — "filtered white noise **mixed with** the sound
   of the channel," COLOR = noise-filter cutoff. Detector: broadband noise
   component whose spectral centroid sweeps over time, **superimposed on** (not
   replacing) program audio. **SWEEP** — moving band-stop (notch) or band-pass.
   A9-scoped (see §1 for the NXS2 gate divergence).
4. **CDJ-3000 tempo slider** — four ranges ±6/±10/±16/WIDE(±100)% with fixed
   resolutions 0.02 / 0.05 / 0.05 / 0.5%; playback stops at −100%. Detector prior:
   manual varispeed ratios land on a quantized step grid. *Caveats:* BEAT SYNC sets
   arbitrary off-grid ratios; jog pitch-bend adds continuous transients. **Soft
   prior, not a constraint.**
5. **Key-lock vs varispeed** — Master Tempo ON = speed change with pitch
   preserved (time-stretch); OFF = linked pitch+tempo scaling. Detector: stretch
   artifacts vs proportional shift. *Caveat:* pitch preservation is guaranteed by
   the manual; **formant preservation is algorithm-dependent — do not assume it.**
6. **Key Shift / Key Sync** — manual Key Shift moves in **exact integer
   semitones**; Key Sync snaps to the least-change key among {same, dominant,
   subdominant, relative, relative-of-dominant, relative-of-subdominant},
   **capping Key Sync at ±2 semitones**. Detector prior: deck-side key moves are
   integer-semitone; continuous detune ⇒ varispeed or DAW repitch instead.
7. **Quantize** — with Quantize on, cue/loop-in/loop-out/hot-cue points snap to
   the closest **rekordbox-analyzed** beatgrid position; Quantize Beat Value is
   configurable down to 1/8 beat. Detector prior: event onsets on a **sub-beat
   lattice of the analyzed grid** (which can diverge from the perceptual beat,
   especially on swung material). Quantize requires rekordbox analysis and is a
   toggle — soft prior.

## 3. FX-detection & mix-reverse-engineering prior art (all ✅ 3-0)

- **André, Fourer & Schwarz 2024** ([arXiv:2410.04198](https://arxiv.org/abs/2410.04198)),
  multi-pass NMF DJ-mix transcription — formalizes exactly our generative-inversion
  framing: mix spectrogram = Σ time-warped, gain-modulated source spectrograms +
  additive noise. Deliberately tiny operation vocabulary (warp + gain only;
  pitch-shift/EQ disregarded; reverb/distortion/positive-EQ lumped into noise). Its
  SOTA-ness is **expressiveness** (unconstrained warp can represent loops/jumps),
  **not error metrics**: gain error beats the DTW baseline only under none/bass-boost
  conditions and warp error is *worse* (the baseline exploits UnmixDB's affine-stretch
  assumption). The one-hour-scaling claim was **refuted** (§7) — do not cite scaling.
  Matches the existing [[project_dj_mix_prior_art]] verdict.
- **Schwarz & Fourer 2019** (CMMR, [HAL](https://hal.science/hal-02172427v1/document),
  [proceedings PDF](https://cmmr2019.prism.cnrs.fr/Docs/Proceedings_CMMR2019.pdf)) —
  five-step pipeline (MFCC-DTW rough align → cross-correlation sample align →
  track-removal verification → T-F gain-curve estimation → linear-regression cue
  estimation). 25 ms median start-time error on synthetic UnmixDB (~100 ms under
  distortion). **Directly reusable gain LF:** fade curve = median over active bins of
  |X(n,m′)|/|Sᵢ(n,m′)| (their Eq. 2); median fade error ≈ 0.3 dB over ~16 s fades.
  *Caveat:* fully synthetic eval, linear crossfades — expect degradation on real
  nonlinear/EQ fades.
- **The scope boundary that justifies our LF catalog:** Schwarz & Fourer explicitly
  handle level-1 "broadcast" (volume crossfade) and level-2 "lounge" (beat-synced +
  speed + EQ) mixing and scope out level-3 **"performative mixing — creative use of
  effects, loops, and mashups"** because it "can blur the identifiability of the
  source tracks." The performance vocabulary in §4 is precisely what optimization
  prior art declines to model — PWS is **complementary, not redundant**.
- **Kim et al. 2020** (ISMIR, [arXiv:2008.10267](https://arxiv.org/abs/2008.10267)) —
  key-invariant subsequence DTW over beat-synchronous CENS chroma (12 circular
  shifts) + MFCC at corpus scale: 1,557 real 1001Tracklists mixes, 13,728 tracks,
  20,765 transitions, gated by match-rate ≥ 0.4 (fraction of diagonal one-beat
  warp-path moves). *Caveat:* cue-point errors are coarse (median ~11–28 s; >80%
  only within 30 s) — validates **scale and gating strategy, not fine placement**.
- **Tempo/key distribution estimates** (Kim et al., 24,202 aligned plays): tempo
  adjustment double-exponentially concentrated near zero — **86.1% < 5%, 94.5% <
  10%, 98.6% < 20%**; transposition rare — **2.5% of tracks, 94.3% of those exactly
  one semitone** ⇒ DJs leave Master Tempo on by default.
  **Status: weak initialization only — these are detector-limited, selection-biased
  estimates, not ground truth about DJ behavior** (John, 2026-07-14). Why:
  (a) *survivorship bias is baked in* — only plays passing the match-rate ≥ 0.4
  chroma-DTW gate were counted, and heavily warped/transposed plays align worst, so
  they're filtered before measurement; the figures are "adjustments the detector
  could confirm," not the population; (b) *the instrument is blind* to sub-semitone
  repitch (integer chroma shifts) and to fine warp play (~11–28 s median placement);
  (c) *circularity risk* — baking these in as strong priors biases our aligner
  toward easy alignments and reproduces their selection effect (the
  [[project_opinion_audit]] failure mode in prior form); (d) 2020 data predates
  CDJ-3000 key-sync and the stems era; (e) per our own BB11 GT, **31% of acappella
  overlays are re-pitched** ([[project_key_change_breaks_chroma]]) — full-track
  plays only, **never the stem axis**. Use as label-model initialization the model
  overwrites, never as a gate. Note we already hold a native GT-derived tempo prior
  (`warp_prior.json`, n=316, beds ≈ N(1, 0.012) — [[project_warp_prior_phase0]]), so
  Kim et al. is corroboration, not foundation — and **re-estimating these
  distributions at 40k scale with a stronger instrument is itself a deliverable of
  our benchmark/paper effort.**
- **Chen et al. 2022** (ICASSP, [arXiv:2110.06525](https://arxiv.org/pdf/2110.06525)),
  differentiable DJ mixer + GAN — operationalizes the transition as fade curves +
  per-band DDSP EQ ("EQs and faders are two essential components in the DJ-made
  mixing effects"): supports a transition-region LF parameterized by fade + per-band
  EQ, a **core subset, not the full grammar** (echo/reverb/roll/noise transitions
  unmodeled). Replicable corpus recipe: 7,064 transition sections from 284
  human-annotated livetracklist mixes (consecutive-track sections, <1 min discarded).
  Audio never released (copyright); boundaries are crowd-grade — suited to weak
  supervision, not precise eval GT. The "24 trainable parameters" claim was
  **refuted** (§7).

## 4. The LF catalog (proposal)

⚠ **This section is authored, not externally verified** — each LF cites its §2/§3
grounding where one exists. Expected-precision entries are priors for the label
model to overwrite, not claims. Existing repo probes are marked `[exists]` — they
are already LFs in the PWS sense (`Probe` ABC + abstain).

**Reminder from the gate verdict ([[project_pws_gate_verdict]]):** categorical
DS-style aggregation is for **these operation labels**; continuous offsets need the
EM-over-per-probe-σ label model. Don't bin offsets to feed a categorical model again.

### Family A — Gain / EQ (level-1/2; grounded §3)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| A1 | `lf_gain_fade` | Smooth per-track gain trajectory | Median STFT ratio \|X\|/\|Sᵢ\| over active bins (Schwarz & Fourer Eq. 2) vs aligned ref | high on clean blends | all |
| A2 | `lf_eq_band_kill` | Band-limited attenuation of one source | Per-band ratio curves; **model 3-band −inf vs V10 4-band (−26 dB mid floor)** | med-high | CDJ clusters |
| A3 | `lf_bass_swap` | Low-band energy handoff between sources at phrase boundary | Low-band source-attribution flip synchronous with grid | med | house/tech-house |
| A4 | `lf_crossfade_overlap` | Two refs simultaneously active, complementary gains | Joint activation from A1 curves | high | all |

### Family B — Delay / echo / reverb (grounded §2.1–2.2)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| B1 | `lf_echo_tail` | Decaying repeats at beat-rational lags (1/16–16 beats) | Autocorrelation of residual (mix − aligned refs) at beat-rational lags | med-high | all CDJ |
| B2 | `lf_echo_out` | Dry source vanishes; beat-synced decaying tail persists | A1 gain cliff + B1 tail continuation; **disambiguate from roll/reverb by tail structure** | high (as transition marker) | all CDJ |
| B3 | `lf_pingpong` | Alternating L/R repeat lags | Stereo channel-difference periodicity | high, low recall | CDJ perf |
| B4 | `lf_spiral_helix` | Feedback echo with rising pitch/turntable-wind character | B1 + monotonic pitch trajectory in tail | med | CDJ perf, festival |
| B5 | `lf_reverb_tail` | Dense non-quantized tail; SHIMMER adds +1-octave content (V10) | Decay-rate + spectral-flatness of residual tail; octave-band check | med | all |
| B6 | `lf_dub_echo_send` | Filtered feedback repeats (band-narrowing per repeat) | B1 + per-repeat bandwidth shrinkage | med | V10-era, house |

### Family C — Noise / filter / riser (grounded §2.3, §1)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| C1 | `lf_noise_sweep` | Broadband noise **superimposed** on program, sweeping centroid | Residual spectral-flatness ↑ + centroid trajectory | med-high | all CDJ |
| C2 | `lf_sweep_notch` | Moving notch/band-pass on the program itself | Time-varying spectral dip tracking; **gate-not-notch on NXS2-era sets** | med | CDJ |
| C3 | `lf_filter_sweep` | Global HPF/LPF with resonance ridge | Cutoff trajectory + resonance peak vs aligned ref spectrum | med-high | all |
| C4 | `lf_mobius_riser` | Shepard-tone rise locked to beat (1/16–64 beats) | Chroma-circular ascending pattern, constant spectral envelope; **A9-era only** | high, low recall | post-2023 CDJ |
| C5 | `lf_trans_gate` | Beat-synced amplitude gating | Amplitude-modulation comb at beat divisions | high | CDJ perf |
| C6 | `lf_crush_flanger_phaser` | Bit-crush aliasing / sweeping comb notches | Harmonic-distortion or moving comb detection on residual | low-med | CDJ |
| C7 | `lf_riser_sample` | DAW riser one-shot: noise/tonal crescendo into a drop, **not** derived from either ref | Non-catalog broadband object ending exactly at a downbeat drop | med | DAW-mashup, festival |
| C8 | `lf_impact_downlifter` | One-shot impact/sub-drop at phrase boundary | Transient + pitch-fall template at grid boundary | med | DAW-mashup, bass |

### Family D — Loop / roll / jump / structure (grounded §2.7; repo fibers/looptrace)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| D1 | `lf_roll_micro` | Exact-repeat micro-loop (≤1 beat, incl. triplet lattice) | Short-lag exact-copy autocorrelation; slip-resume check | high | CDJ perf |
| D2 | `lf_loop_periodicity` `[exists: fibers/looptrace]` | 4/8/16-beat exact repeats | Fiber self-repeat classes ([[project_fibers]]) | per fiber-v4 gate | all |
| D3 | `lf_beat_jump` | Ref-time offset discontinuity of exactly N beats, mix continuity preserved | Piecewise offset decode jump = integer×beat ([[project_path_decode]]) | high | CDJ perf |
| D4 | `lf_hot_cue_jump` | Offset jump landing on a cue-like ref position | D3 + cue-detr ref-time prior ([[project_cue_detr_ref_offset_prior]], soft channel) | med | CDJ perf |
| D5 | `lf_quantize_lattice` | Event onsets on sub-beat lattice of analyzed grid | Onset-to-grid residual histogram (tolerate 1/8-beat lattice) | med (soft prior) | all CDJ |
| D6 | `lf_vinyl_brake` | Exponential pitch+tempo fall to stop | Joint pitch/tempo decay template | high, low recall | V10/NXS2-era |

### Family E — Tempo / pitch (grounded §2.4–2.6, §3 priors)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| E1 | `lf_tempo_step_grid` | Warp ratio on quantized slider steps (0.02/0.05/0.5%) | Ratio-mod-step residual; **soft prior** (sync/jog break it) | low-med | CDJ |
| E2 | `lf_tempo_prior` | Warp magnitude concentrated near 1 | Native GT prior `warp_prior.json` (n=316) primary; Kim2020 figures = weak init only (see §3 status note) | prior, not detector | all |
| E3 | `lf_keylock_stretch` | Pitch preserved while tempo shifts; stretch artifacts | Pitch-track constancy vs tempo ratio ≠ 1; **don't assume formant preservation** | med-high | all |
| E4 | `lf_varispeed_linked` | Pitch shift exactly proportional to tempo ratio | log-pitch-shift ≈ log-tempo-ratio test | high | CDJ, UnmixDB-style |
| E5 | `lf_key_shift_semitone` | Integer-semitone shift; Key Sync capped ±2 st | Chroma circular shift = integer; ±2 window for sync | high | CDJ |
| E6 | `lf_micropitch_detune` `[exists: pitch_detune.py]` | Sub-semitone detune of overlay to bed key | bb_pitch_detune_v1 ([[project_micropitch_detune]]) | per its eval | DAW-mashup |

### Family F — Mashup / overlay / DAW (the Two Friends / GT-corpus cluster)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| F1 | `lf_acapella_overlay` `[exists: stem-routed HuBERT]` | Vocal energy with no matching instrumental from same ref | [[project_stem_match_bootstrap]] identity-under-crosstalk | per scorecard | DAW-mashup, CDJ+stems |
| F2 | `lf_double_drop` | Two refs' drop sections co-located on one downbeat | Two high-confidence placements + both refs at drop cues | med-high | festival, CDJ perf |
| F3 | `lf_sidechain_pump` | Periodic amplitude dip at beat rate on sustained overlay | Beat-locked AM depth on non-percussive band; note bb grammar says **LUFS-match, not sidechain** — expect low firing rate on BB ([[project_mashup_grammar_prior]]) | med | DAW-mashup |
| F4 | `lf_vocal_span_grammar` | ~30 s/16-bar vocal spans, pickup-led, from first chorus | bb_mashup_grammar_v1 span prior | prior | DAW-mashup |
| F5 | `lf_drop_tag_jingle` | Recurring non-catalog audio object across sets by same DJ (e.g. "Big Bootie" tags) | Cross-set recurrence clustering of unmatched spans (§6 loop) | high once clustered | DAW-mashup, festival |
| F6 | `lf_lufs_matched_overlay` | Overlay level ≈ bed level (LUFS) | Relative loudness of separated overlay vs bed | prior | DAW-mashup |

### Family G — Live / venue (hybrid & recorded-live sets)

| # | LF | Signal signature | Detector sketch | Exp. precision | Clusters |
|---|---|---|---|---|---|
| G1 | `lf_live_instrument` | Sustained melodic layer in set key/tempo matching **no** catalog ref (trumpet, drums, keys) | Non-catalog harmonic object + onset coherence with grid | low-med | live-hybrid |
| G2 | `lf_mc_vocal` | Speech/hype vocal over program, not in any ref | VAD on residual + ASR-not-in-lyrics check (reuse lyrics channel) | med | festival, live |
| G3 | `lf_crowd_ambience` | Crowd noise floor, venue reverb on everything | Broadband noise floor + global reverb-time estimate | high (as *live-recording* flag, conditions all other LFs) | live recordings |

### Family H — Identity / offset channels (already built — listed for completeness)

`fp_probe` ([[project_fingerprint_localizer]]), HuBERT vocal ref-offset, chroma
offset, lyrics-ASR ([[project_lyrics_ref_decode]]), mel reconstruction,
boundary novelty (`surprise`), per-stem HuBERT set-start, instrumental stem-fp.
These are the repo's existing probes = LFs; the PWS move is putting **all of the
above under one label model** with learned, feature-conditioned accuracies.

**Coverage note (no silent caps):** ~44 LFs total incl. existing probes. Families
B/C/D are CDJ-club-scoped and may fire rarely on the BB GT corpus if BB mixes are
DAW-composited (§5, Open Question 1) — which is precisely why per-cluster
conditioning matters before reading LF empirical accuracy as truth.

## 5. Roster clustering — mostly ⚠ hypothesis (verification largely failed here)

**Honest accounting:** of the per-DJ rig research, exactly **one** claim survived:
the official DJ Mag "How I DJ" tutorial (2021, Pioneer-sponsored,
[youtube 6hXc_oClj24](https://www.youtube.com/watch?v=6hXc_oClj24)) is structured as
named timestamped segments — four-deck mixing (0:56), hot cues (2:13), looping
(6:25), mixer section (11:09) — usable for **operation-class seeding** (✅ 3-0).
The inference that this defines James Hype's performance vocabulary/cluster was
**refuted 0-3**. No claims survived for Ableton acts, live-instrument acts, or —
critically — **the Two Friends Big Bootie production workflow (our GT corpus)**.

The clustering below is therefore a **prior for FABLE conditioning, with each DJ's
cluster assignment itself a latent variable** — not a fact table. A targeted
follow-up pass (interviews, stage photos/videos per DJ, rider documents) is Open
Question 1.

| Cluster (⚠) | Hypothesized members (from `data/djs/` roster) | Dominant LF families |
|---|---|---|
| **DAW-mashup composited** | Two Friends (BB), Party Pupils, Lost Kings, Cheat Codes, Mako | F, C7–C8, E6; club families B–D largely absent |
| **CDJ performance / 4-deck** | James Hype, Fisher, John Summit, Dom Dolla, Chris Lake, Cloonee, PAWSA, Matroda, Westend, Biscits, Noizu, Odd Mob, Sonny Fodera, Kyle Watson, Walker & Royce, Wax Motif, Shiba San, Eli Brown, Chris Lorenzo, Dombresky, Hugel, CID, Deeper Purpose | B, C1–C5, D, E |
| **Festival mainstage / prepared edits** | Martin Garrix, Hardwell, Tiesto, DVLM, Afrojack, Nicky Romero, R3hab, Blasterjaxx, Bassjackers, Timmy Trumpet (+trumpet→G1), Vini Vici, KSHMR (+orchestration→G1), Steve Aoki, W&W-adjacent, Alesso, Axwell/Ingrosso/Angello/SHM, DJ Snake, Dada Life | F2, C7–C8, B, G2 |
| **Melodic / live-hybrid (Ableton)** | Illenium (drums), Odesza (band), Rufus Du Sol (band), Gryffin (guitar/keys), Porter Robinson, Madeon, Kygo, Lane 8, Ben Böhmer, Zedd | G1, F, A; Quantize/step-grid priors (D5, E1) do **not** apply |
| **House / melodic-house blends** | Disclosure, Green Velvet, CamelPhat, Meduza, Vintage Culture, ZHU, Tchami, Malaa-adjacent, Moguai, Deadmau5 (custom rig) | A, B, C3, long blends |
| **Bass / trap / mid-tempo** | RL Grime, Jauz, Ghastly, Valentino Khan, Dillon Francis, AC Slater, Dr. Fresch, Habstrakt, Moksi, Louis The Child, Diplo, Marshmello, Morten | C7–C8, D1, F2, halftime/double-time switches |
| **Pop-dance / radio** | Chainsmokers, Alan Walker, Alok, Jonas Blue, Sigala, Sam Feldt, Felix Jaehn, Bakermat, Lucas & Steve, Mike Williams, Brooks, Don Diablo, Oliver Heldens, Lost Frequencies, Galantis, Vicetone, Retrovision, Shermanology, Nimino, Maup, D.O.D | A, F, shorter radio-style transitions |
| **Legacy (pre-CDJ-3000 era sets)** | Avicii, early Calvin Harris/Guetta/Armin — condition on **set date × mixer generation** (§1) | era-keyed variants of all families |

## 6. Open-vocabulary ID/operation discovery — the ViLD analogy

**Can we do this? Yes — the recipe maps 1:1, and audio-side precedents exist (all ◻
surfaced, none adversarially verified — first task is to verify these five).**

ViLD ([arXiv:2104.13921](https://arxiv.org/pdf/2104.13921)): (1) class-agnostic
region proposals from an RPN, (2) distill CLIP image embeddings into the detector's
region embeddings, (3) classify by cosine against **text embeddings** of class
names — so novel classes are just new text at inference, no retraining.

The audio mapping:

| ViLD component | Our analog | Status |
|---|---|---|
| Class-agnostic region proposals | Foote novelty ([[project_boundary_novelty_placement_prior]]), fiber boundaries, fp-break segmentation — **already built** | exists |
| CLIP image tower | **CLAP** zero-shot audio-text classification ([arXiv:2409.09213](https://arxiv.org/abs/2409.09213) ◻, [arXiv:2310.13759](https://arxiv.org/abs/2310.13759) ◻); **MuLan** two-tower music-text model, 44 M recordings ([arXiv:2208.12415](https://arxiv.org/abs/2208.12415) ◻) | ◻ surfaced |
| Text-embedding classifier head | **FlexSED** — open-vocab SED, audio SSL encoder + CLAP text encoder, free-text queries ([arXiv:2509.18606](https://arxiv.org/abs/2509.18606) ◻); **DASM** — frame-level retrieval against text **or audio** query vectors ([arXiv:2507.16343](https://arxiv.org/abs/2507.16343) ◻) — the audio-query path is exactly what unnamed IDs need | ◻ surfaced |
| Novel-class handling | **OW-SED** open-world SED paradigm: detect known, flag unseen, incrementally learn ([arXiv:2605.03934](https://arxiv.org/abs/2605.03934) ◻) | ◻ surfaced |
| Per-instrument FX repr. | **Fx-Encoder++**: instrument-specific FX embeddings from full mixtures via CLAP-derived queries, no separation ([arXiv:2507.02273](https://arxiv.org/html/2507.02273) ◻) — candidate backbone for per-stem FX LFs (B/C on an acappella overlay) | ◻ surfaced |

**The discovery loop** (Phase-2 of the PWS design, now grounded):

1. **Propose:** spans where all identity LFs abstain → class-agnostic "some
   audio object" proposals (never silently discard — the [[project_opinion_audit]]
   failure mode this fixes).
2. **Embed:** CLAP/MuLan-style span embeddings (+ fp landmarks for exact-copy
   matching). *Known risk:* [[project_mert_equivalence_floor]] (~0.92 within-track
   self-similarity) warns generic music embeddings may be too coarse to separate
   near-duplicates — validate embedding discriminability on BB GT before building.
3. **Cluster across sets:** the same unreleased ID recurs across many sets in a
   season; recurrence clustering turns N unknowns into one cluster with N
   supports. This is the community's own mechanism, industrialized —
   [TrackId.net](https://trackid.net/) (◻) already runs fingerprint-based
   auto-tracklisting with unresolved-span "ID" segments crowd-resolved over time;
   [1001Tracklists](https://www.1001tracklists.com/) (◻) IDs carry the `ID - ID`
   convention our tokenizer already parses. Cluster mass estimation connects to
   the capture-recapture/Good-Turing machinery in
   [[project_benchmark_certification]].
4. **Name or surface:** cosine against a name library (operation names from §4;
   track names from the corpus; audio queries from ref library). High margin →
   auto-name ([[project_abstention_margin]]: margin, not absolute cosine). Low
   margin → **LISTENING QUEUE**, ordered by the acquisition function below.

## 7. Refuted claims — do not re-cite

| Claim | Vote | Source |
|---|---|---|
| James Hype's core performance vocabulary = four-deck/EQ/loops/hot-cues, placing him in a CDJ-performance cluster | 0-3 | the DJ Mag tutorial (structure verified, inference refuted) |
| André 2024 multi-pass NMF "scales to realistic one-hour mixes" and recovers loops/jumps a DTW baseline cannot | 0-3 | arXiv:2410.04198 |
| A full DJ transition is representable in a 24-parameter differentiable mixer | 0-3 | arXiv:2110.06525 |

## 8. Human-as-oracle (John) — budgeted gold, not an LF row

Direction set 2026-07-14: human input joins as a **budgeted oracle** — high
quality, minimal usage — NOT as a labeling function in the matrix. Rationale: the
label model's job is learning unknown LF accuracies; John's accuracy ≈ 1.0 is known
a priori, so spending unlabeled-data statistics to estimate it is waste.

- **Mechanism:** verdicts enter as *observed* variables (gold anchors) that pin
  label-model calibration — the weak-supervision + active-learning hybrid
  (Active WeaSuL-style; also how Confident Learning uses small trusted sets).
- **Acquisition rule:** surface spans maximizing information about **per-LF
  accuracy parameters** (high posterior entropy × high parameter influence), not
  merely hard spans — one answer recalibrates an LF across all ~40k sets.
- **Routing:** existing LISTENING QUEUE, one batched `.als` per session
  ([[feedback_batch_listening_jobs]]), reordered by the acquisition function.
- **Question design:** prefer small-vocabulary verdicts that calibrate a *class*
  ("noise sweep or riser sample?") over span-specific facts.

## 9. Open questions → next actions

1. **Per-DJ rig verification pass (RQ1 unanswered).** Highest priority: **what
   workflow produces Big Bootie mixes?** If BB is DAW-composited, our GT corpus
   exercises almost none of the club-mixer vocabulary verified here — a
   train/deploy distribution gap the label model must know about. Then a
   structured per-DJ pass (stage videos, "How I DJ"-style interviews, gear
   databases) to replace §5's hypothesis table.
2. **Verify the five ◻ open-vocab papers** (DASM, FlexSED, OW-SED, MuLan, CLAP
   zero-shot, Fx-Encoder++) — read primaries, check benchmarks/code, and run the
   embedding-discriminability check (§6.2) on BB GT.
3. **Stems-era operations:** how do rekordbox/Serato live stems mutes/solos
   (post-2023) manifest in recordings, and do they collide with our
   Demucs/Roformer stem-axis assumptions?
4. **Empirical validation of §2 signatures on BB11/BB12** — which manual-derived
   signatures survive contact with real mastered audio, and at what per-LF
   precision (feeds the label model's priors; per SSOT, results go to
   `docs/alignment_status.md` machinery, not here).
5. **Aligner bug tickets independent of the PWS lane** (from the failure tables):
   the "ref≈0:00 intro-grab" decode collapse, and repeated-track instance
   disambiguation (BB12 −746 s Slide error, four spans from one anchor).
