# fibers/NOTES.md — research notes (sensor phase closed: notes, not code)

## Phase cancellation as a fiber verifier (2026-07-09, John's idea)

**Idea.** Two instances of the same section overlaid in a DAW phase-cancel if
they are the same recording. Use the null test as a deterministic equivalence
certificate for fibers.

**Status: the mechanism already exists in-repo** — `looptrace/selfsim.py`
audit-v3 is exactly this: whitened-mel lag-diagonal candidate scan →
sample-accurate waveform xcorr with ×8 sub-sample refinement → CLONE iff
residual `1−r²` ≤ −12 dB; `verify_pair_spectral` catches key-locked
(phase-vocoder) copies via STFT-magnitude corr + per-window timing drift.
Fixture-tested (`looptrace/tests/test_selfsim.py`): integer-lag, gain-scaled,
half-sample, codec-noise clones all classify CLONE; random content ~0 dB.

### Empirical probe (2026-07-09, scratchpad; three gt_als test refs)

| ref | pair | r | residual | mag-r | drift | verdict |
|---|---|---|---|---|---|---|
| Congratulations acappella (orig m4a) | chorus 93.6→171.6 s | .955 | −10.6 dB | .96 | **0.0 ms** | **CLONE** (flown-in comp; AAC caps residual above the −12 bar) |
| Emily vocals (Roformer stem) | all 8 machine pairs | .11–.42 | −0.1…−0.8 dB | ≤.90 | 3–22 ms | DISTINCT |
| Emily instrumental (stem) | drop 60.8→155.8 s | .886 | −6.7 dB | .65 | 17.6 ms | near-repeat, not clone |
| Emily FULL original | same drop pair | .651 | −2.4 dB | .60 | 18.8 ms | DISTINCT |

Findings:

1. **The null test works and is deterministic.** It cleanly separates
   Congratulations' flown-in chorus (drift exactly 0.0 ms, mag-r .96) from
   Emily's duplicate-then-tweak repeats — same codec class, unambiguous gap.
2. **The −12 dB CLONE bar is too strict for lossy originals.** A true AAC-coded
   clone caps at ~−10.6 dB. The right certificate is a fused verdict:
   `residual ≤ −12 dB OR (r ≥ .9 AND mag-r ≥ .9 AND drift ≈ 0)`. The spectral
   path already fires correctly (Congratulations classified KEYLOCK).
3. **Separation cuts both ways.** The instrumental *stem* nulls better (.886)
   than the *original* (.651) at the same drop pair — separation removes the
   differing vocal layer above a shared bed. But Roformer artifacts impose a
   noise floor (~−7 dB best observed), so per-lane thresholds are required:
   originals can reach −12 dB; separated stems cannot.
4. **Clone-rate is low on this material** — consistent with the audit-v3
   corpus numbers (BB12 14/414 pairs, BB11 2/234). Phase cancellation is a
   **precision certificate, not a recall engine**: it cannot replace
   HuBERT/harmony fibers as the detector; it grades what they (or the
   selfsim scan) find.

### External literature (2026-07-09 deep-research sweep)

- **Production practice** (producer-lore consensus, no measured survey
  exists): duplicate-then-tweak is the default for drops/choruses; bounced
  **audio-clip duplication → exact clones**; **MIDI re-render through
  analog-emulation plugins never nulls** (random seeds, free-running LFOs —
  "Render doesn't null", Gearspace). BV stacks are the most reliably
  clone-identical element; lead vocals / final-chorus lifts most likely
  differ. Predicts: clone certificates concentrate on instrumental beds and
  backing-vocal stacks; graded near-clone (sparse additive deltas over a
  shared bed) is the modal repeat.
- **MIR gap**: no published measurement of within-track waveform-clone rates;
  structure literature (FMP/thumbnailing) assumes variation-robust features.
  Audit-v3's clone decomposition appears to be novel data.
- **Failure-mode numbers** (Production Expert / LAME FAQ / Apple TN2258 /
  BS-RoFormer paper; details in session transcript): the null test is a
  **lineage detector, not a similarity detector**. Residual vs transformation:
  0.2 dB gain error ≈ −32 dB; 1-sample delay ≈ −3 dB broadband (sub-sample
  alignment is mandatory — audit-v3's ×8 refinement is the published best
  practice); **AAC/MP3 round trip ≈ −10…−30 dB in-band** (masking-shaped —
  this is exactly why Congratulations caps at −10.6 dB) plus deterministic
  528–2112-sample framing offsets; ≥1 ppm clock/rate error kills minute-scale
  nulls, 1-cent varispeed (578 ppm) holds only ±27 ms; phase-vocoder pitch
  shift has **no coherence at any offset** (the KEYLOCK spectral path is the
  only detector for those); **neural separation floor = −SDR ≈ −10…−12 dB**
  best-case (BS-RoFormer 9.8–12 dB SDR) — a separated stem can never null
  much past that, matching the −6.7 dB observed on the Emily instrumental.
  Within-file fiber pairs are the *friendly* case: same lineage, same codec
  timeline, zero clock drift — the only caps are codec noise and separation.
- **Forensics (copy-move forgery detection)**: entirely speech, entirely
  feature-level — no published method uses sample-level waveform matching even
  as a baseline. Best transferable pipeline (Yang et al. 2023,
  arXiv:2302.07584): Shazam-style landmark constellation → linear-time hash
  dedup → DTW confirmation; survives 32 kbps MP3 / 6 kbps Opus / resampling /
  filtering at ~98–100% recall. Structurally identical to our
  `fp_probe`+fibers stack — independent validation of the design.
- **Graded middle ground (the real "make them cancel" upgrade)**:
  magnitude-squared coherence (per-band, "is this a linearly-filtered version
  of the same recording?") and **adaptive-filter nulling** (fit the unknown
  EQ with NLMS, then subtract — succeeds under filter sweeps/EQ automation
  where the plain polarity-flip null fails). Both unpublished as music-dedup
  tools. External calibration (Microsoft RARE, MSR-TR-2004-19): exact
  duplicates < 0.026 (95th pct), *different mix of the same song* ≈ 0.099,
  cutoff 0.14 — same-lineage-processed vs different-master separates on a
  graded scale.
- **Novelty check**: no published system disambiguates repeat instances inside
  highly repetitive content — Sonnleitner ISMIR 2016 (Qfp on DJ mixes, the
  SOTA) explicitly declines to evaluate within-track position there; Panako's
  documented failure mode IS repetitive material. The only within-track
  fingerprint-identity precedents are Burges 2005 (thumbnailing) and
  Ogle & Ellis 2007 (personal audio). The fiber program is open territory,
  and audit-v3's clone-rate numbers are novel data.

### Where it fits fibers (proposal, gated on the fiber-GT program)

A **clone tier inside fibers**, not a new probe channel (sensor phase stays
closed): `gt_als` render + the fiber-GT importer annotate each within-class
instance pair with `(r, residual_db, mag_r, drift_ms)` →
`clone | keylock | near-repeat | distinct`. Uses:

1. **GT determination** — clone-certified pairs need no human audition
   (John only auditions non-certified pairs; shrinks the Ableton pass).
2. **Decode credit** — within-clone picks are *provably* free (no content
   evidence can distinguish); within-fiber-but-distinct picks stay penalized.
   This is the honest version of fiber-aware credit.
3. **Calibration anchor** — clone pairs are P=1 by construction; they anchor
   the μ-calibration curve without costing GT labels.
4. **Graded residual as a feature** — per-pair residual/mag-r/drift feed the
   learned instance selector (trajectory lane) as deterministic evidence.

The one technique worth adding beyond what `selfsim` has: **adaptive-null
(NLMS) residual** as a graded lineage score between the binary CLONE verdict
and DISTINCT — it cancels through the EQ/filter-sweep automation that
producers put on duplicated sections (the modal "duplicate-then-tweak" case
the plain null misses). Fit a short FIR per window, residual-dB is the score.

Costs to name: verifier is per-pair O(n log n) xcorr (fine at fiber counts;
not a corpus-wide detector), thresholds are codec/lane-dependent (needs the
per-lane calibration above), and low clone-rate on vocal takes means the tier
mostly pays off on instrumental beds + BV stacks.

## Medley SIC — successive cancellation for the pileup sections (John, 2026-07-10; NOTE ONLY, sensor phase closed)

The cocktail-party sections (median-concurrency ≥4, now stratified in the
scorer; where identity whiffs concentrate) are NOT a blind-separation
problem. Classic ICA is mathematically out — it needs ≥ as many channels as
sources and we have a stereo fold of 5+ layers; the field's answer to blind
underdetermined separation is learned models, which we already run
(Roformer, ~−10 dB floor). But our case is INFORMED: we know the candidate
songs and align them to ~seconds (fp to ~0.2 s). So: identify the loudest
layer → align → adaptive-null it out of the mix (NLMS through the EQ, the
same machinery as the clone tier) → identify the next voice in the residual
→ repeat. Successive interference cancellation, reusing references we own.
In-repo priors that say it works: `cancel.py` (mix − acappella), the Disco
Lines live-set result (fingerprint residual after removing known tracks
exposed the unreleased reworks). Distinct from the CLOSED vocal-enhance and
overlay-pop dead ends (those were enhancement-for-ASR / energy detection,
not reference-informed cancellation). When opened: evaluate on the med≥4
bucket only — that's the failure mass it targets.
