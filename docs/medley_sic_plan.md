# Medley SIC — informed successive cancellation for pileup sections

**Opened by John 2026-07-10** (supersedes the sensor-freeze for this lane).
Target: the med≥4 overlay bucket (scorer-stratified 2026-07-09) — BB12 13
spans @37% fiber-aware / BB11 13 @21%, where identity whiffs concentrate.
The cocktail-party problem, but INFORMED: we know the candidate songs and
align them to seconds (fp to ~0.2 s), so blind separation (ICA — needs
channels ≥ sources; we have 2 vs 5+) is the wrong frame. Peel the onion:
cancel the layers we already trust, re-identify on the residual, repeat.

## Phases + gates

**Phase 0 — oracle feasibility (execute immediately, hours).**
On BB12's worst block (finale 3456–3658 s, med=5): cancel the layers the
aligner already gets right, using GT alignment (oracle isolates "does
cancellation bite on DJ-processed audio" from "is our alignment good
enough"). Warp each known ref onto the mix timeline (GT linear map),
refine per-window offset by xcorr, fit per-window least-squares gains
JOINTLY across layers, subtract. Measure:
1. residual energy drop per cancelled layer (dB) — the physics gate;
2. matched-filter salience of each MISSED layer against mix vs residual —
   does the quiet guest become audible to the machine?
3. residual wavs for John's ears.
GATE: ≥3 dB total reduction AND missed-layer salience improves, else CLOSE
(cheap death, note in attic ledger). Known physics risk: DJ master-bus
limiting is time-varying gain — per-window fitting is the mitigation;
Roformer-floor logic does NOT apply (we cancel with the true ref, not a
separated stem).

**Phase 1 — self-informed SIC (days).** Replace GT with the aligner's own
committed spans (identity_hit + set_start_err < 2 s from the span table).
Pre-registered eval: med≥4 bucket ONLY, both sets; metrics = identity hits
+ placement in-bucket. No cherry-picking outside the bucket.

**Phase 2 — wire as an agentic action (gated on Phase 1).**
"cancel-and-relisten" as an escalation move in the agentic loop for
pileup-flagged windows; the residual re-identification feeds the existing
fp/MERT probes. Opt-in flag until it wins on the race board's noPileF% vs
headF% gap.

## Phase 0 RESULTS (2026-07-10, BB12 finale 3440–3670 s, oracle GT alignment)

1. **Waveform cancellation: dead on arrival, as literature predicted.**
   Cancel-layer waveform salience 0.01–0.04 even at 0.6 s placement truth —
   Ableton warps are phase-vocoder (keylock), no waveform coherence exists.
   Any future SIC work MUST be spectral-domain.
2. **Spectral cancellation: physics gate PASSED.** Magnitude-domain adaptive
   subtraction (cancel.py's design) removes 25–43% of remaining energy per
   layer, total −2.2 dB (soft/Wiener cap 1.2) to −4.0 dB (aggressive cap 4)
   on a 5-deep block. The onion peels.
3. **Re-identification gate: NOT passed as measured — but the METRIC is
   suspect.** Missed-layer magnitude-cosine salience degraded/flat after
   cancellation. Diagnosis: broadband pop stems share spectral envelopes, so
   ANY energy removal lowers cosine similarity to everything; the metric is
   not layer-specific. Specified next step before a close/continue call:
   re-identification via **fp landmark hits on the residual** (sparse,
   content-specific) instead of magnitude cosine. Not run yet.
4. **Bonus bug-lead:** Honest (Virtu) has salience 0.74–0.76 in the RAW mix
   — the loudest thing in the block, yet the aligner misses it by 123 s.
   That's not cocktail-party masking; suspect decode/placement bug. Added to
   the worst-spans listening list (it's already rank-high there).

**FINAL VERDICT (2026-07-10, fp-landmark deciding probe): CLOSED.**
The premise dissolved under measurement. fp votes mix→residual: flat/noise
for every missed layer. Decomposition of the med≥4 failures:
1. **fp-visible layers mis-placed by DECISION LOGIC** — Honest SAVI/Virtu
   carry 1,427/2,769 votes in the RAW mix (massively identifiable, zero
   masking) yet land 42 s/123 s off. Cancellation is irrelevant; the bug is
   in how placement consumes fp evidence on these spans. TOP bug lead for
   the worst-spans listening pass.
2. **fp-INVISIBLE layers stay invisible after cancellation** — keylock
   warping breaks landmark geometry itself (control: Macy Gray, 43% of the
   block's energy, only 9 votes in the raw mix). The lever for these is
   warp/pitch-tolerant hashing (Panako-triplet/Qfp-quad ratio-invariant
   hashes — already noted in looptrace/NOTES.md), NOT separation.
Between those two classes, SIC adds nothing measurable. Spectral
cancellation itself works (physics gate passed) and remains available as an
operator (clone tier, cancel.py); it is the *SIC-for-identification* loop
that is closed. Probe preserved: `attic/sic_phase0_probe.py`; wavs in
`~/aligning/_review/_sic/`.

## Prior art in-repo (why this isn't a re-run of closed threads)

- `cancel.py` (mix − acappella) and the clone tier's adaptive-null: the
  cancellation operator exists.
- Disco Lines live-set: fingerprint residual after removing known tracks
  exposed the unreleased reworks — SIC's identity step, already observed.
- CLOSED and different: vocal-enhance (enhancement-for-ASR), overlay-pop
  (energy detection) — neither used reference-informed cancellation.
