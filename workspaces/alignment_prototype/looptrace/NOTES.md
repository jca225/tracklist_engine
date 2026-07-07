# looptrace — NOTES

Running log for the acappella loop-tracing effort (non-monotonic alignment of DJ-mix
vocals). Plan of record: the phased brief (Phase 0 recon → Phase 1 ill-posedness audit
→ Phase 2 loop-collapse → Phase 3 landmark Hough + segment-cover DP → Phases 4–5
conditional patches). Context doc: [docs/loop_tracing_research_brief.md](../../../docs/loop_tracing_research_brief.md).

Module home: `workspaces/alignment_prototype/looptrace/` — inside the existing
alignment workspace so it can import the legacy machinery (`ref_fibers`,
`path_decode.trajectory_acc`, `landmark_fp`) without path gymnastics, while staying a
separate module the legacy decoder never imports. Runnable side-by-side.

---

## Phase 0 — repo recon (2026-07-06)

### The existing decoder(s)

- **`workspaces/alignment_prototype/path_decode.py`** — the legacy segment decoder this
  effort must beat. Viterbi over ref-offset states (`_viterbi`, lines ~99–149; stay-on-
  diagonal free, jump costs `lam`, `lam_back` optional — the asymmetric sweep HURT,
  don't revisit). Emissions = windowed normalized matched filter
  (`_scores_at_stretch`, ~152–178; 12 s window, 2 s hop) on chroma or HuBERT
  (`--feature`, acappella routes to HuBERT layer 9). `decode_path` (~181–253) returns
  the segment list. **`--eval` slices the mix at GT `set_start_s` — the CLI eval IS the
  oracle-placement baseline** from the research brief.
- **`joint_ref_decode.py`** — applies `decode_path` post-infer, feature-routed
  (acappella→HuBERT, else chroma), writes `ref_segments` into the predicted timeline
  JSON (real-placement numbers).
- Matched-filter scalar ref-offset: `refine_ref_offsets.py` (single-window, coin-flips
  repeats — known-failed approach #1). Continuity stack: `continuity_refine.py`
  (no-op for acappella — 93% of acappella spans non-linear).

### Existing landmark fingerprinting (REUSE for Phase 3)

- **`landmark_fp.py`** — Shazam-style constellation already implemented:
  `constellation()` (STFT NFFT=2048, FHOP=512, peak_size=19, 60 dB floor),
  `hashes()` (peak-pair `(f1, f2, dt)` keys, fan-out 8, dt_max 80 frames),
  `fp_offset()` (offset vote histogram over a stretch band → offset, votes, stretch,
  sharpness), `LandmarkFingerprint.from_blob()` (cached blobs in DB
  `track_fingerprints`, kind=landmark).
- **`mix_fp_hits.py`** — mix-side scanning: `span_from_offset_votes` (vote-extent →
  set_start), `offset_candidates` (top-K diagonals). Thresholds: HIT_MIN_VOTES=25,
  HIT_MIN_SHARPNESS=1.2, z<1.0 band gate.
- So Phase 3's "point cloud" step is a **modification**, not a from-scratch build: the
  existing pipeline collapses matches into an offset *histogram* per stretch; we need
  to keep the raw `(t_mix, t_song)` match points, transform to intercepts at the known
  slope, and do 1D Hough + segment-cover DP over them. Known caveat from the corpus:
  the full-mix fp is **weak on vocals** — Phase 3 must hash `mix_vocals.flac` against
  the ref *vocal stem* (axis routing per `harness/axes.py`), and fixtures must test
  repitched (tempo+pitch) vs key-locked stretch, since ~31% of BB11 acappellas are
  re-pitched (breaks fixed-frequency hashes → need coarse freq quantization or
  Panako-style ratio-invariant hashes; decide on fixtures).

### Fiber machinery (REUSE for Phase 1)

- **`ref_fibers.py`** — self-repeat equivalence classes over the ref:
  `compute_fibers` / `compute_fibers_soft` (HuBERT + RMS silence-gate
  (ratio 0.35) + `_long_repeats` diagonal scan (min 6 s, thresh 0.5) +
  `_avg_linkage` grouping (thresh 0.5)). Output: per-frame labels @ ~8 Hz,
  membership μ, per-fiber confidence. **Fibers are HuBERT-based, never chroma**
  (chroma blobs the whole track into one fake fiber).
- Phase 1 extends this: fibers say *which regions repeat*; the audit adds
  **sample-accurate cross-correlation within each fiber** to split CLONE
  (production copy-paste, residual at noise floor) from DISTINCT TAKE.

### Eval harness, test mixes, answer keys

- **Frozen legacy metric:** `path_decode.trajectory_acc` (lines ~293–325) — sample
  mix-times at 1 s steps across the span, predicted ref(mix_t) via piecewise-linear
  interpolation of the segment list, correct if within **2.0 s** of GT ref(mix_t);
  span score = fraction correct. Fiber-aware variant additionally credits a prediction
  landing in the same fiber as GT. Span classes via `_span_class` (~328–336):
  `loop` (is_loop) / `multiseg` (has ref_segments) / `oddratio` (tempo_ratio outside
  0.9–1.15) / `linear`. **This metric is frozen**; all phases also report the brief's
  per-second accuracy at ±0.25 s and ±1.0 s (stricter tolerances, same sampling
  scheme) as the new metric.
- **Answer keys:** `labeling/fixtures/bb12_ground_truth.yaml` (set `1fsnxchk`,
  ~/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/) and
  `labeling/fixtures/bb11_ground_truth.yaml` (set `2nvzlh2k`, Episode 11 folder).
  Schema: per span `set_start_s/set_end_s/ref_start_s/ref_end_s/tempo_ratio/
  pitch_shift_semi/is_loop/ref_segments[{mix_start_s, ref_start_s, ref_end_s}]/
  gain_curve/claimed_stem`. GT is FINAL for both sets (150 rows); never tune on it —
  thresholds tune on synthetic fixtures only (§9 of the brief).
- **End-to-end scorer (real placement):** `score_timeline_vs_gt.py --fibers`.
- **Reproduction command (oracle placement, the baseline this task quotes):**

  ```bash
  venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode --eval \
      --feature hubert --stems acappella --fibers --workers 8 \
      [--gt labeling/fixtures/bb11_ground_truth.yaml]
  ```

### Audio I/O, sample rates, tempo ratio

- Feature SR=22050, HOP=512 → FPS≈43.066 (path_decode); landmark fp uses NFFT=2048,
  FHOP=512. Mix audio: `~/aligning/<set>/mix_vocals.flac` (Roformer vocal stem of the
  mix). Ref audio: `manifest.json` rows → `local_path` + `stems.vocals`.
  HuBERT/chroma features cached at `workspaces/alignment_prototype/.feat_cache/`
  (666 files present).
- **Tempo ratio:** GT carries `tempo_ratio` per span; at decode time the stretch band
  comes from the **beat grids** (`_stretch_band`: mix `set_measures` bar length around
  the span vs ref `track_measures` bar length via `mert_store.load_bb12_mert`),
  octave-folded to [0.7, 1.45] × fine grid (0.96–1.04). So slope for Phase 3 is
  available without touching GT: beat-grid ratio, refined per span. GT `tempo_ratio`
  is used only for scoring/fixtures.
- `pitch_shift_semi` in GT flags re-pitched spans (repitch vs key-locked matters for
  landmark hashing).

### Baseline reproduction (Phase 0 gate)

Command above, run 2026-07-06 on this machine (log:
`looptrace/baseline_repro_hubert.log`). Oracle placement (GT set_start), HuBERT L9,
lam=0.15, traj-acc <2 s — **matches the brief exactly; GATE MET**:

| class | BB11 (2nvzlh2k) | BB12 (1fsnxchk) | fiber-aware BB11/BB12 |
|---|---|---|---|
| ALL acappella | 35% (n=17) | 44% (n=21) | 35% / 47% |
| linear | 43% (n=5) | 62% (n=8) | 43% / 62% |
| **multiseg** | **12% (n=7)** | **27% (n=7)** | 12% / 31% |
| loop | 0% (n=1) | 81% (n=1) | 0% / 100% |
| oddratio | 75% (n=4) | 32% (n=5) | 75% / 34% |

These are the **frozen baseline** numbers all later phases compare against (same
command, same tolerances). Note one BB12 span is skipped for no-audio (n=21 of 22).
Loop class is n=1 per set — never read anything into it alone.

### Known-failed approaches (do NOT re-walk)

1. Single-window matched filter (`refine_ref_offsets`) — coin-flips repeats.
2. Viterbi over windowed matched-filter emissions (`path_decode`) — noise decides
   among repeats.
3. Fiber-aware *scoring* — measures ambiguity, doesn't resolve it (machinery reused).
4. Backward-jump penalty (`lam_back` sweep) — hurt.
5. HuBERT features for localization — too self-similar within a track.
6. Seeding decode with GT start (soft ref_start prior, 2026-07-06) — flat.

## Phase 1 — ill-posedness audit (2026-07-06)

**Built:** `selfsim.py` (whitened-mel lag-diagonal repeat detection +
sample-accurate waveform verification with sub-sample refinement),
`audit.py` (CLONE/DISTINCT classification, clone equivalence lag-maps,
GT-second decomposition), `eval.py` (per-second accuracy at ±0.25/1/2 s under
strict / clone-aware / repeat-aware equivalence), `rules.clone_tiebreak`,
`config.py` (audit-v3 / eval-v1), 7 fixture tests. Additive `--dump` flag on
`path_decode` to export per-span predictions. Full write-up: `AUDIT.md`;
numbers table: `RESULTS.md`.

**Numbers:** clone-unwinnable ≈ **0%** of GT seconds on multiseg (both
sets); distinct-take repeats 51–55%, unique 45–49%. Baseline re-scored:
clone-aware ≡ strict (the decoder never lands on clone-mates); repeat-aware
lifts multiseg 27→45% (BB12), 12→34% (BB11).

**Surprises:**
1. The reusable machinery both failed and taught us why: HuBERT fibers
   under-detect (audit-v1); per-frame silence gating fragments sparse vocals
   at every breath (audit-v2 — median voiced run 1.6 s); raw mel cosine
   saturates on stable spectral envelopes; fractional-sample clones read as
   DISTINCT without sub-sample xcorr refinement. All four are encoded as
   fixture tests now.
2. **The ill-posedness bound does not bind.** Almost nothing is clone-tied;
   the brief's ~35–47% "ceiling" was decoder-limited. The honest ceiling is
   ~100%; multiseg failure ≈ half wrong-instance (addressable with
   instance-sensitive/long-range evidence) + half wrong-content (exactly
   Phase 3's target).

**Go/no-go → GO to Phase 2/3.** Phase 4's discriminative-frame weighting is
*expected* to be needed (51–55% of GT seconds are distinct-take repeats and
per-frame phonetics can't split them), but Phases 2+3 come first per the
leverage ordering.

## Phase 2 — loop-collapse preprocessing (2026-07-06)

**Built:** `fixtures.py` (synthetic pseudo-vocal song + fake mix spans:
straight / 2×/4× loops / section replays / jumps / natural distinct-take
repeats, × none/repitch/key-locked ±8%, × clean/−6 dB per-iteration bleed),
`loops.py` (`detect_loops` → `Loop(source, period, n_iter)`, image-based
`collapse`/`CollapsedSpan` time maps, `expand_segments` with the same-source
constraint, CLI gate runner), `LoopConfig` loop-v2, 6 loop tests (13 total).

**Fixture gate: 100% precision / 100% recall** on the battery (≥95%
required), incl. −6 dB bleed. One strict xfail documents the known gap:
tile-then-key-locked-vocoder (live-hardware order) defeats waveform
cloneness AND spectral-magnitude+drift verification (measured
indistinguishable from distinct takes) — irrelevant for the Ableton-produced
GT corpora (duplicated warped clips = copies AFTER the stretch).

**Design deviation from the brief (measured, justified):** "DJ loop =
bit-near-identical" fails on real mix_vocals — Roformer separation under
different beds degrades a CONFIRMED real loop's iterations to −6.7 dB, into
the distinct-take band. The discriminator is instead **structural**: a mix
self-repeat at a lag whose song-time image is absent from the song's
Phase-1 repeat map must be a DJ edit; matches of song structure are
rejected (played straight). The residual floor (−3 dB) only rejects noise.
Also: GT "loops" are section-scale (periods 15–109 s, incl. non-contiguous
replays), not just 1–8-bar button loops — Loop generalizes to
source-region + images; max_lag 120 s.

**Real-mix gate (answer-key spot-check):**
- BB12: 3/3 GT-loop spans detected, periods exact (41.17 / 14.95 / 15.24 s
  vs GT 41.17 / 15.02 / 15.3); 1 extra on slot 071 = natural chorus repeat
  straddling a DJ jump (the structural test assumes straight play between
  occurrences — measured blind spot, small 2.5 s source).
- BB11: 2/2 GT-loop spans hit at span level; period-level disagreements
  from three understood causes: song-side repeat-map coverage gaps
  (Seasons of Love cov 21% → its internal repeat reads as a DJ edit),
  overlapping w-layer spans sharing mix audio (twin 65.4 s detections in
  039w2/039w3), and a 109 s replay's internal structure outranking the
  outer period.

**Go decision → Phase 3**, with one carry-over: detected loops enter the
segment-cover DP as STRONG SOFT constraints (bonus for same-intercept
tiling across iterations) arbitrated against landmark evidence — not
inviolable hard constraints; real-mix precision doesn't justify hard.
Collapse/expand machinery stays available for the unambiguous short-lag
button-loop case.

## Phase 3 — decoder (2026-07-06, in progress)

**Built:** `landmarks.py` (pitch/tempo-tolerant keys: coarse log-f1 +
pitch-invariant log-ratio + log-dt; raw point cloud; ref cache),
`segments.py` (inlier-median-refined intercept Hough; cover DP over
diagonal states + NULL with per-time median floor-subtraction + column
normalization; no monotonicity / backward penalty), `run.py` (collapse →
hash → match → slope-by-support → DP → expand). Fixture battery ≥90%
per-second (±0.25 s) across straight/jump/loop2/loop4/replay ×
none/repitch/keylock.

**§9 change log (post-real-mix changes, everything rerun):**
1. `pair_cap` 1024→64 (fixture-tuned): huge-key noise (~300 pts/bin)
   buried weak keylock diagonals on fixtures.
2. Slope objective changed from candidate-vote SUM to histogram PEAKINESS
   (top-3 bins above median background) after the first real-mix run
   picked octave-folded wrong slopes (0.53/0.39/1.37) on 12 of 38 spans —
   noise fills bins at every slope, so summing many bins is degenerate;
   fixtures re-validated ≥90% after the change.
3. Slope candidates = beat-grid band ∪ canonical near-1 band — the grid's
   bar ratio is wrong on some refs (slot 073: true-slope peakiness 856 vs
   the grid band's best 292). Fixtures unaffected.

**Phase 3 gate (real mixes, honest verdict): PARTIAL.** Full table in
RESULTS.md. Beats the frozen baseline decisively on BB11 (ALL 38 vs 35;
multiseg 22 vs 12 = +83% relative — a "clear margin" on the target class;
loop 24 vs 0; linear 60 vs 43) and on BB12 loops (94 vs 81), ties BB12
multiseg (26 vs 27), loses BB12 linear/oddratio (44 vs 62 / 26 vs 32).
The ≥70%-of-addressable sanity target is NOT met (addressable ≈ 100%).

**Surprises / diagnosis:**
- The decoders are complementary: looptrace finds the right CONTENT far
  more often on BB11 (repeat-aware ALL 60 vs 47; oddratio 93!) but lands
  on the wrong repeat INSTANCE — the strict↔repeat-aware gap is now the
  dominant recoverable error, exactly Phase 4's target (discriminative
  frame weighting + the clone tie-break as vote priors).
- Slope selection is the other lever: 14/21 BB12 slopes correct; 4
  weak-evidence spans lose to noise peaks by small margins. Next idea:
  decode the top-2 slopes fully and pick by DP path score, not histogram
  peakiness (the DP sees collinearity the histogram can't).
- Landmark evidence on Roformer-separated vocals is noise-dominated
  (30–70k points/span, true diagonals ~hundreds of inliers) — the mel/
  HuBERT matched filter and the landmark cloud fail on DIFFERENT spans,
  so a per-span router (peakiness margin → looptrace | legacy) would
  already beat both overall; wire after Phase 4.

**Go/no-go: GO to Phase 4** (the audit predicted this: 51–55% of GT
seconds are distinct-take repeats and per-frame phonetics can't split
them; the repeat-aware headroom is +14–50 pp depending on class).

## Phase 4 — discriminative frame weighting (2026-07-06): NEGATIVE, default OFF

**Built:** `discrim.py` — per-frame discriminability mask over the ref
(whitened-mel distance between aligned repeat images, ±1-frame alignment
slack for non-integer lags; cached) + post-decode instance re-selection
(discriminability-weighted landmark support per repeat image, 25% switch
margin, incumbent wins ties). 3 unit tests (mask ≈0 on clones / 1 on
unique; switch-when-supported; hold-on-tie). Wired as `run.py --discrim`.

**Gate: FAIL — regresses on both sets** (BB12 ALL 36→28, linear 44→23;
BB11 ALL 38→35, multiseg 22→17). Diagnosis: on noise-dominated real
clouds the weighted-support comparison is unreliable — image regions
OUTSIDE the audit's repeat coverage carry full mask weight, so a wrong
image can win on weighted noise; the mask only discounts frames the audit
explicitly covered. Per the gate rule (no regression elsewhere) the
default is OFF; machinery retained behind the flag. NOT tuned further —
that would be tuning on the answer keys.

**Phase 5 not entered.** Its precondition holds (instance ties remain),
but the measured Phase-4 failure changes the right design: the arbiter
needs *calibrated* evidence (waveform residual at discriminative frames,
Phase 5's tool) INSIDE the DP rather than post-hoc support re-counting.
Next-session levers, in expected-value order:
1. per-span looptrace|legacy ROUTER (the decoders are complementary;
   route by Hough peakiness margin) — cheapest real win,
2. DP-path-score-based slope pick for the 4 weak-evidence spans,
3. Phase-5 waveform-residual tiebreak as a DP term (not post-hoc),
   restricted to audit-covered discriminative frames.

## Post-phase levers (2026-07-06, second pass)

**Lever 2 — DP-evidence slope selection: WIN.** Top-3 peakiness slopes
compete on ±0.3 s path-inlier evidence with an MDL per-segment charge.
Two fixture-caught bugs in the evidence measure on the way (cross-
candidate median floor self-annihilates on all-true candidate sets →
random-probe noise floor; loose tolerance favors wrong slopes → tight
currency). Result: BB12 multiseg 26→35 (baseline 27), BB12 ALL 36→43;
BB11 ALL 38→44, oddratio 43→68. Both sets now beat baseline on multiseg.

**Lever 1 — router: NULL.** After lever 2 the decoders' complementarity
mostly evaporated (oracle-per-span 48/50 vs looptrace-only 43/44) and
`evidence_rate` does not transfer LOSO (BB12's theta hurts BB11 34 vs 44).
`router.py` kept for the record; not wired.

**Lever 3 — residual tiebreak in the DP: inert as specified, mechanism
discovered.** Sound rival groups (2–4 images at audit lags) never occur —
Hough candidates are dense and lag-matching chains degenerately. Allowed
to act degenerately it becomes a global mel-verification prior with big
oddratio wins (BB11 68→93) but flips one correct linear span. Default =
strict grouping (inert). Follow-up design recorded in RESULTS.md: a
deliberate mel-consistency emission for ALL candidates (hybrid landmark +
matched-filter DP).

## Hybrid emission + pipeline wiring (2026-07-06, third pass)

**Hybrid mel emission (option 1): flat overall, default OFF.** Built as
the lever-3 follow-up: bounded (±0.12 cap) per-candidate mel contrast in
the DP. Fixtures pass with it ON; real mixes: BB11 oddratio 68→95 and
ALL +3, but small erosion elsewhere (BB12 −2 ALL, BB11 loop n=1 flip) —
combined 43.7 vs 43.4, flat. `run.py --hybrid` to enable. Refinement if
revisited: inverse-landmark-density weighting (fill deserts, don't argue
with dense landmark evidence).

**Pipeline wiring (option 2): DONE, real-placement gains, placement now
binds.** `joint_ref_decode --decoder looptrace [--audit ...] [--out ...]`
routes acappella spans through the landmark decoder; legacy for other
stems; `--out` avoids clobbering the in-place timeline. End-to-end
(fiber-aware): acappella 10→13 (BB12) / 12→13 (BB11), ≥80%-covered
2→7 / 4→7, headline +1 both sets, no regressions. The oracle-vs-real gap
(43–44% vs 13%) makes the next dependency explicit: **acappella set_start
placement** is the binding constraint end-to-end — decoder work past this
point pays off only after placement improves, or by letting looptrace
self-place (decode a widened mix window around the prior, which its
Hough geometry supports naturally).

## Self-placement (2026-07-06, fourth pass)

**Mechanics validated, gating signal missing.** `run.py --pad-s/--jitter-s`
+ `joint_ref_decode` adaptive retry. Controlled: 15 s placement error
collapses the tight decode to 6%; a ±45 s pad recovers it to 23% and makes
the decode placement-invariant — but costs well-placed spans 43→24, so
always-pad is wrong. The adaptive gate (retry when tight evidence_rate <
floor) failed to transfer: fixtures separate placed/misplaced at ~20, but
real separated vocals sit far lower — 29/33 spans "retried" and e2e
acappella went 13→12. Retry disabled (`retry_evidence_rate=0`); this is
the second time evidence_rate failed as a real-data quality signal (router
was the first) — a calibrated confidence is genuinely the missing piece,
which is the known C1 learned-arbiter work, gated on a third GT set.
Candidate cheap signal to try first: fraction of the padded decode's
path evidence lying OUTSIDE the tight window (directly measures "the
content wasn't where placement said").

**UPDATE (same day, fifth pass): the out-of-window evidence fraction
WORKS.** `ev_out_frac` (fraction of the padded decode's path-inlier
evidence outside the believed window) is a ratio within one decode and
transfers where absolute rates didn't. Controlled real-data validation
(jitter as the manipulated variable, no accuracy tuning): placed median
0.53 vs 15s-misplaced 0.82; gate θ=0.8 keeps placed spans at tight
quality (43%→43%) while rescuing misplaced ones (6%→17%),
plateau-robust θ 0.7–0.85. SHIPPED as the joint_ref_decode default:
e2e acappella combined 11.2% (legacy) → 13.0% (lt) → **14.5%
(lt + self-placement gate)**; BB11 12→16. NOTE the segment-TIME variant
of the same signal is uninformative (real padded windows tile fully with
weak segments — placed vs jittered measured IDENTICAL); the evidence
weighting is essential.

## BB12-linear gap forensics (2026-07-06, sixth pass): two negatives that close the lane

Per-span diagnosis of the 4 zero-scoring oracle linear spans:
- **047**: true slope 1.050 falls between the 3% grid points → true diagonal
  smeared below a noise rival. Local slope REFINEMENT by peakiness was
  built and REGRESSED (BB12 multiseg 35→24: the noise-prone objective gets
  27 chances/span to sharpen a wrong diagonal before path evidence sees
  it) → reverted (`slope_refine_fracs=(0.0,)`); untested alternative:
  refine by full path evidence (~27 decodes/span).
- **029/073**: the true diagonal has healthy landmark support (247/179
  inliers) but is OUTVOTED (405/800). Mel content verification was tested
  as the arbiter and prefers the WRONG diagonal on 94–97% of steps — both
  evidence channels agree on the "wrong" answer, i.e. these are genuinely
  instance-ambiguous (or the GT picks one of two equally-valid images).
  No content-evidence fix exists; this is the discriminative-frames /
  tie-break-prior territory that Phases 4–5 already measured as hard.
- **003**: 2× octave-fold slope error, still open.

Conclusion: the BB12-linear gap vs legacy (50 vs 62 at oracle) is mostly
NOT reachable by better geometry or content verification — it needs
instance-level evidence (Phase-5 waveform residual on audit-covered
frames, properly calibrated) or the learned arbiter, both gated on the
third GT set. The lane is closed until Murph lands.

## Lyrics-placement wiring (2026-07-06, seventh pass): NO GAIN on BB12 — and it corrected the narrative

Built `lyrics_placement_refine.py` (timeline-consuming variant of the
lyrics channel: identity from the timeline, acoustic set_start as the
position prior, leash 120 s). BB12: 35/41 acappella spans placed, but e2e
trajectory 10% vs 12–13% acoustic — no gain, slight harm. Root cause of
the null: **the acoustic placement was already good** — matched by
recording, BB12 acappella set_start is 4.2 s median / 73% <15 s; lyrics
re-placement measures 5.0 s median. The lyrics channel's 18x win was over
the OLD (pre-fp/stem-placement) pipeline.

**This falsifies the session's "placement is the binding e2e constraint"
narrative as stated.** The oracle(43%)↔e2e(~10%) gap survives on the same
clean population (9% vs 43%), and an ad-hoc coverage probe suggests
span-window/GT correspondence issues (many zero-overlap matches, raw-yaml
set_end_s semantics, w-layer matching) — but the ad-hoc joins themselves
are unreliable there. OPEN: decompose the oracle↔e2e gap using
score_timeline_vs_gt's own matching internals (identity match, span
extents, per-span pairing) before believing any single-cause story.
BB11 lyrics test pending its stem transcription (warm-up started,
~3 h MPS; cache at .cache/lyrics).

## Lyrics wiring, BB11 arm (2026-07-07): NULL confirmed cross-set — thread CLOSED

Full transcript coverage achieved via a Vast 3090 (42 stems, ~$0.05,
box destroyed; transcripts imported under Mac cache keys). With 63/71
BB11 acappella spans lyrics-placed (0 leash rejections), e2e trajectory
is 15% vs 16% gated-acoustic — same null as BB12. Conclusion: for the
looptrace stack, set_start REPLACEMENT is redundant on both sets — the
landmark decode's intercepts are absolute (window offset only crops
content) and the self-placement gate already catches gross misplacement.
The lyrics channel remains valuable where it was validated (placement
for the OLD pipeline / as an agentic probe / identity evidence), but it
is not an e2e lever here. The oracle(43-44)↔e2e(~13-16) gap therefore
lives in span/GT correspondence (extents, identity, w-layer matching) —
the queued decomposition with score_timeline_vs_gt's own matching is THE
next analytic step before any further decoder or placement work.

## Invisible-identity-losses thread (2026-07-07): CLOSED — causes named, all upstream

The 9 BB12 acappella GT recordings never matched by any timeline span
(--decompose finding) split into two upstream classes:
1. **5 inventory gaps**: online-candidate acappellas hand-added during
   labeling; tlp-namespace ids with no id-map entries and no manifest
   rows — the aligner cannot predict a recording absent from its
   candidate pool. Territory: ingest / stem-library matcher /
   acappella-selection gate.
2. **4 claimed_stem gaps**: slots EXIST on pi (028w2, 031w2, 029w1,
   039w1) but with claimed_stem='regular' despite acappella titles —
   tokenizer claimed_stem parsing — and infer's identity decode assigned
   those slots to other recordings. Territory: tokenizer +
   identity/inventory, NOT the ref-trace decoder.

BB11 has zero such losses. No looptrace action; filed for the
tokenizer/ingest side.

### Plan deltas vs the brief (repo-specific adjustments)

- `looptrace/` lives under `workspaces/alignment_prototype/`, not top-level (repo
  rule: experiments incubate in workspaces/; new top-level dirs need justification).
- Phase 3 landmarks: reuse/extend `landmark_fp.py` instead of a fresh ~200-line
  implementation — it's already the proven localizer (0.2 s median on regular spans);
  the changes are (a) keep raw match points instead of histogramming, (b) vocal-stem
  routing, (c) pitch-tolerant hashing decided on fixtures.
- Phase 1 equivalence classes: extend `ref_fibers` fibers with sample-accurate
  within-fiber cross-correlation rather than building self-alignment from scratch.
- Loop detection (Phase 2): no existing DJ-loop detector in the repo (the Viterbi
  *represents* backward jumps but never detects mix-side digital copies) — this is
  genuinely new code.
