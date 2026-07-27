# DJ Agent — Design Spec (SHELVED)

**Status: SHELVED 2026-07-10 — do not build yet.** Captured for later. Nothing in
this doc is scheduled; it exists so the brainstorm, research, and decisions
survive until picked back up.

## Vision

A natural-language DJ agent: the user tells it songs, mashup instructions, and
constraints ("lay the *Freed From Desire* acappella over the *Turn It Up*
instrumental at 128, drop at 1:30"), and it produces **an editable Ableton Live
set (.als) plus rendered audio**. Not a coding agent — a DJ agent, built on the
tracklist_engine scaffolding. Potential startup: sell the tool (user brings
their own audio library), not the content.

## Decisions locked during brainstorm (2026-07-09/10)

1. **Compiler, not collaborator.** The user dictates musical intent; the agent
   resolves songs → stems, computes placement/stretch/keys, places clips,
   emits .als + audio. Taste stays with the user. Deterministic, testable.
2. **Single mashup per request** (2–4 songs, one musical idea, ~1–5 min). A
   full set is a later sequence of these.
3. **Linear-warp-only for v1.** No non-linear WarpMarker nudging. Concretely:
   - **Placement** (phrase-aligned drop of vocal onto instrumental grid) is
     required regardless of warp policy — always in scope.
   - **Single global linear stretch** per clip (two WarpMarkers, the derived
     ratio mix_bpm/acap_bpm) is cheap and stays in scope.
   - **Compatibility gate** replaces hard warping: only accept pairs with
     |stretch ratio − 1| ≤ ~6% for vocals (looser for instrumentals), key
     compatible within ±1 semitone transposition. Out-of-tolerance requests
     get near-BPM alternatives proposed from the corpus, not a warbly render.
     BPM + key already computed corpus-wide (Essentia, 16k tracks).

## Existing assets this stands on

| Asset | Role |
|---|---|
| `labeling/als/` codec (parse∘print=id) | .als emission — the product wedge |
| `alignment/synthetic_mix/render_v2.py` | structure→audio renderer (adapt for mashup render) |
| ingest + Roformer stems + `candidate_vocal_gate` | song name → audio → stems |
| analysis (beat_this grids, Essentia BPM/key, sections, cue-detr) | placement + compatibility inputs |
| `warp_prior.json` (n=316) + BB12 GT taxonomy | priors for gain edges, stretch sanity |
| micro-pitch detune estimator | detune correction in the audio-quality lane |
| GT schema (`labeling/ground_truth/schema.py`) | shared IR between codec and renderer |

## Architecture (v1 compiler)

NL request → **intent parse** (songs, roles acap/instr/bed, target BPM, drop
placement) → **resolution** (corpus lookup → track_audio → stems; identity axes
respected) → **compatibility gate** (BPM ratio + key) → **placement** (snap
vocal phrase starts to instrumental phrase grid; downbeat-accurate) → **linear
stretch + transposition** → **emit**: (a) .als via codec — clips placed, warped
(2 markers), gain-curved from prior; (b) rendered audio via render_v2-style
backend. Both from the same GT-schema timeline.

### Output backends (note: Ableton-MCP-style)

Two complementary emission modes; v1 = (a), (b) is a later option:

- **(a) Offline .als compilation** (the codec). Works headless, no Live
  running, versionable artifact.
- **(b) Ableton-MCP-style live session control.** AbletonMCP-type servers
  (MCP → Live remote scripts) puppet a *running* Live instance — create
  tracks, drop clips, set warp/tempo interactively. Could make the DJ agent
  conversational *inside* Live ("nudge that drop 2 bars later" applied to the
  open session). Unverified diligence item: no known MCP agent compiles .als
  offline or does mashup math — confirm before positioning (see Diligence).
  Also a candidate integration surface rather than a competitor: our compiler
  as the brain, an MCP bridge as the hands.

## Parallel plan (lanes)

**V1 (active when unshelved):**
- **Lane C — compiler v0**: rules + prior, as architected above.
- **Lane D — reconstruction eval**: given GT ingredients + intent, does the
  compiler reproduce the human's choices? Metrics: placement error (s),
  stretch-ratio error; blind A/B renders vs human .als. Restrict to
  near-linear GT spans. Extends `harness/` + `make scorecard`.
- **Lane E — audio quality**: stretch algorithm choice for vocals (linear case
  only), formants, detune correction, transposition policy. Independent.

**Background / later ledger:**
- **Lane A — GT scaling** (John labels more sets; optional under linear-only
  v1, binding again if warp SOTA resumes). BB10 + Murph are pulled, unlabeled.
- **Lane B — warp-grammar mining** (nudge residuals = the hand-craft; nobody
  else has WarpMarker-level GT) and **Lane F — learned warp model**: DEFERRED.
  This is the phase-2 research moat, not v1 scope. Synthetic renders can
  pretrain but can't teach above the prior — only human GT reveals residual
  grammar.
- **Lane G — hardware / real-time**:
  - G1: FLX4 "agent-as-software" — board is class-compliant USB-MIDI + audio
    interface; agent receives MIDI, emits audio, drives LEDs. Mixxx 2.4
    mapping = Rosetta stone (incl. SysEx keepalive that dumps fader state).
    Shared-control demo: agent performs, human grabs a fader anytime.
  - G2: real-time executor of compiler timelines (same core, live backend).
  - G3: club rig — (a) Pro DJ Link listener peer (Deep Symmetry beat-link /
    dysentery: read BPM/beat/position/on-air from CDJ-3000s + DJM-V10 — also
    free live alignment telemetry); (b) prep agent writing rekordbox-format
    USB (crate-digger/rekordcrate); (c) performer as line-input deck
    beat-matching against human CDJs.
  - NOTE: hardware research claims came from strong primary sources (Mixxx
    repo/manual, dysentery) but adversarial verification failed on rate
    limits — re-verify load-bearing claims before building G1.

## Prior-art findings (deep research, verified 2026-07-09)

Verdict: **partially done; the combination is whitespace.** No verified system
combines even two of {NL interface, automatic stem-level mashup compilation,
editable DAW-project output}.

- AutoMashUpper (ISMIR'13/TASLP'14): automatic key/tempo-matched mashups, whole
  songs, sliders, .wav out. Lee et al. (ISMIR'15): vocal-over-instrumental
  multi-segment. Huang (AAAI-21): learned stem compatibility. Stem-JEPA
  (ISMIR'24): stem-fit retrieval embedding only.
- **AutoMashup (July 2025)** — closest: Demucs→Allin1→pyrubberband
  acappella-over-instrumental; coarse segment conformance, Streamlit, no DAW
  out; weak paper (failed core experiment).
- **DJ.Studio** — nearest commercial neighbor: ships stem-separated **.als
  export**, but compiles a user-arranged timeline; no NL, no auto-mashup.
  ⇒ .als emission alone is not novel; NL-driven automatic compilation to .als is.
- **Warp quality is an open research problem** (DTW baseline; André'24 NMF only
  "promising"; all generators use off-the-shelf rubberband). Deferring warp
  SOTA costs nothing competitively; owning it later is the moat.
- DJtransGAN / DJ-AI: whole-track transitions only, audio-in/audio-out.

## Risks / constraints

- **Licensing (the elephant):** mashups of commercial recordings are
  derivative works; research-corpus acquisition (rips) is not shippable.
  Product shape: BYO-library tool (Serato/rekordbox/Mixed In Key model).
- Single-annotator GT conflates "the grammar" with John's style (fine for a
  taste-model product; flag before any SOTA claim).
- Prior-art scan goes stale in months (AI-music space velocity).

## Pre-launch diligence (open, from research)

1. **LLM / Ableton-MCP lane**: do any text-to-DAW agents generate .als
   programmatically? Zero surviving claims either direction — biggest hole.
2. Consumer-tool teardown (RaveDJ, Mixed In Key Mashup, Neural Mix, Serato
   Stems, Moises): what they automate, where vocal warp breaks. Unverified.
3. **Patent scan** (Serato, Algoriddim, NI, DJ.Studio) — none performed.
4. DJ.Studio hands-on: can automix + .als export be chained to approximate
   this today (missing product vs missing integration)?
5. Re-verify hardware claims (Mixxx FLX4 mapping file is public source — read
   it directly).

## Definition of SOTA (when warp lanes resume)

No public benchmark exists for generative mashup warp quality (published SOTA
is alignment-side: André 2024 / UnmixDB). The bar is constructed: (a)
reconstructs human GT warps within tolerance, (b) beats AutoMashUpper-class +
consumer tools in blind ear tests, (c) degrades gracefully to editable .als.
Defining the benchmark is itself a moat.
