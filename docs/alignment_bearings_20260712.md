> ⚠️ **SUPERSEDED SNAPSHOT (2026-07-12).** This was a one-time desk-review
> stock-take, not a living doc. For current best solution + settled decisions,
> read **[alignment_state_of_record.md](alignment_state_of_record.md)**. Kept for
> history; do not treat anything below as current (e.g. it says "20,000 sets" —
> the operative target is now ~40,000).

# Bearings — weekend of 2026-07-10…12, grounded against the north star

> **What this is.** A stock-take produced after a weekend of heavy creation, to
> answer one worry: *did we hallucinate?* Every weekend thread was desk-reviewed
> against on-disk reality (code, tests, output artifacts, `aux.db`) — **no pi,
> no re-runs.** Claims are tagged ✅ verified / ⚠️ unverifiable-without-rerun /
> ❌ contradicted. This is a grounding artifact, not a status doc: **canonical
> headline numbers still live only in [alignment_status.md](alignment_status.md).**
>
> **Method caveat:** desk-review depth. Unit tests were run (all passed);
> end-to-end scorer/aligner numbers were *not* re-run, so any "+Npp" improvement
> claimed in a commit message but not persisted as an artifact is marked ⚠️.

---

## The two-tier north star (as clarified 2026-07-12)

**Operative North Star (now) — the gate.**
A **SOTA, rigorous alignment algorithm that works across ~20,000 DJ sets.**
Every thread is judged on: *does it move us toward SOTA + rigorous alignment at
20k-set scale, and is it real?*

**North-North Star (preserved future work, gated on the above).**
A **DJ Music Research Lab** — fuse **SoundCloud + 1001Tracklists** into one data
engine to gain new knowledge about music, *specifically why we like it*, and to
**empower humans (and ourselves) to create better DJ sets.** Alignment is the
*necessary-but-not-sufficient* foundation; the lab activates only once alignment
is solved at scale.

**Organizational consequence of the pivot:** when the lab activates it
**separates** the music-understanding work (EDA, `information_dynamics`, and the
Appleseed / compiler / cast product layer) from the lean alignment engine. Until
then, that work is *preserved but deprioritized* against the alignment gate.

**Verdict tags used below:**
- **Foundation** — serves the operative alignment gate (priority now).
- **Scale-infra** — serves the "across 20k sets" clause (corpus throughput).
- **NN-star** — north-north-star / research-lab work; preserve, defer.

---

## Net verdict

The weekend **did not hallucinate at the code or numbers layer.** Across 11
threads the code is tested (all unit suites green), artifact-backed, and
unusually self-critical (cotrain flags its own reverted wrong-turn; infodyn
reports "weak go"; the paper draft explicitly disclaims SOTA; the SSOT ships a
self-auditing corrections ledger). The real problems are **(1) sprawl**,
**(2) a ground-truth base only n=2 sets wide**, and **(3) a product/lab layer
tangled into the alignment engine** that the pivot cleanly separates.

---

## Ledger

Schema: **Thread · Claim(s) · On-disk evidence · Verdict · North-star fit · Disposition.**

### 1. Scorer de-inflation (WS0/WS3) — `a69afb8`, `a35e4d9`, `de6ceda`
- **Claims:** score trajectory over played+gain-audible intervals, not the
  gap-inflated envelope; multiseg+loop traj BB11 15.7→26.0%, BB12 19.5→28.3%,
  GT-seconds-lost 85→76%; linear spans byte-identical; genuine walls (acappella
  oddratio 3%, loop 1%) unchanged.
- **Evidence:** `tests/alignment_prototype/test_audible_recall.py` +
  `test_score_spans.py` pass; `score_spans()` refactor golden-diff byte-identical.
  The **+~10pp numbers are commit-message-only** — `build_span_table.py` writes a
  CSV but none is committed.
- **Verdict:** ✅ code · ⚠️ improvement numbers unverifiable without re-running the scorer.
- **Fit:** Foundation (honest metric + correct attribution). **Disposition: keep.**

### 2. GT-export bug fixes — `c5872bc`, `862060d`, `f6ad55a`
- **Claims:** drop deactivated-track clips (BB11 108.6s/3 spans, BB12 141.6s/5
  spans phantom GT removed); fail-loud id-coverage gate; slot_label→recording_id
  bridge lifts stale-`.als` re-export to 98%/99%. Good Charlotte was a red herring.
- **Evidence:** `tests/labeling/test_export_drops_deactivated.py` +
  `test_export_id_coverage.py` pass; bridge fixtures present
  (`labeling/fixtures/id_maps/{1fsnxchk,2nvzlh2k}_slots.json`).
- **Verdict:** ✅ code + fixtures · ⚠️ **effect not yet applied to canonical state**
  — commits self-state "set_ground_truth on pi-storage still holds the old export."
- **Fit:** Foundation (clean GT). **Disposition: keep.** Carries the one concrete
  pending action (see below).

### 3. Pitch/detune — `75c8b30`, `08ad00c`, `ec23dd4`
- **Claims:** log-freq cross-corr cents estimator (±2.5¢ synthetic recovery);
  varispeed rejected (H1 R²=0.005); integer offsets = deliberate harmonic
  key-match to the paired bed (bed-compat 1/56→50/56 after transpose).
- **Evidence:** `tests/test_pitch_detune.py` passes; `aux.db :: bb_pitch_detune_v1`
  = **28 rows**; `out/pitch_detune_clips.csv` = **521 clip-units**.
- **Verdict:** ✅ verified. **Caveat:** hypothesis flipped mid-weekend
  (08ad00c "wrong-key rips" → ec23dd4 "harmonic key-match"); resolved, but a
  visible churn artifact.
- **Fit:** Foundation-leg (re-pitched acappellas explain why chroma alignment
  fails — a detune-aware channel serves the aligner) **+** NN-star preview
  (harmonic-mixing = "why DJs do what they do"). **Disposition: keep.**

### 4. Cotrain / LOSO — `8ef4ad3`…`e67182f`
- **Claims:** LOSO ran (MPS ~4min); identity transfers 100% both directions;
  placement does NOT (bb11 18.6s vs bb12 1436s, >75× unstable → head memorizes
  placement per-set, doesn't learn transferable placement).
- **Evidence:** `cotrain_loso_findings.md` — exceptionally honest: flags a
  **reverted wrong-turn** (`_pad_slot`), a real wiring bug it found
  (`cue_seconds`→`cue_s`), and "n=2 cannot disambiguate, do not over-explain."
- **Verdict:** ✅ verified & self-critical · ⚠️ its *internal* north-star line
  still reads "SOTA GT-closeness on all 1001tl data" (old framing → re-anchor).
- **Fit:** Foundation (aligner generalization). **Disposition: keep.**

### 5. Experiments / ablation harness — `2771fa0`…`4232ae7`
- **Evidence:** `experiments/results/scores.db` = **1363 rows**; 9 cache files;
  tests pass.
- **Verdict:** ✅ verified.
- **Fit:** Foundation/tooling (rigorous ablation → paper rigor). **Disposition: keep.**

### 6. eval_bench / André (UnmixDB) — `b47d571` + resample arm
- **Claims:** calibrated standing vs SOTA (André 2024 NMF); resample-arm fixes strata.
- **Evidence:** `external/unmixdb_findings.md` reads the paper directly and frames
  it correctly ("detection not identification; not a leaderboard entry");
  `external/out/reduction_table.txt` holds real per-stratum warp-error numbers
  (nmf/dtw/fused × timewarp × effect, n≈57–60/cell); external tests pass.
- **Verdict:** ✅ verified, rigorously framed.
- **Fit:** Foundation (external rigor / calibrated standing → paper). **Disposition: keep.**

### 7. streaming_mir WS2 — `c3d2b20`…`cad14a3`
- **Claims:** overlap heals stem-seams; boundary SDR 26→42dB; plateaus ~6s;
  167dB spike = ½-core artifact; "WS2 SETTLED."
- **Evidence:** results exist **only as an in-prose table in
  `workspaces/streaming_mir/RESEARCH_BRIEF.md` (≈L146–163), n=1 clip
  (90s, single model)** — no reproducible json/csv/db. Thinnest evidence of the weekend.
- **Verdict:** ⚠️ result present but low-n & in-prose.
- **Fit:** Scale-infra (block-online separation for corpus throughput) — on-axis
  for "at scale," low priority. **Disposition: park/deprioritize.**

### 8. information_dynamics — `2153cba`, `bb_mashup_surprise`
- **Claims:** chroma-KL surprise separates real mashups from key/BPM-matched-but-
  unchosen pairs; "WEAK GO," AUC 0.576 vs 0.498 baseline.
- **Evidence:** `lab/information_dynamics/FINDINGS.md` honest about the modest
  signal; persisted to `aux.db :: bb_mashup_surprise_p0_v1` (11 rows) / `v2` (22 rows).
- **Verdict:** ✅ verified (honestly weak).
- **Fit:** **NN-star preview** — "why did the DJ pick *this* mashup" is core to
  the lab but premature against the alignment gate. **Disposition: preserve, defer;
  separated by the pivot.**

### 9. SSOT + corrections + 3-axis recharacterization — `c19ca9b`, `eb21a5e`, `8b94922`
- **Evidence:** `alignment_status_corrections_20260711.md` is a **self-auditing
  ledger** — documents its own drift with 3-independent-source evidence (C1
  BB11/BB12 set-id swap; C2 loss-attribution 45%→38%).
- **Verdict:** ✅ verified, self-correcting.
- **Fit:** Foundation (truth-anchor). **Disposition: keep;** add a two-tier
  north-star header (currently single-star framing).

### 10. Specs + plans (7 + 7) — `docs/superpowers/{specs,plans}/`
- **Evidence:** alignment specs map to **built** code (ablation-harness→scores.db;
  andre-absorption→reduction_table; cotrain-loso→LOSO ran). Two are
  **APPROVED-but-deferred** and honestly labeled as such (synthetic-structure-
  benchmark, fx-ladder). Product/lab specs (appleseed-react-frontend,
  compiler-v2-grammar, cast-output-lane) are NN-star.
- **Verdict:** ✅ no phantom "done" claims found; honestly staged.
- **Fit:** split — alignment specs Foundation; product specs NN-star.
  **Disposition: alignment specs keep; product specs → lab bucket at pivot.**

### 11. Paper draft — `docs/alignment_paper_draft.md` (+ `papers/` = 8 reference PDFs)
- **Content:** *"Alignment Is Not a Scalar"* — decomposes alignment into
  identity/placement/structure. Numbers match everything verified above (identity
  84–85%; UnmixDB fp 73% rank@1; placement 37% ≈ structure 38%; per-axis traj).
- **Evidence:** explicitly disclaims SOTA (§6.2, §7) and flags n=2 as
  "suggestive, not decisive" (§6.3). `papers/` is a **literature collection**, not
  authored claims.
- **Verdict:** ✅ honest, real, near-publishable. **Not hallucinated.**
- **Fit:** Foundation/output — but see Finding B (it is a *different* paper than
  the operative north star implies). **Disposition: keep; decide the fork.**

---

## Findings that matter for bearings

### Finding A — the entire alignment effort rests on n=2 ground-truth sets
SSOT, paper, LOSO, scorer, and the decomposition are **all** BB11 (`2nvzlh2k`) +
BB12 (`1fsnxchk`). The operative north star is "rigorous across **~20,000** sets."
The gap is **ground-truth scale**, not algorithmic cleverness. The flywheel
(label more sets → train → measure held-out) is *specced* (`flywheel-escalate-
select`) but **not scaled**. The weekend built measurement machinery *on* 2 sets
rather than widening the base. **Highest-leverage move toward the operative north
star: grow GT beyond n=2.**

### Finding B — the paper we have ≠ the paper the north star wants
The draft is an honest **"we reframed the problem and are deliberately not SOTA"**
contribution — publishable, near-ready, n=2. The operative north star implies a
**"SOTA rigorous alignment at 20k-set scale"** paper. These are **two distinct
papers**, and the draft is explicitly the first. Strategic fork (owner decision,
not resolved here):
- **Ship the re-characterization paper** now (defensible, n=2), then pursue
  SOTA-at-scale as the next arc; **or**
- **Hold** and fold everything into the SOTA-at-scale paper (requires the
  GT-scaling flywheel to actually run first).

---

## One concrete carry-over (not an action taken here)
The branch `ws0-scorer-deinflation` fixes (scorer de-inflation + GT cleaning)
make the canonical [alignment_status.md](alignment_status.md) numbers stale in a
**known-good** direction. Realizing them needs the "WS1 step": re-export GT →
pi write-back → rescore → SSOT regen. Until then the improved numbers (e.g.
traj +~10pp) are **provisional**, not canonical. This directly explains the
"worst results, this is impossible" instinct: results did not get *worse* — the
metric got *de-inflated*, and spans that looked catastrophic were GT/scorer bugs.

---

*Produced 2026-07-12 by desk review (local artifacts only) against branch
`ws0-scorer-deinflation`. Unit suites run green; end-to-end numbers not re-run.*
