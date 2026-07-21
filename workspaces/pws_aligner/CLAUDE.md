# pws_aligner — Programmatic Weak Supervision Aligner (infrastructure; DS fusion REFUTED)

This fork rebuilt probe fusion as programmatic weak supervision per
[docs/superpowers/specs/2026-07-14-pws-aligner-design.md](../../docs/superpowers/specs/2026-07-14-pws-aligner-design.md).
The Phase-1 kill-gate was RUN and the DS instantiation was **refuted** — the
module is kept as infrastructure + a documented negative result.

## Gate RESULT (2026-07-14, v2b — genuine votes, correct frames): REFUTED

Dawid–Skene over categorical 2s offset bins **lost to hand-tuned
`source_priority`** on BB12: identity 32% vs 84%, ref-offset median 33.6s vs
14.0s, trajectory 33% vs 42%; DS NULL-abstained on ~95/152 spans. Stopped
before FABLE per plan (FABLE was never built). Gate v1 (votes reconstructed
from `probe_proposals`) also failed but was confounded — v2b on genuine
per-probe votes is the honest refutation.

**Root cause (representational, not plumbing):** categorical bin-agreement is
the wrong granularity for CONTINUOUS offsets with heterogeneous probe
precisions (fp ~0.2s vs chroma/hubert ~seconds) — genuinely-right probes land
in different bins, DS reads pervasive disagreement, floors every accuracy,
NULL wins. The designed calibration tripwire fired correctly: DS learned
fp .038 / hubert .014 where GT measured .474 / .318 — it under-trusts exactly
the probes GT says are good.

**Lever if revisited:** a CONTINUOUS label model — EM over per-probe Gaussian
noise σ (= *learned* inverse-variance fusion, the `neuro/` lane made
self-supervised), optionally FABLE-style instance-conditioned σ on top.
Categorical DS-style aggregation remains well-matched to CATEGORICAL LFs
(operation-type detectors from the DJ/DAW ontology — loop present? key-lock vs
varispeed?), which is the Phase-2 lane. NOT more categorical offset machinery.

## Offset-frame convention (load-bearing)

Harness probes emit `offset_s` in MIXED frames despite one contract:
chroma/continuity/hubert = ABSOLUTE ref-time; the fp path = RELATIVE diagonal
(ref − mix). The votes-file convention is RELATIVE
(`offset_s = ref_start_s − set_start_s`); `capture_votes.py` normalizes at
capture (`_ABSOLUTE_FRAME_PROBES`). Getting this wrong is catastrophic and
self-consistent — the absolute-frame probes outvote fp in the wrong frame.

## What's reusable (why this module is kept)

- `corpus_harvest.py` — flywheel step-2 batch CLI: pi DB → cue-anchored
  positive-only cases → certified-probe scorer (`corpus_mix_resolver`) → harvest
  ledger; `--census` reports eligibility (recall ceiling) before the GPU stem
  pass. CPU-only for the certified axes (regular/instrumental); runs on pi-storage.
- `capture_votes.py` — genuine per-probe vote capture: runs the real harness
  probes per span, records every `AlignmentResult` verbatim (frame-normalized).
- `verifier.py` — Confident-Learning joint estimator + the **GT calibration
  report** (`--calibrate --sets <csv>`): learned vs GT-measured accuracy per
  probe; a standing diagnostic, validation-only.
- `votes.py` (typed abstention), `hypotheses.py`, `decode_bridge.py`,
  `run_phase1.py`, `density_gate.py`.
- `label_model.py` — the refuted DS baseline, retained as the calibration
  harness / baseline, not a shipping fusion.

History: `export_votes.py` (reconstructed votes from `probe_proposals` — flat
confidence, identity pinned, incompatible frame) was the gate-v1 confounder
and was removed at merge; see `attic/EXPERIMENTS.md` (alignment_prototype) and
the spec for the full account.

### Lane 2 — operations (keylock/varispeed)

**Runner:** `run_operations.py` — applies `keylock_vs_varispeed` LF from `operations.py`
to every span in a predicted timeline.  Reuses `capture_votes.py` helpers
(`_find_aligning_dir`, `_load_manifest_by_rid`, `_mix_audio_path`,
`_ref_audio_path`, `_load_timeline`) for audio resolution.

**BB12 histogram (2026-07-14, analysis only — no accuracy claim):**

| label | count |
|---|---|
| keylock | 11 |
| varispeed | 1 |
| no_tempo_change | 66 |
| abstained (UNKNOWN) | 41 |
| TOTAL processed | 119 |
| skipped (degenerate spans) | 33 |

**tempo_ratio source:** computed from `ref_span / set_span` (placed durations)
for all 119 spans — no per-span `tempo_ratio` field was present in the
predicted timeline.  This is placement-error-contaminated; a tempo_ratio
derived from warp markers or the GT YAML would be cleaner.

**Key finding — abstain pattern:** 41/119 spans abstained (label=UNKNOWN),
meaning pitch moved but matched neither keylock (shift ≈ 0) nor varispeed
(shift ≈ 12·log₂(tempo_ratio)).  Abstains are dominated by acappellas
(33/41) — consistent with the re-pitched-acappella memory: acaps are often
pitch-shifted to fit the mix key, so the measured pitch shift does NOT
equal the tempo-derived expected varispeed shift.  The UNKNOWN class is a
genuine third operation category (key-shift under key-lock), not a LF
failure.

**Not 100% one label** — the set shows a real mixture (keylock/varispeed/
no_tempo_change/unknown), satisfying the sanity check.

**Next lever:** wire GT `tempo_ratio` from
`workspaces/source_detection/out/<set_id>_ground_truth_verified.yaml`
(295 records with measured `tempo_ratio` + `pitch_shift_semi`) as a
validation signal.  The GT also carries `pitch_shift_semi` which could
supervise a per-span pitch-shift estimator directly.

---

## Sensor phase still frozen

No new probes here. This module aggregates the existing channel inventory.

---

## Gate v3 (2026-07-14, continuous label model): PARTIAL

**Verdict: PARTIAL** — continuous EM model decisively beats refuted DS on all
axes and approaches hand-tuned fusion on identity, but does NOT clear hand
fusion on ≥2 of 3 headline axes.  Calibration tripwire: zero rank inversions
(technically clean) but EM collapsed to a degenerate uniform solution — same
root cause as v2b, different symptom.

### 3-way scorecard (BB12, 152 spans)

| metric | continuous | hand-tuned | refuted DS | winner |
|---|---|---|---|---|
| identity | 82% (124/152) | **84% (127/152)** | 32% (49/152) | hand (−2pp gap) |
| ref-offset median (straight) | **13.1s** | 14.0s | 33.6s | continuous (+0.9s) |
| strict traj HEADLINE (pileups excl.) | **17%** | **17%** | 8% | TIE |
| abstain rate | 2.0% (3/152) | 0% (0/152) | 58.6% (89/152) | continuous/hand |

Continuous beats DS decisively on every axis.  Vs hand-tuned: ref-offset and
trajectory tie/win narrowly (+0.9s, +0pp), identity misses by 2pp.  The
abstain collapse from DS's 58.6% to 2.0% confirms the continuous model solves
the NULL-flood structural problem.

### Calibration tripwire (ran — DEGENERATE COLLAPSE)

```
probe       lrn_sigma  meas_mad   lrn_acc  meas_acc  n_scored
--------------------------------------------------------------
chroma          0.050    15.176     0.980     1.000       131
continuity      0.050    21.170     0.980     1.000        84
fp              0.050     0.576     0.980     1.000        49
hubert          0.050     5.546     0.980     1.000        21
```

Rank inversions: **0** (trivially — all learned sigma_s collapsed to the
0.05 floor; ordering is undefined when all values tie).  This is a degenerate
EM solution: the model learned **uniform** parameters across all probes
(accuracy=0.98, sigma=0.05 for all), while GT-measured MADs span 15 octaves
(fp=0.576s vs continuity=21.170s).  The model is NOT learning heterogeneous
precision — it converged to a uniform-trust solution that happens to yield
reasonable scorecard numbers because fp's higher inlier weight (0.70 vs 0.41)
carries some signal.

**Root cause of degenerate collapse: singleton-match σ-unidentifiability
under sparse co-voting.** On most spans a given recording is matched by only
ONE probe (a singleton — no second probe agrees on the same recording within
that span). When one probe matches a recording alone, `_fused_mu` returns
that single vote's own offset as μ, so `resid = offset − μ = 0` exactly,
which floors σ for that probe on that span. Across a corpus dominated by
singleton matches, every probe floors uniformly. The sidecar n_scored counts
confirm the sparsity: fp fires on only 49/152 spans, hubert on 21/152 — far
too sparse for dense cross-probe co-voting on the same recording.

**Contrast with identity consensus:** dense high-consensus co-voting does
NOT cause this collapse — it recovers σ ordering correctly, as
`test_oracle_learns_sigma_ordering_and_identity` demonstrates (multiple probes
co-vote the same true recording per span and σ ordering is recovered). The
"identity consensus" hypothesis was refuted by controlled simulation. Sparsity
— not consensus — is the mechanism.

### Why PARTIAL not FAIL

The scorecard is genuinely better than DS (identity 82 vs 32, trajectory 17
vs 8, abstain 2 vs 59%).  The model is making decisions and they are better
decisions.  The degenerate calibration means it's getting the right answer
for partially the wrong reasons (fp's higher inlier weight rather than
learned precision weighting), but the result stands as measured.

### Levers for v4 (if revisited)

1. **Supervised σ prior for fp (~0.02s, UnmixDB-calibrated):** give fp a
   tight prior anchored to its empirically-measured UnmixDB precision and
   shrink toward it via hierarchical regularization. This works by
   REGULARIZING singleton-match σ toward the prior rather than letting a
   zero-residual singleton drive σ to the floor. (Previous framing said
   "decoupling from consensus" — that was wrong; the lever is
   anti-singleton-floor regularization.)
2. **Hierarchical σ shrinkage for singleton matches:** when a recording is
   matched by < 2 probes on a span (a singleton), the span is σ-uninformative
   for that probe — the M-step residual is exactly 0. Do not update σ from
   singleton matches; shrink toward the per-probe prior instead. This directly
   addresses the sparsity mechanism and is the highest-priority lever.
3. **Instance-conditioned sigma (FABLE-style):** per-span feature vector
   (stem, layer density, set-position) → σ_probe_span. Introduces span-level
   variation the M-step can exploit; orthogonal to the singleton fix.
4. **Factored identity / offset model:** separate the identity vote from the
   offset vote; run offset-only EM on identity-confirmed spans where σ learning
   is well-conditioned. Does NOT address sparsity directly (lower priority
   than levers 1–2) but reduces noise for spans where multiple probes agree.

---

## Gate v4 (2026-07-17, singleton σ-shrinkage): calibration collapse CURED — PARTIAL (advance)

Implements Gate-v3 levers 1+2 in `continuous_model.py`: the M-step accumulates σ
ONLY from co-voted matches (≥2 probes on the same recording) and shrinks σ²
toward a supervised per-probe prior with a small pseudo-count (κ=3,
`_SIGMA_PRIOR_S = {fp 0.3, hubert 3.0, chroma 8.0, continuity 10.0}`). Singleton
matches (resid ≡ 0) no longer drive σ to the floor. TDD: the old
`test_singleton_matches_floor_sigma_uniformly` (pinned the degeneracy) → new
`test_singleton_matches_shrink_to_prior_not_floor` (pins the fix); 181/181 pass.
Branch `align-pwsv4-transitions` (worktree), commit 2a6dcf5.

### BB12 scorecard (152 spans, same genuine v2b votes + scorer as Gate v3)

| metric | v4 | v3 continuous | hand-tuned | v4 vs hand |
|---|---|---|---|---|
| identity | 82% (124/152) | 82% | **84%** | −2pp (3 spans) |
| ref-offset median (straight, n=46) | **9.8s** | 13.1s | 14.0s | **WIN −4.2s** |
| strict traj HEADLINE (pileups excl) | 17% | 17% | 17% | TIE |
| abstain | 2.0% | 2.0% | 0% | — |

### The v3 degenerate collapse is CURED (the headline diagnostic)

Learned σ is now heterogeneous and **correctly ordered vs GT-measured MAD** —
where v3 floored ALL probes to 0.05:

| probe | v4 learned σ | GT meas_mad (BB12) |
|---|---|---|
| fp | 0.070 | 0.576 |
| hubert | 1.234 | 5.546 |
| chroma | 1.327 | 15.176 |
| continuity | 8.025 | 21.170 |

0 rank inversions **non-trivially** (v3's "0 inversions" was trivial — all tied at
the floor). The model now does genuine inverse-variance weighting; fp's tight σ
decisively pulls placement, which is why ref-offset beats hand-tuned by 4.2s (up
from v3's marginal +0.9s "right answer for the wrong reason").

### Verdict: PARTIAL — a real advance, not yet full promotion

Still LOSES identity by 3 spans vs hand → does not clear the strict ≥2-of-3-axis
promotion bar. But v4 (a) fixes the diagnostic degeneracy that made v3's numbers
untrustworthy, and (b) converts the marginal placement tie into a decisive win.
Residual gaps: (i) the identity delta is NOT a fusion lever; (ii) acappella
ref-offset median 29.8s (re-pitch wall; regular/instrumental ~0.1s).

### Identity misses are ingest-bound, NOT fusion-flippable (diagnosed 2026-07-17)

Classified the first 10 of v4's 28 BB12 identity misses against the votes:
**0/10 flippable, 10/10 not-voted.** In every miss ALL firing probes agreed on a
single recording — the one predicted — and that recording was never in the GT
accepted-sibling set; no probe ever proposed a GT-acceptable recording. The
fusion did the right thing given its inputs; the correct candidate never entered
the vote set. The scorer's own track names confirm the SONG is right (e.g. span 1
"Circle of Life" predicted as "Circle of Life"), and every GT is a multi-sibling
recording_id set — consistent with a recording-id **sibling-linkage / candidate-
generation** artifact upstream of the aligner (unverified on canonical DB: the
local dev copy predates the phase-4 `recording`/`work` tables). ⇒ No σ/label-model
lever closes the 3-span identity gap vs hand; it's ingest/inventory, matching
"identity solved — stop investing." The strict promotion bar's identity axis is
therefore partly a GT-linkage artifact, not an aligner deficit.

### Acappella ref-offset wall is COVERAGE-bound, not a fusion lever (diagnosed 2026-07-17)

Tested the natural PWS lever for the 29.8s acap ref-offset (stem-conditioned σ —
inflate chroma on acaps so key-invariant HuBERT dominates; Gate-v3 lever 3).
Time-overlap join of acap vote spans vs GT (approximate — medley overlap): HuBERT
FIRES on only ~20/67 acap spans; where it fires it is the good probe (~9s median,
vs ~2s in prior clean measurement), but on the ~47 spans where it abstains the
only firing probes are chroma (~60s) and continuity — both broken by re-pitching
(key change breaks chroma). So the fusion CEILING is capped by HuBERT abstention:
stem-conditioned σ could only help the ~14–20 spans where HuBERT both fires and
beats chroma (a modest squeeze), and cannot touch the abstention-dominated
majority. ⇒ The acap wall is a SENSOR/coverage problem (HuBERT abstention), which
the PWS sensor phase freezes — not a label-model/fusion lever. Matches standing
finding "HuBERT vocal ref-offset: gap = coverage." Lever lives outside PWS fusion
(HuBERT coverage / lyrics-ASR channel / learned decoder), consistent with
"acappella is the hard axis" in the recharacterization.

**BB11 (2nvzlh2k) validation NOT OBTAINED (2026-07-17)** — attempted repeatedly,
abandoned. The full chain is `infer` (builds predicted timeline — prereq) →
`capture_votes` (genuine probes; needs the Mac-local `~/aligning/2nvzlh2k` folder)
→ `run_phase1 --model continuous` → `score`. Blockers hit: (a) BB11 votes not
cached, and both `infer` + `capture_votes` re-run HuBERT-on-MPS (~1.5h total,
contended by the live synthetic-transfer spike); (b) long background-bash driver
wrappers repeatedly died ~1h in (exit 144), orphaning compute children so the
chain never completed end-to-end; (c) a Vast attempt dead-ended on a dud
oversubscribed box (load 14, clone wouldn't persist) + a `vast_bootstrap.sh`
Demucs-dep (`dora-search`) build failure — the aligner harness has never run
off-Mac and would need a slim-deps + pi-pull port first.
**Consequence:** v4's cross-set generalization is UNVERIFIED (n=1 real set, BB12).
Promotion DEFERRED. To get the cross-set read cleanly, either run the chain as one
detached `nohup` process (survives wrapper death) on an idle Mac, or do the
slim-deps Vast port (reusable infra for 40k-set-scale capture) — the latter only
worth it when capturing many sets, not one. BB12 remains the decisive primary
evidence; the two gap diagnoses below show v4 is at the fusion ceiling regardless.

**Known wiring gap:** `verifier --calibrate` reads `_pws_probe_accuracy.json`
(DS-format) but the continuous model writes `_pws_probe_noise.json` — the σ
sidecar above IS the calibration evidence; wiring the formal CLI to the continuous
sidecar is a follow-up, not a blocker.

**Repro (BB12):** `run_phase1 --set-id 1fsnxchk --votes out/1fsnxchk_probe_votes.json
--model continuous` → `score_timeline_vs_gt --set-id 1fsnxchk --timeline
out/1fsnxchk_pws_timeline.json`.
