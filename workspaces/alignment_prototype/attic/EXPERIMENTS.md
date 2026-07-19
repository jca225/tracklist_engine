# Attic — closed experiments, with verdicts

One-off probes and evals whose questions are **answered**. Nothing here is
imported by the live kernel (verified mechanically before the move: zero
import/invocation references in tracked `*.py` / `*.sh` / `Makefile`).
Do **not** re-run one of these to "check an idea" without reading its verdict
first — most of them are here precisely because the idea was measured and
closed.

Scripts still run in place (`python -m workspaces.alignment_prototype.attic.<name>`);
`_REPO` path hacks were bumped for the extra directory level. To resurrect one,
`git mv` it back up and revert `parents[3]` → `parents[2]`.

| script | question it asked | verdict |
|---|---|---|
| `acappella_dtw` | acappella placement via BPM stretch-lock + slope-constrained DTW | Superseded — per-stem HuBERT placement (`stem_placement.py`, `--stem-placement`) won this lane (61→76% <15 s). |
| `acappella_ref_offset_eval` | matched-filter ref-offset MAE on straight acappella clips | Metric absorbed into `eda/alignment/failure_analysis` (the Finding-3 "X-ray" table). |
| `acappella_warp_decode_probe` | is acappella decode failure anchor/grid vs warp? | Warp is near-linear; a wider stretch grid **regresses**. Lever = placement + learned instance selection, not the decode grid. |
| `artifact_invariance_probe` | do separation artifacts break stem matching (contrastive adapter go/no-go)? | **NO-GO** — HuBERT on vocals already .905; artifacts are not the wall (commit 7eb54d6). |
| `beat_domain_detect` | instrumental-BPM-anchor ref-offset heuristic | Folded into the warp-prior work (acappella warp is BPM-derived, commit 1bbb919). |
| `eval_placement` | end-to-end FP placement pipeline vs GT | Superseded by `eda/alignment/failure_analysis` placement attribution; `--fp-placement` is an `infer` default now. |
| `eval_ref_detection` | ref-offset detection vs BB12 GT, stem-routed probe windows | Superseded by the failure-analysis span table (same aggregates, impact-weighted). |
| `fp_fuse` | does vote-gated fingerprinting recover wrong-CONTENT decodes? | Yes — the vote-gated override is wired into `infer_fused`; this standalone eval is closed. |
| `host_grader_eval` | FP sharpness as a host-channel precision signal | Folded into `neuro/` precision fusion; fp is over-trusted and unstable under leave-one-set-out (0.90→0.53). |
| `instrumental_ref_offset_eval` | instrumental mirror of the acappella ref-offset eval | Chroma is fine for instrumental where-in-song; the weak axis is set_start-under-crosstalk, not ref offset. |
| `joint_decode_probe` | oracle-cancellation feasibility for joint two-bed decode | Probe only; no production follow-up. |
| `lyrics_grader_eval` | lyrics-ASR as the vocal-channel grader | Folded into precision fusion; lyrics precision transfers across sets. |
| `lyrics_placement_refine` | wire lyrics-channel set_start into the predicted timeline | Superseded — the lyrics channel lives in the live probe runners (`agentic --live`) via `lyrics_align.py`; the gap is coverage, not wiring. |
| `phaseb_probe` | can lyric CONTEXT discriminate the right repeat instance? | Thread closed — lyric fibers over-merge (~45%); decode-time instance selection is the live lever. |
| `recon_rerank` | reconstruction-margin placement refiner for regular (host) spans | Recon localizes REGULAR only (79 vs 8%) and fails acappella/instrumental; kept as a finding, not a component. |
| `reinfer_driver.sh` | (not an experiment) one-off post-fix re-infer orchestration, 2026-07-09 | Ran to completion (produced `out/*_predicted_timeline_lt_v2.json`); superseded same morning by `drivers/` + `make race`. |
| `render_timeline` | render a predicted timeline to audio for perceptual A/B | Viz utility, no verdict — superseded by `render_review_snippets` (live: review-loop step 2). |
| `run_enhance_ab.sh` | does VoiceFixer restoration of candidate acappella stems before Whisper improve the lyrics channel? | **NEGATIVE** — BB11 full A/B: candidate coverage unchanged (26/26 spans), MONOTONIC+prior median 2.2→8.3 s (<5 s 54→42%). Enhanced-BB12 arm died mid-run and was not rerun — the BB11 regression already answers the gate. Real lever landed instead: route acappella candidates to source audio when no vocals stem (4935f32). `--enhance-vocals` stays default-off; raw logs in `out/enhance_ab/` (untracked). Includes its GPU-idle watcher `wait_then_run_ab.sh`. |
| `run_recon_experiment` | does recon refinement improve the aligner held-out? | See `recon_rerank` verdict; the held-out A/B did not earn a slot in `infer`. |
| `seed_tempo_test` | stage-0 crash-test writing a real tempo envelope onto the seed template | Landed as `labeling/als` tempo automation; covered by `tests/labeling/test_als_properties.py` + `tempo_curve.py`. |
| `stem_correct` | fix scraped `claimed_stem` before alignment, validated vs hand GT | Superseded by the row-text materialize fix (888caca) + `candidate_vocal_gate`. |
| `stem_match_probe` | stem→stem matching robustness (the open-lane litmus) | **POSITIVE** — stem-routed + HuBERT lifts acappella identity 0–14%→84%; wired into live stem-routed matching. |
| `transition_probe` | do regular/instrumental placement errors concentrate in transition zones? | Probe; findings folded into the failure-analysis placement bucket. |
| `sic_phase0_probe` | can informed successive cancellation (spectral SIC) make missed medley layers identifiable? | CLOSED 2026-07-10 — cancellation works (−4 dB, physics gate passed) but adds nothing to identification: fp-visible layers were never masked (Honest 1.4k/2.8k votes in raw mix, mis-placed by decision logic = bug lead), fp-invisible layers are invisible from keylock warp geometry, not masking (lever = warp-tolerant hashing, not separation). See docs/medley_sic_plan.md. |

## Candidate critic v0 (2026-07-19) — ORACLE POSITIVE, CRITIC NO-GO

**Question:** can a baseline-aware logistic critic select the best existing
placement candidate using serialized proposal agreement, baseline provenance,
native top-K FP strength, and synthetic instrumental-FP hard negatives?

**Verdict:** the candidate oracle clears the two-set placement gate, confirming
that proposer recall is sufficient. The learned critic does not transfer
bidirectionally under set-level holdout. Adding baseline-source provenance
improves ranking in one direction, but calibrated acceptance remains
low-precision and accepted regressions dominate in the reverse direction.
Threshold tuning is closed because only two real GT sets exist and further
held-out adjustment would be leakage.

**Audio-pair follow-up:** a pinned stem-routed verifier was then trained on
local diagonal similarity, nearby-shift margin, continuity and synthetic exact
instrumental pairs. It improves ranking in one holdout direction but reverses
in the other; MERT-only verified candidates also fail transfer. Therefore
simple chroma-summary verification is not sufficient.

**Do not re-test** these tabular/pinned-summary critics on BB11/BB12 with more
thresholds. Revisit candidate arbitration only after one of: (a) a third
independent real GT set, (b) proposer-native evidence for MERT/lyrics/HuBERT
rather than mostly serialized argmax positions, or (c) a learned
superposition-invariant audio-pair verifier with a genuinely independent
validation set. Reusable contracts remain in `candidate_arbiter/`; no critic is
wired into a driver.

## Symmetric instrumental landmark FP (2026-07-19) — NO-GO AS DECIDER

**Question:** does correcting the live observation model from full-mix hashes
against reference-instrumental hashes to separated-instrumental hashes on both
sides make landmark FP safe as an instrumental placement decider?

**Verdict:** no. The symmetric lane is the correct architecture and remains
useful as a candidate generator, but its joint decoded placements regress
placement cleanliness on both complete-GT sets. An FP-only overlay also
regresses both sets while leaving the instrumental and overall trajectory
headlines unchanged. Native top-K oracle headroom improves slightly in one
direction, confirming that some useful alternatives exist, but vote count,
density, runner-up ratio, and displacement do not separate the catastrophic
repeat/alias diagonals from the wins across sets. The production runner now
reads `mix_instrumental.flac` and a stem-specific cache name, but downstream
acceptance must remain fail-closed/shadow-only.
`AGENTIC_LIVE_ENABLE_FP_PLACEMENT=1` is an experiment-only escape hatch.

**Do not re-test** thresholds on BB11/BB12. Revisit only with an independent
validation set and a verifier trained to distinguish the same musical material
from repeated instrumental texture, or with local monotonic path evidence that
corroborates an FP diagonal before it may move a placement.

## Multi-channel FP path corroboration (2026-07-19) — NO-GO AS GATE

**Question:** after whole-mix NULL-aware segment decoding, can independently
decoded full-mix/full-reference fingerprint paths verify the primary
instrumental-mix/instrumental-reference paths without changing their geometry?

**Verdict:** the typed fusion behavior is correct and synthetic tests pass, but
the real-set signal does not transfer. Full-channel agreement accompanies many
correct BB11 paths but also many false ones; on BB12 it misses the small correct
set while corroborating false paths. Missing channels remain distinct from
negative evidence, as intended, but no cross-set acceptance threshold is
earned. Keep the fusion contract for future independent representations; do
not use exact full-channel FP agreement as a placement gate.

**Scope correction:** this was also not the requested production architecture.
The aligner uses independent instrumental-to-instrumental and vocal-to-vocal
lanes; it does not require full audio to corroborate either.

**Do not re-test** thresholds on these two sets. The next verifier must add
genuinely different evidence (sustained chroma/HuBERT or learned
superposition-invariant similarity) and an independent validation set, not
another view of the same landmark collisions.

## Vocal-to-vocal landmark segment lane (2026-07-19) — REPRESENTATION NO-GO

**Question:** with routing mechanically restricted to `mix_vocals.flac`
against each constituent's `vocals.flac`, can sparse constellation landmark
correspondences drive the same NULL-aware segment decoder?

**Verdict:** the strict independent lane is operational and never falls back to
regular/instrumental audio, but landmark coverage is too weak and false paths
dominate across the real sets. This is consistent with the existing axis
contract, which omitted fingerprinting for vocals. The routing architecture is
correct; the vocal observation must use HuBERT/phonetic or lyric anchors
instead of Shazam-style constellation hashes.

**Do not re-test** vocal landmark thresholds on BB11/BB12. Reuse the lane,
segment schema, and NULL-aware decoder with vocal-specific correspondences.

## Instrumental segment-bank BB11/BB12 autopsy (2026-07-19) — DIAGNOSTIC

**Question:** is BB12’s weak instrumental landmark segment bank the same
failure as BB11’s residual errors (collision / decoder), or a different wall?

**Verdict (corrected):** the first pass under-counted BB12 because Ableton GT
slot labels were zero-strip-matched onto unrelated tracklist slots. After
recording-id stem overrides (`fp_segments.stem_overrides`), BB12 coverage is
20/≈21 GT instrumentals (1 true inventory gap) with segment recall@15 ≈ 0.60
vs BB11 ≈ 0.77. Remaining asymmetry: BB12 misses are mostly ridge-absent in
the landmark cloud; BB11 misses usually keep strong GT-band support and fail
at boundary/false-extra selection. Match counts are huge on both sets.

**Do not** use slot-norm GT overrides on BB12. Next levers: chroma (or other)
second observation for ridge-absent misses; false-run rejection for
BB11-style near-misses — still no threshold mining on these two sets alone.

## Instrumental chroma peak segment lane (2026-07-19) — MIXED / NO DEFAULT

**Question:** can peak-sparsified instrumental chroma (same vocal HuBERT
adapter pattern) beat landmark FP on the honest recording-id instrumental
slice?

**Verdict:** mixed. BB12 recall@15 improved (0.60 → 0.65) with fewer
false-only decodes; BB11 regressed (0.77 → 0.64). Keep `--observation chroma`
as a shadow tool; do not make it the instrumental default. Landmark remains
the default instrumental observation.

**Do not** retune chroma peak_frac on BB11/BB12 to chase the asymmetry.

## Instrumental secondary-run evidence floor 0.25 (2026-07-19) — REGRESSED

**Question:** does raising `min_run_evidence_fraction` from 0.05 to 0.25 cut
false extras without losing true paths?

**Verdict:** no on these two sets. Landmark recall@15 fell on both (BB12
0.60→0.45, BB11 0.77→0.64). Default remains 0.05; the stricter fraction stays
available for synthetic/explicit calls only.

## Vocal-to-vocal HuBERT peak segment lane (2026-07-19) — REPRESENTATION NO-GO

**Question:** with the same vocal-to-vocal route and NULL-aware decoder, can a
whole-mix HuBERT-L9 cosine matrix sparsified to local peaks
(`fp_segments.hubert_retrieve`) recover useful constituent segments?

**Verdict:** the observation adapter is operational (`--lane vocal
--observation hubert`, default for vocal), reuses `.feat_cache`, and never
crosses stems. Real shadow banks are denser than landmark vocal (more
`decoded` rows; median matches hit the frozen `max_peaks=256` cap), but
placement recall stays near floor and false-path duration dominates on both
complete-GT sets. Peak-sparsified HuBERT is therefore the wrong
correspondence producer for this decoder — not a routing failure.

**Do not re-test** HuBERT peak-frac / neighborhood / max-peaks on BB11/BB12.
Keep the lane + decoder. Next vocal observation candidates are sustained
phonetic/lyric anchors or a different decode of the dense \(M\) (not another
peak-threshold sweep on these two sets).

## PWS categorical label model over offset bins (2026-07-14) — REFUTED
**Question:** can a Dawid–Skene label model (learned per-probe accuracy, no GT)
beat hand-tuned `source_priority` fusion, aggregating genuine per-probe votes
over a (recording_id × 2s offset-bin) hypothesis space?
**Verdict: NO — twice.** v1 (votes reconstructed from `probe_proposals`):
confounded, DS collapsed winner-take-all. v2b (genuine harness-probe votes via
`pws_aligner/capture_votes.py`, offset frames normalized): DS NULL-abstained on
~95/152 BB12 spans, identity 32% vs 84%, ref-offset 33.6s vs 14.0s, traj 33%
vs 42%. Root cause representational: categorical bin-agreement mismatches
continuous offsets with heterogeneous probe precisions (fp ~0.2s vs
chroma/hubert ~s) — right answers land in different bins, DS floors everyone.
Calibration tripwire (learned vs GT-measured) fired correctly: fp .038 learned
vs .474 measured. FABLE never built (gated off, correctly).
**Do not re-test** categorical offset-bin label models. Levers if revisiting
weak supervision: (a) CONTINUOUS label model — EM over per-probe Gaussian σ =
learned inverse-variance fusion (neuro/ lane, self-supervised); (b) categorical
DS aggregation on genuinely CATEGORICAL LFs (operation-type detectors from the
DJ/DAW tool ontology). Infra kept in `workspaces/pws_aligner/` (capture,
calibration report, typed abstention). Spec: docs/superpowers/specs/
2026-07-14-pws-aligner-design.md. Verdict memory: project_pws_gate_verdict.
