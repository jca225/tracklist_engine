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
