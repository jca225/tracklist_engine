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

## TRM decoder graft — sim2real gap MEASURED (2026-07-18)
**Question:** does the Tiny-Recursive-Model decoder (bake-off
`docs/trm_decoder_bakeoff.md`), trained on synthetic mashups, transfer to real
BB? Architecture + wiring first, then the honest cross-set number.
**Verdict: architecture WORKS; the wall is DATA (sim2real), not the model.**
Measured on the `_lt`-independent `trajectory_acc` referee (strict, no fibers),
control = raw match-sim argmax:
- **v0 overfit** (6 real spans, eval==train): traj-acc **0.95** — offset-coord
  encoding + recursion + decode + train loop all correct. It can learn.
- **real-only cross-set** (train BB12 → eval BB11, ~150 real spans): pure
  memorization — train 0.61↑ / eval **0.075**↓, below the **0.239** control.
  Confirms the bake-off "2 real sets = memorization" call.
- **synthetic-only → real** (40 `generate_v2` windows, 311 spans, synthetic-only
  loader, real BB eval-only): train-fit climbs 0.095→**0.87** over 200 epochs
  (learns synthetic fine) while real-BB eval stays **flat ~0.09**, far below the
  **0.306** control. Train-high / eval-flat = **sim2real gap**, NOT underfitting.
**Do not throw GPU/scale at this** — more epochs only memorized synthetic better
(eval never moved). The lever is synthetic REALISM (bb12-lite curriculum is too
clean: no EQ/effects/crowd/transition modeling) or a pivot to the real
pseudo-label flywheel (status doc), with TRM as the decoder trained on real
pseudo-labels. Stability fix on record: answer-latent LayerNorm + grad-clip 1.0
(first run exploded, CE ~8e7). Infra kept + wired: `trajectory/trm.py`,
`trajectory/offset_coords.py`, `train.py --model trm --synthetic-only
--max-train/--max-eval`, ablation framework `pipeline/`. Branch
`trm-ablation-framework`. Numbers here are DIAGNOSTICS, not SSOT — see
`docs/alignment_status.md` for headline metrics (unchanged by this).

## E1 pseudo-label flywheel — STARVED then VIABLE smoke (2026-07-18/19)
**Question:** can pseudo-safe agentic AUTO_COMMIT on an unlabeled pool produce
enough audited labels to train a TRM that beats synthetic-only / conv on held-out
BB11?
**Verdict (Disco Lines pool): starved** — `pool=1rfb0yl9` → `eval=2nvzlh2k`,
fp-only / no exportable 330M MERT; `accepted=0` at G0. Infra (Tasks 1–4) was fine.
**Verdict (BB10 smoke, 2026-07-19): viable** — `pool=w1mgcjt` → `eval=2nvzlh2k`,
`--smoke-only`. Artifacts: `out/e1/e1_result.json` status `"completed"`,
`accepted=3` (lyrics∩HuBERT G2 survivors after fixes below). Smoke TRM train ran
(`out/e1/logs/smoke_trm.log`).
**Verdict (BB10 full E1, 2026-07-19): noise-floor** — same pool/eval, drop
`--smoke-only`, `--reuse-agentic`. Pipeline completed (`smoke_only: false`); logs
under `out/e1/logs/{conv,synth_trm,pseudo_trm}.log`. Pseudo-TRM overfits the 3
train spans and does **not** beat the raw control (or conv) on held-out BB11.
Do **not** regenerate `docs/alignment_status.md` from this run. Next lever is
more real pseudo mass (better lyrics∩HuBERT agreement / fp landmarks / a third
labeled set), not more epochs on n=3.
**Fixes that unblocked BB10 (do not weaken G1/G2):** (1) prefer `mix.wav` over
`mix.m4a` for fp load; (2) `resolve(require_independence=True)` in pseudo-safe so
lyrics-alone auto does not skip HuBERT; (3) HuBERT prior falls back to
lyrics/timeline when mert/cue absent; (4) `resolve_track_id` accepts rid stored
as manifest `track_id` when `recording_id` is null. Whisper long-mix checkpoint
kept the lyrics cache warm. Disco Lines remains a dead pool for E1.
