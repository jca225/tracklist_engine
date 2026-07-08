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
| `render_timeline` | render a predicted timeline to audio for perceptual A/B | Viz utility, no verdict — superseded by `render_review_snippets` (live: review-loop step 2). |
| `run_recon_experiment` | does recon refinement improve the aligner held-out? | See `recon_rerank` verdict; the held-out A/B did not earn a slot in `infer`. |
| `seed_tempo_test` | stage-0 crash-test writing a real tempo envelope onto the seed template | Landed as `labeling/als` tempo automation; covered by `tests/labeling/test_als_properties.py` + `tempo_curve.py`. |
| `stem_correct` | fix scraped `claimed_stem` before alignment, validated vs hand GT | Superseded by the row-text materialize fix (888caca) + `candidate_vocal_gate`. |
| `stem_match_probe` | stem→stem matching robustness (the open-lane litmus) | **POSITIVE** — stem-routed + HuBERT lifts acappella identity 0–14%→84%; wired into live stem-routed matching. |
| `transition_probe` | do regular/instrumental placement errors concentrate in transition zones? | Probe; findings folded into the failure-analysis placement bucket. |
