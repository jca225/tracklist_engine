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
| `sic_phase0_probe` | can informed successive cancellation (spectral SIC) make missed medley layers identifiable? | CLOSED 2026-07-10 — cancellation works (−4 dB, physics gate passed) but adds nothing to identification: fp-visible layers were never masked (Honest 1.4k/2.8k votes in raw mix, mis-placed by decision logic = bug lead), fp-invisible layers are invisible from keylock warp geometry, not masking (lever = warp-tolerant hashing, not separation). See docs/archive/medley_sic_plan.md. |

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

## Instrumental FP-segment → timeline materialize (2026-07-19) — NO-GO AS UNGATED PATCH

**Question:** if landmark instrumental segment banks are written onto the agentic
baseline timeline (`set_start`/`ref_segments`/`start_source=fp_segment_dp`) and
scored with `score_timeline_vs_gt`, does the real board improve?

**Verdict:** no as an ungated patch. Instrumental trajectory slices can rise
(BB11 instr traj-acc 40%→50%, BB12 32%→40%) while **overall set_start placement
regresses** on both sets (BB11 median 1.2→2.5s, <15s 76%→73%; BB12 median
2.9→5.5s, <15s 78%→70%). BB12 identity also slipped (83%→81%). False/weak
decoded runs overwrite good baseline placements.

**Do not** promote ungated `fp_segment_dp` materialize. Revisit only with an
acceptance gate that preserves baseline when segment evidence is weak, plus an
independent validation set. Materializer kept at
`fp_segments/materialize.py` for gated experiments.

## Instrumental FP-segment gated materialize (2026-07-19) — STILL NO-GO FOR BOARD

**Question:** does a baseline-consistency gate (`gate_s=90`, same frozen window
as instr-stem/fp-placement; keep only segments with
`|mix_start - baseline_set_start| ≤ gate_s`) fix the ungated regression?

**Verdict:** damage control, not promotion. vs agentic baseline:

- BB12: applied 14 / rejected 4. Placement still worse (median 2.9→3.9s,
  <15s 78%→73%) though better than ungated (5.5s / 70%). Instr traj 32%→46%.
  Identity held 83%. Headline traj ~flat (33%→34%).
- BB11: applied 21 / rejected 0 (all proposals already within 90s of prior).
  Placement still worse (1.2→2.5s, <15s 76%→73%). Instr traj 40%→49%.
  Headline 26%→29%.

Near-baseline wrong overwrites still hurt overall set_start; the 90s gate
only stops teleports. **Do not** tighten `gate_s` on BB11/BB12. Needs a
stronger GT-free accept rule (evidence vs baseline, or leave set_start and
only patch `ref_segments`) and an independent set.

## Instrumental FP-segment ref-only materialize (2026-07-19) — PARTIAL

**Question:** with `--gated --ref-segments-only`, can segment banks improve
ref/traj without touching baseline `set_start`?

**Verdict:** first mode that does not regress the board. vs agentic baseline:

- Placement + identity unchanged on both sets (by construction).
- BB11: instr traj 40%→45%, headline 26%→29%, instr straight-clip ref median
  25.0→3.2s; overall ref median 37.5→26.8s.
- BB12: instr traj 32%→46%, headline 33%→34%; instr straight-clip ref median
  worsened 32.9→46.3s (n=6) while overall placement held.

**Do not** promote to default driver yet — gains are stem-local / modest, BB12
scalar ref mixed, still n=2. Keep as the preferred shadow materialize mode
(`fp_segment_dp_ref`). Next: independent set or evidence-vs-baseline accept
before wiring a driver.

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
## Ableton ReAct harness — daw_env (2026-07-19)
**Question:** does place→listen→sense→iterate in an ALS-first session beat
probe-only agentic on hard placement (ridge decoder_wall), without Live OSC?
**Infra:** `workspaces/alignment_prototype/daw_env/` — Mode A/B CLI, unit tests
green (`tests/alignment_prototype/test_daw_env.py`). Spec:
`docs/superpowers/specs/2026-07-19-ableton-react-harness-design.md`.
**Live sensors (2026-07-19 follow-up):** listen→sense runs LiveContext `fp` +
`stem_hubert` (lyrics off by default; `--with-lyrics` opt-in). Nudges are
**content-only** within 24s of geom — onset fallback caused 36s cue-seed drifts
on regulars.
**Hard-bucket scorecard (BB11, agentic seed, aca+instr hard spans n=18):**
median place-err **82.0 → 82.0 s** (`<15s` still 0%). Ridge `decoder_wall`
slots **39 / 34 unchanged** (fp proposals ~80s off GT; 24s gate refuses). One
instrumental (37) improved −14.3s err via fp nudge but remains 67s off GT.
**Verdict: NO-GO as placement decider / board mover.** Keep as Mode B labeling
assist + sensor sandbox. Default `make align` / race board **unchanged**.
Revisit only with a third GT set or a verifier that can accept ~80s fp/HuBERT
corrections safely (same bar as symmetric-FP NO-GO).

**Fixes that unblocked BB10 (do not weaken G1/G2):** (1) prefer `mix.wav` over
`mix.m4a` for fp load; (2) `resolve(require_independence=True)` in pseudo-safe so
lyrics-alone auto does not skip HuBERT; (3) HuBERT prior falls back to
lyrics/timeline when mert/cue absent; (4) `resolve_track_id` accepts rid stored
as manifest `track_id` when `recording_id` is null. Whisper long-mix checkpoint
kept the lyrics cache warm. Disco Lines remains a dead pool for E1.

## BB12 inventory coherence (2026-07-19) — neutral

**Question:** after BB12 inventory-audio repair (aligning ref files + spectrogram
`src=yes`), does a fresh score of the frozen `_lt` / agentic timelines move
identity, traj, stem_mismatch, or decode-residual on the real board?

**Verdict:** neutral — inventory coherence fixed evaluation surfaces but not the
end-to-end scoreboard on unchanged timelines. Park further inventory polish;
return to algo (routing materialize, decode, placement).

**Evidence (BB12 `1fsnxchk`, vs `docs/alignment_status.md` §1 regenerated
2026-07-19):**

- **Spectrogram / preflight:** Task 7 `src=yes` **139/139** (was broken); scorer
  preflight still flags **2** acquisition gaps (`tlp2853054`, `tlp2853062`) —
  `--strict-inventory` aborts; scored without strict.
- **Live `_lt` classical** (frozen timeline, `--decompose`): identity **127/152
  (84%)** unchanged; set_start median **5.1s / <15s 71%** unchanged; multiseg+loop
  traj **30% strict → 50% fiber** unchanged; span-table **stem_mismatch 62/139**
  unchanged (materialized `claimed_stem` routing not re-run).
- **Agentic timeline** (same inventory, no re-infer): placement **2.9s / 78% <15s**
  (better than classical); traj headline **35% strict → 55% fiber** — algo delta,
  not inventory.
- **Oracle `_gtstem_lt` (Task 7):** stem_mismatch **0/139** confirms routing was
  confounding diagnosis; modest traj lift (+6 spans ≥0.95) shows inventory alone
  is insufficient for board movement.

**Do not** re-litigate inventory-audio paths on BB12 before fixing live stem
materialize / re-infer. **Do** run BB11 inventory pass next only if routing +
algo blockers are addressed in parallel.

## BB11 inventory coherence (2026-07-19) — manifest wired, pi debt unchanged

**Question:** same inventory-repair tooling as BB12 on `2nvzlh2k` — does manifest
reconcile + GT slot remap clear blocking inventory and move the board?

**Verdict:** manifest-only progress — aligning tree wired (27 stem rewrites,
2 added slots); **blocking stays 26/152** (110 satisfied, 16 fallback-warn). No
Demucs mass-promote (23/23 GT-blocking slots have local vocals but count is too
high for lean pass). Re-score skipped: frozen `lt_pred_2nvzlh2k_regular_noaudit.json`
lacks `recording_id` on spans (schema drift).

**Evidence:**

- `reconcile_aligning_manifest 2nvzlh2k --apply`: wired 152 slots; unresolved
  **017w2, 027, 038w2** (disk/manifest gaps).
- GT remap: **1** row (`013w1` → `013w3`, Kill FM - Fresh); `needs_human`: `mix`.
- Worklist: `labeling/out/bb11_inventory_worklist.csv` — **23** GT-blocking
  (mostly claimed-acappella `wrong_stem` + a few `replace_version` regulars);
  **23** acquisition cases opened in `data/acquisition_cases/2nvzlh2k.jsonl`.
- Preflight on scorer attempt: **0** missing ref audio (GT fixture) — spectrogram
  surface OK; pi `track_audio` stem rows still the blocker.

**Next:** selective Demucs promote or official acappella acquire per worklist
(not batch-16 like BB12); fix slot **027** missing bed; resolve reconcile
orphans **017w2/038w2** before strict inventory.

## Source-family Beta calibration for Phase 2 beliefs (2026-07-26) — STRUCTURE NO-GO

**Question:** can a deliberately low-capacity, leakage-safe LOSO calibrator turn
the existing timeline's source-family labels into transferable PLACEMENT and
STRUCTURE posteriors, sufficient to exercise the provenanced timeline decoder?

**Method:** fit Beta precision posteriors on one complete-GT set and freeze them
before applying to the other; repeat in both directions. Placement correctness
uses the canonical placement gate. Structure correctness uses the canonical
strict-trajectory acceptance gate. Sparse source families fall back to the
train-set global posterior. Every model is stamped `development_only`; same-set
fit/apply is rejected.

**Verdict:** useful infrastructure and a placement shadow lane, but **NO-GO for
structure or cutover**. Placement source families clear the existing posterior
floor in both transfer directions. No structure source family clears it, so the
honest result is universal structure abstention. Lowering the posterior floor
would force guesses from a source label that has not demonstrated sufficient
cross-set precision; do not do that.

**Keep:** the LOSO producer, model artifact, belief bundles, null-preserving
decoder, and default-off scorer seam. **Do not re-test** source-family-only
structure calibration on these two sets. The next structure calibrator needs
candidate-level evidence (path score shape, agreement, fiber ambiguity, or a
learned trajectory posterior), not merely `ref_decode_status`.
