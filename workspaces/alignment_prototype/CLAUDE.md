# alignment_prototype — P5 span aligner (offline)

Incubates in `workspaces/` per [alignment_program_plan.md](../../docs/alignment_program_plan.md).
Promote to top-level `alignment/` when stable.

**Closed experiments live in [attic/](attic/EXPERIMENTS.md)** — a verdict ledger
of ~20 one-off probes/evals whose questions are answered. Read the verdict
before re-testing an idea; most attic scripts are there because the idea was
measured and rejected.

## Live kernel

Six entry points; everything else at top level is a module they import.

| entry point | role |
|---|---|
| `infer.py` | cross-set inference: identity (MERT `predict_sequence`) + placement (`--fp-placement` + `--stem-placement`, both default on) + ref offsets |
| `joint_ref_decode.py` | post-infer segment decode: `path_decode.decode_path` Viterbi writes per-span `ref_segments` into the timeline JSON (feature-routed: acappella→HuBERT, else chroma; `--decoder looptrace` for the loop-collapse variant) |
| `train.py` | MERT head training / eval (`--eval`, `--train-mert`); watch the `MISS` report on new sets |
| `path_decode.py` | Viterbi span decoder over ref offset (loop/jump/odd-ratio); `--eval` = oracle-placement upper bound; also home of `trajectory_acc` scoring |
| `agentic/` | POMDP agentic loop (`python -m ...agentic --set-id <id> --gt <yaml>`); `--live` runs real fp/lyrics/HuBERT/mert/surprise probes |
| `harness/` | Probe/AlignmentResult/DeterministicDriver contract; `axes.py` = stem→axis routing; `merge.py` `source_priority` = axis-priority fusion |

Core modules (imported, not run): `dataset/records/split/losses/eval` (GT spans),
`mert_store/mert_features/mert_model` (identity), `landmark_fp/fp_index/mix_fp_hits/
fp_placement_refine` (fingerprint placement), `refine_ref_offsets` (chroma
matched filter), `stem_placement` (HuBERT acappella set_start), `ref_fibers`
(repeat classes), `lyrics_align` (+`vocal_enhance`/`enhance_vocal` subprocess),
`continuity_refine`, `sequence_decode`, `slot_priors`, `fine_refine`, `fiber_ab`,
`eval_bench`+`nmf_baseline` (external André-2024 benchmark), `tempo_curve`
(als tempo primitives; imported by main-suite tests), `seed_als_from_timeline` +
`render_review_snippets` (review loop), `review_server`/`fiber_server`/`fiber_ui`/
`discern_server` (human-review UIs), `export_mert_from_pi`, `pretrain` (UnmixDB).

Subdirectories: `looptrace/` (acappella loop-collapse decode), `trajectory/`
(learned segment-trajectory decoder), `neuro/` (probe-precision fusion),
`synthetic_mix/` (synthetic mashup pretrain data), `external/` (UnmixDB/SALAMI
loaders + caches).

## Scorecard — the source of truth for "did it help"

```bash
make scorecard        # per-span table + impact-weighted failure attribution (BB11+BB12)
# or per set:
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
    --set-id 1fsnxchk [--fibers] [--decompose]
```

`eda/alignment/failure_analysis/` is the canonical breakdown (one binding cause
per span, weighted by GT-seconds lost). **Axis rule: take `claimed_stem` from
the matched GT row, never from the timeline span** — the materialized value was
corrupted by the row-text drop bug (fixed 888aca, but pre-fix timelines and the
canonical DB until re-materialization still carry it; BB12 showed 2
instrumentals vs 25 real). `score_timeline_vs_gt` does this now.

State (2026-07-08, corrected routing, BB11+BB12): identity 84/83%, set_start
median 6.3/7.9 s, acappella trajectory 21% (up from 11% mis-routed), 81% of
GT-seconds still lost. Loss attribution: decode-residual 45% (the "which
chorus" repeat-instance wall) > placement ~31% > identity 6%. Acappella is 51%
of mix-seconds and the worst axis. Full numbers + prioritized levers:
[failure_analysis/FINDINGS.md](../../eda/alignment/failure_analysis/FINDINGS.md).

## Design decisions (load-bearing — do not relearn these)

- **Axis decomposition:** song ≈ timbre × harmony × language, near-orthogonal.
  timbre=MERT (identity ONLY — pooled-MERT cosine cannot localize; ~900 s off
  unconstrained), harmony=chroma, language=HuBERT. Match on the
  nuisance-invariant axis per stem: vocals→HuBERT (key-invariant — 31% of BB11
  acappellas are re-pitched and key changes break chroma; 2.1 s vs 39.6 s median
  ref-offset), instrumental→chroma+fingerprint. Fusion uses the axis prior
  (`harness/merge.py` `source_priority`), never raw cross-probe confidence.
- **Stem-wise alignment:** a mix moment is a sum of layers; full-mix-only
  matching entangles them. Align per stem channel (mix_vocals↔ref vocals, etc.)
  AND full mix, fused at decode.
- **set_start = ref_start + fp diagonal offset.** The old ~30 s "placement wall"
  was a decomposition error: the landmark fp localizes the alignment diagonal to
  0.2 s/76%; DJs start tracks mid-song. `mix_fp_hits.decode_placements` (vote
  extent + monotonic decode) is the placement source in `infer`.
- **fp is precise but unleashed** — wrong-diagonal picks land hundreds of
  seconds off. `--fp-placement-gate-s 90` trusts fp only as a local refinement
  of MERT (BB12: median 30.5→6.6 s, p90 78→61 s; gate helps median AND tail).
- **Acappella set_start needs HuBERT** (`--stem-placement`): full-mix fp is weak
  on vocals; `place_joint` votes the diagonal in mix_vocals (BB12 <15 s
  61→76%, p90 96→72 s; known regression <4 s 44→34% — confidence floor TODO).
  Refines set_start ONLY; joint ref_start stays repeat-ambiguous.
- **Segment-list output:** 63% of GT spans are non-straight (multiseg/loop/
  odd-ratio); the aligner emits per-span `ref_segments` and `trajectory_acc`
  scores every class. Headline = multiseg+loop fiber-aware.
- **Fibers are HuBERT+silence-gate, never chroma** (`ref_fibers`); externally
  validated precise-but-low-recall (SALAMI P .88 / R .06) — that under-merge is
  why `fiber_gate` doesn't transfer; multimodal fibers are the open lever.
- **Old checkpoints:** any MERT head trained before the 2026-06-11 GT
  regeneration (a450005) learned ~0 ref offsets — retrain, don't reuse.

## Commands

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.train --eval --train-mert
venvs/audio/bin/python -m workspaces.alignment_prototype.infer --set-id 2nvzlh2k --band-s 45
venvs/audio/bin/python -m workspaces.alignment_prototype.joint_ref_decode --set-id 2nvzlh2k
# fingerprint backfill (corpus-wide, done) + per-set hit cache:
venvs/audio/bin/python scripts/backfill_track_fingerprints.py --dry-run
venvs/audio/bin/python scripts/cache_set_fingerprint_hits.py --set-id 1fsnxchk
# UnmixDB pretrain (external/unmixdb.py loader):
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain --dry-run --unmixdb-root ~/data/unmixdb-v1.1
```

## Human review loop (predictions → GT)

1. `infer` writes `out/<set_id>_predicted_timeline.json`.
2. `render_review_snippets --set-id <id>` → per-span A/B clips +
   `out/review/<set_id>/review.html` (keyboard verdicts, worst-first; ~30–40 min
   for ~150 spans).
3. `seed_als_from_timeline --set-id <id>` → pre-seeded Live project, **stamped
   `<SET> SEEDED.als`** (hard-refuses `* align.als`; locator #1 marks it
   machine-predicted). Round-trips its own output through `labeling/als`.
4. Human corrects → `labeling/export_als_to_gt.py` → new GT; diff vs the
   predicted timeline = honest scorecard. Requires a pull via
   `labeling/pull_set_for_alignment.py` (slot-spine fix ed7f121).

**Seeding-for-labeling is DEAD as a use case (John, 2026-07-06).** John
hand-labels every set in his own convention (clean session, varying master
tempo). Never pitch seeded sessions as labeling acceleration. The seeder's only
role is review-loop rendering. Master-tempo *emission* is deferred but has an
eventual consumer — the product-grade `.als` output (north-star deliverable:
hand-convention session, varying master tempo, unwarped mix);
`labeling.als.tempo_sec_to_beat` (property-tested) is its core primitive.

## Not wired yet / scoped

- **Multi-set co-train (SCOPED 2026-07-02, not small):** `train.py --yaml` is
  single-set (`_run_mert_eval` binds one set's stores by `gt.set_id`;
  `SpanTarget` has no set tag). Real design: SpanTarget += set_id, per-set store
  map, batches routed by tag. Gated on a third COMPLETE GT set (BB10/Murph not
  started). Don't bolt a concat hack.
- **Acappella ref-offset instance selection** — the biggest modelling prize
  (34% of all loss; six decode-layer threads dead, see looptrace/NOTES.md).
  Live lever: learned selector over {HuBERT diagonal evidence, fiber
  μ/ambiguity, fp sharpness}; needs the third GT set for leave-one-set-out.
- Acappella set_start p90 tail; HuBERT confidence floor (the <4 s regression).
- Per-stem instrumental set_start (chroma fails on instrumental presence;
  GT n=5 can't validate).
- B3: `fiber_ambiguity`/μ not yet fed into live decode; learned fusion arbiter
  (C1/C2) gated on more GT.
