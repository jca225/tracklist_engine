# Handoff: Alignment Paper Effort — where we stand (2026-07-12)

**For:** the next session. Goal = a rigorous conference paper that **re-characterizes
DJ-mix alignment**, backed by a 3-legged dataset and honest numbers.

## The thesis (locked)

Alignment is **not a scalar** but three near-orthogonal axes —
**identity / placement / structure** — with different invariances, different
synthetic-vs-real difficulty, and different generalization. A single scalar on a
synthetic benchmark overstates progress and hides the binding constraint. Full
framing: [docs/alignment_recharacterization.md](alignment_recharacterization.md)
(wired as the interpretive frame from root CLAUDE.md + alignment_status.md).
**Paper is NOT a "we're SOTA" paper** — we beat *reproduced* NMF/DTW on UnmixDB but
that's supporting evidence, not the headline (SOTA claim not defensible; see recharacterization §4b/§6).

## Built + merged this session (on `main`)

- **Ablation harness** — `workspaces/alignment_prototype/experiments/` (matrix/store/
  run/report/cli) + `score_spans()` refactor + `make align-ablate`. One scorer,
  span-bootstrap CIs, per-cell sqlite store. Built subagent-driven, reviewed, merged.
  Later hardened: resilient to per-cell failure (`4232ae7`).
- **Docs SSOT** — [docs/alignment_status.md](alignment_status.md) (numbers regenerated,
  BB11/BB12 set-id swaps fixed corpus-wide).

## Evidence in hand (real, harness-regenerated)

- **BB ablation** (`experiments/results/scores.db`, 9/10 cells; BB12-legacy pending
  pi-storage — non-blocking, BB11 covers C4):
  - identity BB11 85% / BB12 84%; placement median 4.8–5.3 s, p90 48–51 s.
  - per-axis trajectory strict→fiber; **fiber−strict gap +19–27 pp** (structure residual).
  - loss: placement 37% ≈ structure 38% (co-equal walls).
  - **ml learned decoder, LEAVE-ONE-SET-OUT: +4.7pp [+2.1,+7.4] (train BB12→BB11) /
    −2.1pp [−6.5,+2.1] (train BB11→BB12)** — the scientific heart: lever works one
    direction, hurts the other; n=2 → generalization unstable. This is leg 3.
- **UnmixDB** (n=141, `.superpowers/sdd/unmixdb_bounded.log`): ours `fused_resample`
  5.4 s set_start MAE vs reproduced NMF 20.2 s; tempo 0.045 vs 0.10; identity fp 73% /
  chroma 38% rank@1. Weak stratum: resample+compressor (~24 s, all methods).

## The 3-legged dataset (the plan)

1. **UnmixDB** — existing synthetic, easy axis (warp/placement). Done.
2. **Big Booties** (BB11=`2nvzlh2k`, BB12=`1fsnxchk`) — real, all axes, the walls. Done.
3. **Controllable synthetic** — NEW, isolates the hard dims OFAT. **Design proposed,
   pending John's approval:**
   [docs/superpowers/specs/2026-07-12-synthetic-structure-benchmark-design.md](superpowers/specs/2026-07-12-synthetic-structure-benchmark-design.md).
   Knobs (all 4): repeat-count / re-pitch / medley-density / entry-point, OFAT from a
   validated baseline, oracle-identity, scored on structure (score_spans/fiber).
   Heavy reuse of `synthetic_mix/`. Resume by approving that design → spec → build.

## Open framing forks (recharacterization §6, unresolved)

1. **Lean on generalization (leg 3, n=2)?** → submit-now (honest-preliminary) vs
   hold for BB10 (3rd GT set, in backfill) to make it a full leg.
2. Object = the problem (rec.) vs our system front-and-center.
3. Title claim — placeholder: *"Synthetic benchmarks have made DJ-mix alignment look
   solved; decomposed, real-mix evaluation shows the hard problem is structural, not
   spectral."*

## Immediate next step

Approve/adjust the synthetic-benchmark design → writing-plans → subagent-driven build
(mirror the ablation-harness flow). Then draft the 8-page paper on the 3-legged
evidence. Paper draft was deliberately deferred until numbers are real (John's call).

## Loose ends

- **pi-storage down** since ~2026-07-11 late (SSH timeout) → BB12-legacy cell + any
  fresh infer blocked. Retry when back.
- SDD ledger + run logs for this session: `.superpowers/sdd/` (gitignored scratch).
