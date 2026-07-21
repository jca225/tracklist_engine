# Identity → 100%: the two structural levers

> **For agentic workers:** this is a **phased design/plan for the trajectory-decoder
> session**, not a verbatim-code plan. The fixes touch decoder internals
> (`infer.py` / `path_decode.py` / `joint_ref_decode.py`) and the scorer's output
> contract (`score_timeline_vs_gt.py`) — regions this plan's author did **not**
> read line-by-line. Phase 0 is mandatory diagnosis; do NOT skip to
> implementation. Execute Phase 0, then re-plan Phases 1–2 with
> `superpowers:brainstorming` + `superpowers:writing-plans` inside that session.

**Goal:** raise identity from ~84% toward ~100% by fixing the two mechanisms that
cause 95% of identity-miss-seconds — **span segmentation** and **single-label
output** — neither of which is the identity model or data acquisition.

**Evidence base (read first):**
[eda/alignment/failure_analysis/IDENTITY_MISS_DECOMPOSITION.md](../../../eda/alignment/failure_analysis/IDENTITY_MISS_DECOMPOSITION.md)
(reproduce with `identity_miss_decompose.py`). The one-line finding: of 58 identity
misses, **69% are segmentation, 26% are layered transitions, 3% are true
discrimination, ~2% are boundary/acquisition.**

## Global Constraints

- **Numbers SSOT:** [docs/alignment_status.md](../../../docs/alignment_status.md).
  Do not hand-type identity/placement metrics elsewhere; regenerate from the scorer.
- **Sensor freeze is in force** (`workspaces/alignment_prototype/CLAUDE.md`): do NOT
  add new probes/channels/priors. Both levers here are *actor/output* changes, not
  perception — they are inside the two sanctioned lanes (decoder + contracts).
- **Axis rule:** take `claimed_stem` from the matched GT row, never the timeline span.
- **n = 2 GT sets.** Every claimed gain must survive LOSO / per-set inspection; a
  gain on n=2 that does not hold on both sets is not a finding.
- **The scorecard is the only arbiter** (`make scorecard`). Report identity strict
  before/after per set, and confirm no regression in placement / trajectory-acc.
- **Collision:** other live sessions own `workspaces/pws_aligner/**` (harvest+cache)
  and the `cotrain-corpus-harvest` branch. Stay out of both. Branch off `main`.

---

## Phase 0 — Diagnose before building (REQUIRED)

The decomposition tells us *what* mechanism dominates; it does not tell us *why the
decoder does it*. Answer these with measurement, not assumption, before writing any
fix. Deliverable: a short `PHASE0_FINDINGS.md`.

- [ ] **0.1 — Why are the 10 segmentation spans giant?** For each (BB12-heavy: 196 s
  over 11 GT tracks, etc.), determine the root cause. Predicted span extent is set
  per tracklist slot at `infer.py:187-188` (`set_end_s = start + dur`). Is `dur`:
  (a) a **coarse tracklist slot** (one 1001TL entry that is genuinely a medley →
  the fix is sub-slot segmentation), (b) a **placement collapse** (set_start/extent
  computed wrong so the span sprawls), or (c) a **slot-merge** upstream? Tabulate
  the 10 spans by cause. This decides whether Phase 1 is a decoder change or a
  tokenizer/slot change.
- [ ] **0.2 — Are the 26 layer/transition picks co-present or unrelated?** For each,
  check whether the picked recording is a GT track playing *nearby in time* (a
  boundary/placement bleed) vs a track not in the set at all (harder). This decides
  how much of bucket 3 is really placement (Phase 1) vs needs true multi-label
  (Phase 2).
- [ ] **0.3 — Per-set candidate-pool check.** Confirm the correct recording was in
  each miss's *set candidate pool* (`set_track_slots` → recording), not just
  globally downloaded, to rule out a tokenizer/materialize pool gap masquerading as
  discrimination. One canonical-DB query.
- [ ] **0.4 — Define the scoring contract change up front.** Decide, and write down,
  how a *layered* moment should score: e.g. a predicted span is identity-correct if
  its recording matches **any** GT row overlapping it (current rule already does
  this per-span) — the gap is that ONE predicted span cannot cover TWO simultaneous
  GT tracks. Specify the target output shape (below) so Phases 1–2 aim at a fixed
  contract.

**Gate:** present Phase 0 findings + the proposed output contract to the human
before Phase 1. The split from 0.1/0.2 may re-weight the whole plan.

---

## Phase 1 — Segmentation (the 69% lever)

**Hypothesis:** the biggest identity lever is cutting the giant single-label spans
so each track-region gets its own recording. This is the *same* decoder work that
attacks the placement/decode wall (~75% of total loss) — one fix, two payoffs.

Scope depends on Phase 0.1:

- If **coarse tracklist slots** dominate: the fix is **sub-slot segmentation** —
  within a long slot, detect track-change boundaries (the existing boundary-novelty
  / `surprise` signal and fp-diagonal breaks are already in the sensor inventory —
  reuse, do not add) and split the span, assigning a recording per sub-segment via
  the existing MERT identity + fp placement.
- If **placement collapse** dominates: the fix is in `infer.py` placement /
  `path_decode.py` extent — constrain span extent so it cannot sprawl across a
  region MERT identity does not support.

**Success metric (pre-registered):** the 10 segmentation spans (1690 s) shrink to
per-track spans whose identity matches GT; measured as identity strict ↑ on both
sets with **no placement/trajectory regression** on the scorecard. Target: recover
the majority of the 1690 s.

**TDD posture:** add a scorecard-level regression fixture asserting the specific
giant spans (e.g. BB12 196 s Porter/Third-Eye-Blind window) resolve to multiple
correct-identity sub-spans. Re-plan concrete tasks after Phase 0.

---

## Phase 2 — Layered / multi-label output (the 26% lever)

**Hypothesis:** dense transition/mashup moments have 2+ simultaneous GT tracks; a
single-recording-per-span timeline structurally cannot be 100% there. The repo's
stated design ("a mix moment is a sum of layers; align per stem channel AND full
mix, fused at decode") is **not realized in the output** — `infer.py` /
`harness/merge.py` carry no co-present/multi-label field.

Work items (design in-session, do not treat as final):

- [ ] **2.1 — Output contract:** extend the timeline span schema to carry
  co-present recordings (e.g. `layers: [recording_id, ...]` or emit overlapping
  spans per stem channel), consistent with the `.als`-round-trip north-star (a
  layered DAW session is layered anyway).
- [ ] **2.2 — Producer:** fuse the stem-wise channels (mix_vocals↔ref vocals,
  instrumental↔ref instrumental, full-mix) at decode into layered output rather
  than collapsing to one label (`harness/merge.py` `source_priority` currently
  picks one).
- [ ] **2.3 — Scorer:** `score_timeline_vs_gt.py` credits a GT row if **any**
  emitted layer at that moment matches — so a correctly-labelled co-present track
  is not scored as a miss. Keep strict single-label identity as a separate reported
  number so the two are not conflated.

**Success metric:** the 26 layer/transition misses (645 s) score correct once the
co-present track is emitted+credited, with the strict single-label number reported
alongside (no silent metric inflation).

---

## Explicitly out of scope (small levers — separate, cheap jobs)

- **Boundary micro-fragments (1%, 18 spans/28 s):** span-boundary snapping; fold
  into Phase 1 only if free, else defer.
- **True discrimination (3%, 4 spans):** more training data (the harvest's real
  role). Do not build a new discriminator for 4 spans.
- **Acquisition (1 span):** download the Porter Robinson "Shelter" recording
  (`tlp2853054` Rvmor gap). One-off ingest job.
- **Work-grouping population (0/18,812):** link sibling versions under one `work`.
  Data-model integrity + lets the scorer treat "right song, wrong version" as a
  near-miss. Separate agent; does not gate identity.

## Sequencing

Phase 0 → gate → Phase 1 (biggest, shared with the placement wall) → Phase 2. Phase
1 and the trajectory-decoder training work are the *same lane* — do them together in
that session, not as a separate track.
