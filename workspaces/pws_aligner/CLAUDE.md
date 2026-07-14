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

## Sensor phase still frozen

No new probes here. This module aggregates the existing channel inventory.
