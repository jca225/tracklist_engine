# Kernel + data engine execution plan — 2026-07-09 → Aug 1

The execution layer under
[architecture_north_star.md](architecture_north_star.md) (the map) and
alongside [entropy_reduction_plan.md](entropy_reduction_plan.md) (contracts +
data gaps; workstream letters there are unrelated to the W-numbers here).

## The composed frame (settled 2026-07-09)

Two abstractions, deliberately layered:

- **The estimation kernel** (the soul): the aligner is a state-estimation
  system — GPS for a DJ set. Probes are *sensors emitting likelihoods with
  calibrated precisions*, the warp prior is the *motion model*, identity under
  repeats is *data association* (the acappella wall — the classically hard
  part), fibers are *loop closures*, `path_decode` is a *MAP smoother*,
  abstention is *posterior entropy*, the agentic loop is *active sensing*.
- **The OS shell** (the body): contracts=ABI, drivers=kernel interface,
  probes=device drivers, artifact cache=page cache, agentic loop=scheduler,
  race board=perf counters. Shell work is mechanical and never blocks the
  model lane.
- **The data engine** (the metabolism): the loop that grows supervision.
  Prior art adopted (see sources at bottom): Snorkel's
  learn-accuracies-without-GT + correlation correction, Tesla's
  trigger-mining + shadow mode (≈ our race board), Waymo's offboard/online
  labeler split, FixMatch's confidence-gate discipline + consistency
  regularization, core-set dedup from the curation tooling.

Lanes: **[K]** kernel/infra (this plan's owner), **[M]** model (parallel
agent: `trajectory/` training + `drivers/ml`), **[J]** John. The race board
is the only coupling point between [K] and [M]; interface changes ship as
*parallel emissions*, never in-place rewrites of what [M] consumes.

## W0 — the substrate (pi-storage / pi-worker / Vast / Mac)

Added 2026-07-09 after review: the original draft was Mac-centric. The
cluster is not a deployment detail — it is the memory hierarchy of the OS
map, and two kernel correctness items live there.

**Storage tiers (the page-cache analogy, made literal):**

| tier | role |
|---|---|
| pi-storage (`/mnt/storage`) | **origin / disk** — canonical DB, track audio objects, stems, Essentia models. The only source of truth. |
| Mac caches (`mert_store`, `.cache/fp_*`, whisper, `~/aligning/`) | **page cache / RAM** — everything here must be re-derivable from origin + code; W3's content-hash keys are machine-independent so a feature computed anywhere is a hit everywhere. |
| Vast box | **ephemeral compute** — mounts origin over sshfs, writes results back, dies. Never holds unique state. |

**Rules:** (1) any W4/W5 artifact that trains a model (pseudo-labels, queue
state, audited timelines) gets a canonical home on pi-storage (files or DB
table), never only a Mac `out/` dir; (2) the repo's local `data/db` copy is
never consulted (existing rule, restated because the engine raises the
stakes); (3) compute placement: offboard labeler GPU stages on Vast/Mac-MPS,
CPU stages (beats, chroma) can drain to pi-worker idle.

**Kernel-blocking substrate items (do first):**

- **W0.1 — deploy + re-materialize.** The `claimed_stem` row-text fix
  (888caca/f678f3a) is deployed as *code* but the canonical
  `set_track_slots` rows were never regenerated — the DB still shows ~2
  instrumentals/set vs ~25 real. Until `python -m tokenizer.materialize`
  re-runs on pi, the instrumental placement channel cannot route on
  *unlabeled* sets (the `--instr-stem-gt-yaml` workaround only works where
  GT exists — exactly where we don't need it). Procedure: backup DB →
  `make deploy` → materialize → verify counts. This unblocks W1's last
  default flip.
- **W0.2 — stems coverage as a boot gate.** The aligner needs mix stems +
  ref stems from ingest/analysis (pi/Vast loops). W1's inventory preflight
  reads pi state; a set whose stems aren't ready fails at boot with a
  provenance-grade message, not mid-decode.

## Workstreams

### W1 — kernel v1: defaults, determinism, one command [K] — week of 7/9

1. Burn the proven winners into the classical driver as **defaults, not
   flags**: stem-routed placement, `--fp-placement-gate-s 90`, `--lt-stems
   acappella,instrumental`, axis-priority fusion, segment-list output. A flag
   that has been default-on ≥2 weeks becomes code; its flag dies.
2. New guardrails ratchet: argparse-flag count on the kernel path
   (`infer`, `joint_ref_decode`, `drivers/`) only goes down.
3. Determinism check: `make align SET=<id>` twice → byte-identical timelines
   (pin seeds, sort any set-iteration); add as a test.
4. Inventory preflight: `SetContext.for_set` runs the `check-inventory` gate
   before any decode (boot-time validation, not a separate manual step).

**Exit:** fresh clone → `make align SET=<id>` → current-best timeline;
zero flags; two runs identical; flag-count ratchet live.

### W2 — estimation contracts: factors, posteriors, calibration [K, interface negotiated with M] — weeks 1–2

1. **`ProbeFactor` record** (contracts): probe id+version, domain pair,
   support window, likelihood (Gaussian `(mu, sigma)` or sampled curve),
   abstain. Probes emit factors **alongside** their current decisions —
   non-breaking; `merge.py` keeps working while fusion v2 races it.
2. **Span posterior** in the Timeline schema: weighted alternatives (the
   fiber instance set IS the hypothesis set for repeat-ambiguous spans) +
   machine-readable abstain reason. The contracts decoder is already
   tolerant of new fields — additive change.
3. **Fusion v2**: precision-weighted combination with Snorkel-style
   correlation correction, precisions learnable from *agreement structure on
   unlabeled sets* (the no-GT unlock). Validate against LOSO BB11/BB12 before
   trusting any no-GT estimate — fp's 0.90→0.53 collapse is the cautionary
   baseline.
4. **Calibration column on the race board**: coverage-risk / expected
   calibration error next to accuracy. "Abstain, never lie" becomes a number.

**Exit:** fusion v2 ≥ `source_priority` on both sets held-out AND calibrated
(stated confidence matches empirical accuracy); board shows calibration.

*Candidate successor (proposed, unscheduled): a **learned** span-level critic
trained on GT positives + perturbation negatives + synthetic renders, racing
fusion v2 at the same 0.75-AUC bar — see
[learned_critic_plan.md](learned_critic_plan.md). Also touches W4 (audit
tiers) and W5 (auto-accept gate, verification labeling).*

### W3 — page cache + provenance [K] — week 2, parallel with W2

`core/artifacts.py`: content-hash keys (audio hash × extractor version ×
params), provenance fields (`produced_at`, producer commit, input
fingerprints), `is_stale()` loud on load. Adapt `mert_store`, fp caches
(`.cache/fp_instr/`), whisper transcript caches onto it; delete the silent
disk-truth fallbacks the staleness bug class grew (18 fix-commits).

**Exit:** zero silent fallbacks (ratchet); a feature computed by one driver
is a cache hit for every other; stale artifacts fail loudly with provenance.

### W4 — the offboard labeler (the Waymo move) [K, consumes M's decoder when it wins] — weeks 2–3

Make the expensive oracle a **production component, not an eval**: an offline
global decoder with unconstrained compute — full-set joint decode, past+future
context, cross-span + fiber-level consistency constraints, hours per set if
needed — whose output is audited by `als_audit` and graded into quality tiers:

`hand GT > audited-offboard > agentic auto-accept > synthetic render`

Training consumes tiers with weights; the online kernel learns to match the
offboard labeler, exactly as Waymo's onboard models train on offboard labels.

**Exit:** offboard beats the online kernel on both GT sets by a clear margin,
and produces audited timelines for ≥2 sets that have **no hand GT**.

### W5 — the data engine loop [K+M+J] — week 3 onward

1. **Triggers** (Tesla): abstention + **driver disagreement** (classical vs
   ml on the same span — computed today, thrown away today) + calibration
   outliers → items on a queue.
2. **The queue**: worst-first (review-UI pattern), **core-set deduped via
   fibers** (never ask John to label 30 instances of the same chorus
   ambiguity). John burns it in minutes/day; BB10 + Murph GT arrive this way
   — slot-sized active labeling, not set-at-a-time. (Replaces nothing John
   does by hand-convention; it only *orders* what to label next.)
3. **Pseudo-label pool**: tiered per W4; retrain cadence per cycle;
   auto-accept threshold FROZEN until the calibration column exists (FixMatch
   error-propagation discipline — confident-but-wrong compounds).
4. **Free consistency signal**: same synthetic GT mashup re-rendered with
   different warps/gains must decode identically; disagreement is a training
   loss with zero labeling cost.
5. **No-GT precision estimation** (W2.3) unlocks pseudo-labeling corpus sets
   beyond the BB family.

**Exit:** one full cycle — mine → label/pseudo-label → retrain → re-race —
lifts held-out transfer with no hand-built dataset. That's the flywheel
turning under its own power.

### W6 — regression net + promotion gate [K] — background

CPU-cheap golden subset (contract round-trips + scorer goldens) in CI on every
push; full `make race` nightly on the Mac; ratchets extended (flags, silent
fallbacks). Promotion checklist for `workspaces/alignment_prototype` →
`alignment/` (architecture doc P6): P1–P4 exits held on two consecutive new
sets.

## Timeline to Aug 1

| week | [K] | [M] (parallel agent) | [J] |
|---|---|---|---|
| 7/9 | W1 done; W2 factor record designed, handoff note to [M]; W3 started | trajectory set-split training (live now); race iterations | — |
| 7/16 | W2 fusion v2 + calibration column; W3 done | ml driver races with posterior output shape from W2 | B1 (two commands, entropy plan) |
| 7/23 | W4 offboard v1 audited; W5 triggers + queue wired | consumes offboard tiers for training | first queue burn-downs (BB10 slices, minutes/day) |
| 7/30 | W5 first full cycle | P3 gate attempt: ml as default if board says so | queue cadence |

**Aug 1 ship line** (what "done" means if everything slips a week):
`make align` kernel v1 + race board with calibration + offboard labeler v1.
The north-star deliverable (tracklist+audio → round-trippable `.als`) is
satisfiable by kernel v1 alone; W4/W5 determine whether the *learned* kernel
gets there too. **Stretch:** P3 default-flip — [M]'s gate, not a promise.

## Risks, named

- **Interface churn vs [M]**: mitigated structurally — factors/posteriors are
  parallel emissions; nothing [M] reads changes until fusion v2 *wins the
  race*. Coordination artifact: a short handoff note in
  `workspaces/alignment_prototype/looptrace/NOTES.md` when W2.1 lands.
- **Pseudo-label error propagation**: tiered weights, frozen auto-accept
  threshold until calibration exists, per-cycle held-out check with a revert
  rule (any cycle that drops transfer gets its pool quarantined).
- **Correlated probes poisoning precision estimates**: Snorkel correlation
  correction + LOSO validation gate before any no-GT precision is trusted.
- **Determinism vs MPS nondeterminism**: if bit-identity is unreachable on
  GPU paths, the determinism test degrades to tolerance-based span equality —
  stated, not silent.
- **[J] bandwidth**: the queue is minutes-per-day by construction; if it
  isn't, the dedup is failing — that's a bug, not a John problem.

## Non-goals

Everything in the architecture doc's list, plus: no fusion-v2 default flip
without the race board, no new probes (the freeze holds — factors re-expose
*existing* sensors), no schema-breaking Timeline changes (additive fields
only until P6).

## Prior art (research base, 2026-07-09)

Snorkel weak supervision ([arXiv](https://arxiv.org/abs/1711.10160),
[VLDB](https://link.springer.com/article/10.1007/s00778-019-00552-1)) — probes
ARE labeling functions; Tesla data-engine loop + triggers + shadow mode
([survey](https://arxiv.org/html/2401.12888v1),
[Mcity Data Engine](https://arxiv.org/pdf/2504.21614)); Waymo offboard
auto-labeling ([arXiv](https://arxiv.org/abs/2103.05073));
FixMatch ([NeurIPS](https://papers.nips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf))
+ self-training surveys ([arXiv](https://arxiv.org/html/2202.12040v6));
curation/label-error tooling (FiftyOne, Lightly, Cleanlab —
[overview](https://encord.com/blog/active-learning-machine-learning-guide/)).
