# The aligner as a POMDP: an agentic harness with an explicit action space

**Status:** design (2026-07-01, revised w/ agent-architecture grounding). Reframes the aligner
from a fixed feed-forward pipeline into a **partially-observed decision process**: gather
information to resolve ambiguity, reconsider, repeat — until confident enough to commit or
abstain. Architecture patterns adapted from Liu et al. 2026 (arXiv:2604.14228, "Dive into
Claude Code") — see "Architecture grounding" below. Build still gated on the BB11 held-out number.

## Why POMDP is the right abstraction (not a stretch)

The aligner already *is* a POMDP — un-named and hand-wired. The true alignment of a span
(song / where-in-mix / where-in-ref / warp) is a **hidden state** we never observe
directly; every probe is a **noisy observation** of it at a **cost**; and the pipeline
already makes a sequence of decisions (which probes, how to fuse, when to abstain). Making
the POMDP explicit buys three things a fixed pipeline can't:

1. **Value of information.** Probes have wildly different costs — fp is cached/instant,
   HuBERT on an hour mix is minutes, stem separation is expensive, a human label is very
   expensive. A fixed pipeline runs a hand-tuned subset everywhere. A policy runs the next
   probe **only where the belief is still uncertain** — cheaper *and* more accurate.
2. **Principled abstain / escalate-to-human** — the open-set endgame. "Ask human" is an
   action with a cost; the policy invokes it only when its expected value beats the cost.
3. **One decision-maker** replaces the hand-set source-priority fusion arbiter.

## The formal pieces (mapped to what exists)

| POMDP element | Concretely in this system |
|---|---|
| **Hidden state** `s` | per span: (recording_id, set_start, ref_start, warp/tempo, gain, is-out-of-vocab) |
| **Belief** `b(s)` | per-span candidate placements + confidences from the probes; precision/abstention (WS1) is the belief-quality signal |
| **Observations** `o` | probe outputs: fp votes, chroma/HuBERT matched-filter curves, lyrics diagonals, **reconstruction match/margin** — each with a calibrated reliability |
| **Actions** `a` | see action space below |
| **Transition** | trivial for placement (state is static per set); "transition" is really belief update after an observation |
| **Observation model** `P(o\|s)` | **each probe's calibrated precision** — this is exactly what the reconstruction- and lyrics-grader precision tests measure. Building these IS building the POMDP's sensor model. |
| **Reward** `R` | + correct placement (held-out GT) − compute cost(action) − human cost |

## The action space

Info-gathering (reduce uncertainty), each with a cost tier:

| action | module (exists) | cost |
|---|---|---|
| `run_fp(span)` | `mix_fp_hits` / `fp_placement` | ~0 (cached) |
| `run_chroma(span)` | `refine_ref_offsets` | low |
| `run_hubert(span)` | `stem_placement` / `similarity_probe` | med (MPS minutes) |
| `run_lyrics(span)` | `lyrics_align` (cached transcripts) | low if cached, else GPU |
| `run_reconstruction(span)` | `recon_rerank` / `recon_probe` | low–med |
| `separate_stem(region)` | Roformer/Demucs | high |
| `run_fibers(ref)` | `ref_fibers` | med |

Belief-shaping / commit:

| action | effect |
|---|---|
| `re_decode(constraint)` | re-run monotonic decode with an added constraint (force order, exclude a candidate, pin a span) — the "reiterate" |
| `commit(span, placement)` | freeze a span's placement (terminal for that span) |
| `abstain(span)` | emit nothing / flag — resting state for out-of-vocab or irrecoverable spans |
| `escalate(span)` | request a human label (most expensive; feeds active-labeling) |

## The "resolve and reiterate" loop

```
for each set:
    b = init_belief(cheap probes: fp + identity model, over all spans)   # cheap first
    while not done and budget_left:
        span = argmax_uncertainty(b)                 # worst-believed span
        mode = permission_mode(b[span])              # precision -> autonomy level (below)
        if mode == AUTO and confident(b[span]):
            commit(span); log(); continue            # 90%-clean gate -> auto-commit
        if exhausted(span):
            abstain(span) or escalate(span); continue
        a = policy(b[span])                          # value-of-information choice
        if not permitted(a, mode): a = downgrade(a)  # bounded autonomy / consent check
        o = run(a); log(a, o)                        # execute + append to audit log
        b = update(b, o)                             # precision-weighted belief update
    return {auto-committed} + {review-queue} + {escalated to human}
```

`argmax_uncertainty` + `policy` + `permission_mode` are the three brains. The loop spends
expensive actions only on the hard residual, gates autonomy by confidence, and logs every
action to an immutable event trail.

## Architecture grounding — proven agent-system patterns

Adapted from Liu et al. 2026 ("Dive into Claude Code: The Design Space of Today's and Future
AI Agent Systems", arXiv:2604.14228), which reverse-engineers a mature agent's architecture.
We *don't* copy it (that agent is real-time + human-watching; our aligner is BATCH over 20k
sets with no human per span) — we adapt the load-bearing patterns.

### The big one: a **permission-mode ladder** driven by our measured grader precision

Their seven permission modes + ML permission classifier map exactly onto **precision-graded
autonomy**. The grader-precision work this session IS the permission classifier — each span's
confidence picks its autonomy level:

| span confidence (from calibrated graders) | mode | aligner behavior |
|---|---|---|
| ≥90%-clean gate (lyrics-margin / fp-sharpness cleared this) | **auto-commit** | write pseudo-GT, log only |
| mid (recon ~78%, or single-probe agreement) | **execute-with-review** | place + flag into a review queue |
| conflicting probes / low margin | **suggest** | propose top candidates, don't commit |
| out-of-vocab / all probes abstain | **escalate** | send to the human labeling queue (active-labeling) |
| ambiguous but resolvable | **sandbox** | try re-decode variants in isolation, keep the best |

This is the concrete upgrade over "binary auto/escalate": a **ladder** where the precisions we
measured (lyrics 90%, fp 90%, recon 78%) *set the rungs*. Bounded autonomy + explicit consent
+ human-in-loop, adapted to batch = an **escalation queue** the human drains later, not
real-time consent.

### The other patterns, mapped

- **Central loop + permission check** (their observe→reason→act→check→update) — added above.
- **Context compaction pipeline** (5 layers) — the belief→LLM-policy interface: raw probe
  curves → keep top candidates → fuse repeats/fibers → compact numeric summary → inject into
  the LLM prompt. This is how a span's evidence becomes legible to the LLM policy cheaply.
- **Extensibility (MCP / plugins / skills / hooks)** — the **action registry** (increment 2)
  should mirror this: each probe registers via a uniform schema (MCP-style) so new probes plug
  in without touching the loop; **skills** = composite actions ("resolve vocal span" = lyrics →
  if abstain, HuBERT → decode); **hooks** = pre-action cost check / post-action logging.
- **Append-oriented session storage** — the audit log is an **immutable event log**; belief
  state = replay of events. Gives reproducible debugging, resume-from-any-point (matches the
  repo's `.als` round-trip + propose-don't-write ethos), and honest provenance for every
  pseudo-label (which probe, what confidence, auto vs human).
- **Reversibility / clear ownership** — nothing destructive: placements are proposals until a
  human confirms; final GT ownership stays with the labeler. Reinforces existing repo values.
- **Subagent delegation with permission inheritance** — for 20k scale, fan out per-set to
  subagents that inherit the confidence gates; cross-cutting logging/safety across the tree.
  (Coordinates with the existing parallel-aligner-agent conventions.)

## Action-space optimization

The action space is small (~10 actions), so the wins come from **design, not learning**
(learning the selection is the n=2 reward wall again). Five levers, in value order:

**1. Context-condition the action set — dominance pruning (biggest win).** The available
actions are a function of the span's stem/context; we've *measured* which actions are
dominated where, so the policy never wastes a call (or an LLM token) on one we proved won't help:

| span type | run first (cheap+high-info) | pruned (dominated here) | escalate to |
|---|---|---|---|
| host / regular | **fp** (cached, 90% clean where sharp) | lyrics, reconstruction | recon as noisy fallback; HuBERT if needed |
| acappella / vocal | **lyrics** (cached, 90% clean) | **reconstruction (dead)**, chroma (re-pitch fails) | HuBERT if lyrics abstains |
| instrumental | fp / chroma | lyrics, reconstruction | stem-separation if still fuzzy |
| re-pitched (remix tag) | **HuBERT** (key-invariant) | chroma (dominated) | — |

**2. Cost-ordered value-of-information + early termination.** Order by info-per-cost;
run the cheap specialist first; **commit immediately if it clears the gate.** Most spans
should end after *one* action — the effective per-span action space collapses to ~1 for the
easy majority. Greedy cheapest-high-info-first is *provably near-optimal* (sensing under cost
is submodular).

**3. Macro-actions / skills.** Bundle common sequences into single skills
(`resolve_vocal_span` = lyrics → if abstain, HuBERT → decode). The policy picks a skill, not
a micro-action — fewer decisions, tractable + cheaper LLM policy.

**4. Restrict belief-shaping constraints.** `re_decode` takes only a small high-value set
(force monotonic order, exclude conflicting candidate, pin a confident neighbor) — never
arbitrary constraints (that's where a space explodes for no gain).

**5. Data-driven pruning from the audit log (v2).** Once the log accumulates, measure which
actions ever changed the belief / led to correct commits; delete dead actions, re-tune the
cost order empirically.

Every lever also **shrinks the LLM policy's decision surface** → cheaper, less hallucination.
Optimizing the action space *is* optimizing the policy over it.

## The policy — three tiers, and the reframe that matters

**The trap:** do NOT learn the policy with RL now. Reward needs GT; we have n≈2 labeled
sets. RL on n=2 overfits — the same data wall that killed synthetic pretrain and the fusion
model. A self-supervised reward (reconstruction) is noisy/biased (host-only, ~78%) →
reward-hacking. So the learned-RL policy is a *later* upgrade.

Three tiers, in build order:

1. **Heuristic policy (now).** A cost-aware decision rule: cheapest confident probe →
   commit if belief clears threshold → else the next-highest value-of-information probe →
   re-decode → abstain/escalate after budget. This is a POMDP-shaped harness with a
   hand-written brain. Fully buildable from existing modules; deterministic; cheap.

2. **LLM-as-policy (the reframe — the interesting middle).** Use an LLM as the policy that,
   given a span's belief summary ("fp says t=612s conf 0.4; lyrics says t=640s margin 0.1;
   they disagree; HuBERT not run"), **chooses the next action** and adjudicates conflicts.
   Why this dodges the trap: an LLM is a strong **zero/few-shot** policy — it brings a
   reasoning prior, so it needs **no RL on n=2**. Cost is controlled by **only invoking it on
   the ambiguous residual** (the cheap probes resolve the easy ~80%; the LLM adjudicates the
   hard ~20%) — value-of-information applied to the policy itself. This is likely the
   endgame the "agentic harness" intuition is pointing at: not a small trained model, but an
   LLM with a **tightly specified action space** and calibrated probe observations as its
   context. The harness's job is to make the action space and belief legible to the LLM and
   to *execute* its chosen actions deterministically.

3. **Learned RL policy (later).** Once GT scales (more labeled sets) or a trusted
   self-supervised reward exists, distill the heuristic/LLM policy into a cheap learned one.

**Recommendation:** heuristic policy first (deterministic substrate + validates the loop),
then LLM-as-policy on the hard residual. The heuristic is the safety floor and the LLM's
fallback; the LLM lifts the hard cases the heuristic abstains on.

## Honest costs and mitigations

- **LLM latency/cost at 20k×~150 spans.** Mitigation: gate LLM to ambiguous spans only;
  batch; cache decisions by belief-signature. If still too costly, the heuristic handles the
  bulk and the LLM only the top-uncertainty tail.
- **Reward/observation-model quality is the ceiling.** The policy is only as good as the
  probes' *calibrated reliability*. Garbage sensor model → garbage value-of-information.
  This is why the grader precision work (reconstruction host, lyrics vocal) is a prerequisite,
  not a side quest.
- **Nondeterminism (LLM).** Keep the heuristic as the deterministic backbone; log every
  action + belief for reproducibility; the LLM proposes, the harness executes and records.
- **Over-engineering risk.** A fixed pipeline already gets 6.6s median in-domain. The POMDP
  earns its keep on (a) cost efficiency at 20k scale and (b) the hard/held-out residual and
  abstain/escalate — NOT on the easy in-domain spans. Measure it there.

## How it sits on what we've built

- **Execution substrate** already exists: `run_recon_experiment.py` is a baby version — run
  a step, score, compare. Generalize it into the belief/action loop.
- **Observations** already exist as modules (fp, chroma, HuBERT, lyrics, reconstruction).
- **Observation model** is what the grader precision tests produce (reconstruction host,
  lyrics vocal) — feed each probe's precision curve in as `P(o|s)`.
- **Belief/abstention** is WS1 precision fusion — the belief-quality signal.
- **Reward / eval** is `score_timeline_vs_gt` held-out.

## Build increments (revised per the agent-architecture grounding)

1. **Belief object + append-only event log** — a typed per-span belief (candidate placements
   + per-probe confidence + precision) that every probe writes into; belief = replay of an
   immutable action/observation log (audit trail, resume, provenance). (New: `belief.py`.)
2. **Action registry (MCP-style) + skills + hooks** — each probe/decode/separate registers via
   a uniform `Action(run)->Observation` schema with a **cost tag**; composite **skills**
   (e.g. `resolve_vocal_span`); **hooks** for pre-action cost check / post-action logging.
   New probes plug in without touching the loop. (New: `actions.py`.)
3. **Permission classifier + heuristic loop** — the precision-graded autonomy ladder
   (auto-commit / review / suggest / escalate / sandbox) driven by the calibrated grader
   confidences; `resolve()` drives the loop with the permission check. Deterministic; validate
   held-out on BB11 vs the fixed pipeline (does selective probing + auto-commit match/beat it
   cheaper, with a clean review queue?).
4. **LLM-as-policy** — swap the heuristic's action choice for an LLM call on ambiguous spans;
   the compacted belief summary is its context, the action registry its tool schema. Heuristic
   stays the deterministic fallback.
5. **Escalate → active-labeling queue** — the batch-adapted "human-in-loop": escalated spans
   ARE the highest-value sets/spans to hand-label (closes the loop with Steps 4–5 of the
   reconstruction plan). The human drains the queue; the aligner never destructively commits GT.

## Related

`docs/reconstruction_supervision_plan.md` (the graders = observation model + permission
classifier), `project_ws1_precision_fusion` (belief/abstention), `project_open_set_alignment_endstate`
(abstain/escalate endgame), `project_dj_selection_model` (the downstream generation step).
Architecture patterns: Liu et al. 2026, arXiv:2604.14228 ("Dive into Claude Code").
