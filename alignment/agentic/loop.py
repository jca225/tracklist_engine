"""resolve() — the heuristic decision loop (design doc's tier-1 brain).

Per span, run the stem's dominance plan cheapest-first; after every
observation, re-check the ladder and TERMINATE EARLY on auto-commit (most
spans should end after one informative probe — that's where the compute
saving lives). A global budget bounds total action cost; spans are visited
worst-belief-first so remaining budget goes to the hardest spans. Every step
appends to the event log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataclasses import replace as _replace

from alignment.agentic.actions import Runner, bind, plan_for
from alignment.agentic.belief import (
    SpanBelief,
    independence_satisfied,
)
from alignment.agentic.events import EventLog
from alignment.agentic.policy import Ladder, Mode


@dataclass(frozen=True)
class SpanCtx:
    """Everything a runner may need about one span (probe adapters read this)."""

    slot_label: str
    recording_id: str
    claimed_stem: str
    data: dict = field(default_factory=dict)  # timeline row / caches / paths


@dataclass(frozen=True)
class Resolution:
    committed: dict[str, SpanBelief]
    review: dict[str, SpanBelief]
    suggested: dict[str, SpanBelief]
    escalated: dict[str, SpanBelief]
    cost_spent: float
    actions_run: int

    def summary(self) -> str:
        n = (
            len(self.committed)
            + len(self.review)
            + len(self.suggested)
            + len(self.escalated)
        )
        return (
            f"{n} spans: auto={len(self.committed)} review={len(self.review)} "
            f"suggest={len(self.suggested)} escalate={len(self.escalated)} "
            f"| {self.actions_run} actions, cost={self.cost_spent:.1f}"
        )


def resolve(
    spans: list[SpanCtx],
    runners: dict[str, Runner],
    log: EventLog,
    *,
    ladder: Ladder | None = None,
    budget: float = float("inf"),
    model=None,  # PrecisionModel | None — learned selection + calibrated precision
    rng=None,  # numpy.random.Generator, required when model is given (reproducible)
    require_independence: bool = False,
) -> Resolution:
    """Heuristic loop; when ``model`` is supplied, action *order* comes from the
    model's Thompson sampling and each observation's precision is overwritten
    with the model's calibrated posterior mean. Without a model, behavior is the
    unchanged static-plan deterministic default.

    ``require_independence`` (E1 pseudo-safe): do not early-stop on AUTO_COMMIT
    until G2 independence is satisfied or the stem plan is exhausted. Prevents
    lyrics-alone (prec 0.76 ≥ auto 0.75) from committing before HuBERT/fp can
    corroborate — which would otherwise starve materialize at g2_independence.
    """
    ladder = ladder or Ladder()
    runners = bind(runners)
    if model is not None and rng is None:
        import numpy as _np

        rng = _np.random.default_rng(0)
    beliefs = {
        s.slot_label: SpanBelief(s.slot_label, s.recording_id, s.claimed_stem)
        for s in spans
    }
    ctx_by_key = {s.slot_label: s for s in spans}
    plans = {s.slot_label: list(plan_for(s.claimed_stem)) for s in spans}
    spent = 0.0
    n_actions = 0

    # Worst-belief-first scheduling: everything starts at quality 0, so the
    # initial order is plan order; after each pass we re-sort so remaining
    # budget concentrates on the least-resolved spans.
    open_keys = [s.slot_label for s in spans]
    while open_keys:
        open_keys.sort(key=lambda k: beliefs[k].quality())
        progressed = False
        for key in list(open_keys):
            b = beliefs[key]
            plan = plans[key]
            available = [
                a.name
                for a in plan
                if a.name in runners and a.name not in b.probes_run()
            ]
            auto = ladder.mode(b) is Mode.AUTO_COMMIT
            if auto and (
                not require_independence or independence_satisfied(b) or not available
            ):
                open_keys.remove(key)
                continue
            if not available:  # plan exhausted — final mode decides the queue
                open_keys.remove(key)
                continue
            if model is not None:
                # learned selection: Thompson-rank the available actions by
                # sampled precision/cost, pick the top (explores + exploits)
                nxt_name = model.rank(tuple(available), rng)[0]
            else:
                nxt_name = available[0]  # static cheapest-informative-first order
            nxt = next(a for a in plan if a.name == nxt_name)
            if spent + nxt.cost > budget:
                log.append(key, "note", {"msg": f"budget stop before {nxt.name}"})
                open_keys.remove(key)
                continue
            obs = runners[nxt.name](ctx_by_key[key].data)
            if model is not None and not obs.abstained:
                # ladder should weigh the CALIBRATED precision, not the runner's
                obs = _replace(obs, precision=model.precision(obs.probe))
            beliefs[key] = log.observe(b, obs)
            spent += nxt.cost
            n_actions += 1
            progressed = True
        if not progressed:
            break

    out: dict[Mode, dict[str, SpanBelief]] = {m: {} for m in Mode}
    for key, b in beliefs.items():
        mode = ladder.mode(b)
        top = b.best()
        log.append(
            key,
            "commit" if mode is Mode.AUTO_COMMIT else mode.value,
            {
                "quality": round(b.quality(), 4),
                "set_start_s": top.set_start_s if top else None,
                "probes": sorted(b.probes_run()),
            },
        )
        out[mode][key] = b
    return Resolution(
        committed=out[Mode.AUTO_COMMIT],
        review=out[Mode.REVIEW],
        suggested=out[Mode.SUGGEST],
        escalated=out[Mode.ESCALATE],
        cost_spent=spent,
        actions_run=n_actions,
    )
