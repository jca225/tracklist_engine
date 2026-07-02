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

from workspaces.alignment_prototype.agentic.actions import Runner, bind, plan_for
from workspaces.alignment_prototype.agentic.belief import SpanBelief
from workspaces.alignment_prototype.agentic.events import EventLog
from workspaces.alignment_prototype.agentic.policy import Ladder, Mode


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
) -> Resolution:
    ladder = ladder or Ladder()
    runners = bind(runners)
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
            if ladder.mode(b) is Mode.AUTO_COMMIT:
                open_keys.remove(key)
                continue
            plan = plans[key]
            nxt = next(
                (a for a in plan if a.name in runners and a.name not in b.probes_run()),
                None,
            )
            if nxt is None:  # plan exhausted — final mode decides the queue
                open_keys.remove(key)
                continue
            if spent + nxt.cost > budget:
                log.append(key, "note", {"msg": f"budget stop before {nxt.name}"})
                open_keys.remove(key)
                continue
            obs = runners[nxt.name](ctx_by_key[key].data)
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
