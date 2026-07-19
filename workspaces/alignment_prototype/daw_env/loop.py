"""resolve_daw — place → listen → sense → iterate (Mode A / B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
from workspaces.alignment_prototype.agentic.events import EventLog
from workspaces.alignment_prototype.agentic.policy import Ladder, Mode
from workspaces.alignment_prototype.daw_env.actions import (
    CommitSpan,
    EscalateHuman,
    NudgeSetStart,
    RenderListen,
    SoloLayer,
)
from workspaces.alignment_prototype.daw_env.sense import sense_listen
from workspaces.alignment_prototype.daw_env.session import DawSession


@dataclass
class DawResolution:
    session: DawSession
    beliefs: dict[str, SpanBelief] = field(default_factory=dict)
    committed: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    steps: int = 0

    def summary(self) -> str:
        return (
            f"daw_react: committed={len(self.committed)} escalated={len(self.escalated)} "
            f"steps={self.steps} spans={len(self.session.spans)}"
        )


def _listen_window(
    geom_set_start: float, geom_set_end: float, pad_s: float = 2.0
) -> tuple[float, float]:
    return (max(0.0, geom_set_start - pad_s), geom_set_end + pad_s)


def _propose_nudge(belief: SpanBelief, geom_start: float) -> float | None:
    """Heuristic: if cue/mert observations disagree with geometry, nudge toward them."""
    top = belief.best()
    if top is None:
        return None
    delta = top.set_start_s - geom_start
    if abs(delta) < 0.5:
        return None
    # Cap step so one listen cycle does not teleport.
    return float(max(-8.0, min(8.0, delta)))


def resolve_daw(
    session: DawSession,
    *,
    mode: str = "a",
    log: EventLog | None = None,
    ladder: Ladder | None = None,
    max_steps_per_span: int = 3,
    budget_spans: int | None = None,
) -> DawResolution:
    """Run Ableton ReAct over session spans (worst duration-first as a proxy).

    Mode ``a``: commit when ladder says AUTO or after max steps.
    Mode ``b``: escalate_human when ladder says REVIEW / after max steps without AUTO.
    """
    if mode not in ("a", "b"):
        raise ValueError("mode must be 'a' or 'b'")
    log = log or EventLog()
    ladder = ladder or Ladder()
    beliefs = {
        slot: SpanBelief(slot, g.recording_id, g.claimed_stem)
        for slot, g in session.spans.items()
    }
    order = sorted(session.spans.values(), key=lambda g: -(g.set_end_s - g.set_start_s))
    if budget_spans is not None:
        order = order[: max(0, budget_spans)]

    out = DawResolution(session=session, beliefs=beliefs)
    bus = {
        "acappella": "mix_vocals",
        "instrumental": "mix_instrumental",
        "regular": "mix",
    }

    for geom0 in order:
        slot = geom0.slot
        belief = beliefs[slot]
        session.apply(SoloLayer(bus.get(geom0.claimed_stem, "mix")))
        # Seed cue prior once so nudge has an independent target (ladder trust floor).
        if geom0.cue_anchor_s is not None and geom0.cue_anchor_s > 0:
            belief = log.observe(
                belief,
                Observation(
                    probe="cue_prior",
                    set_start_s=float(geom0.cue_anchor_s),
                    confidence=0.7,
                    precision=0.50,
                    cost=0.0,
                    detail="timeline cue_anchor",
                ),
            )
            beliefs[slot] = belief
        for _step in range(max_steps_per_span):
            g = session.spans[slot]
            t0, t1 = _listen_window(g.set_start_s, g.set_end_s)
            session.apply(RenderListen(t0, t1))
            out.steps += 1
            for obs in sense_listen(session, g):
                belief = log.observe(belief, obs)
                beliefs[slot] = belief
            decision = ladder.mode(belief)
            if decision == Mode.AUTO_COMMIT:
                session.apply(CommitSpan(slot))
                log.append(slot, "commit", {"mode": mode, "via": "daw_onset"})
                out.committed.append(slot)
                break
            nudge = _propose_nudge(belief, g.set_start_s)
            if nudge is not None and decision in (Mode.REVIEW, Mode.SUGGEST):
                session.apply(NudgeSetStart(slot, nudge))
                out.steps += 1
                log.append(slot, "note", {"nudge_set_start_s": nudge})
                continue
            # No further nudge — Mode B escalates on REVIEW/ESCALATE; Mode A commits.
            if mode == "b" and decision in (Mode.REVIEW, Mode.ESCALATE, Mode.SUGGEST):
                session.apply(EscalateHuman(slot))
                log.append(slot, "escalate", {"reason": decision.value})
                out.escalated.append(slot)
            else:
                session.apply(CommitSpan(slot))
                log.append(slot, "commit", {"mode": mode, "via": decision.value})
                out.committed.append(slot)
            break
        else:
            # max steps
            if mode == "b" and not session.spans[slot].committed:
                session.apply(EscalateHuman(slot))
                log.append(slot, "escalate", {"reason": "max_steps"})
                out.escalated.append(slot)
            elif not session.spans[slot].committed:
                session.apply(CommitSpan(slot))
                log.append(slot, "commit", {"mode": mode, "via": "max_steps"})
                out.committed.append(slot)

    return out
