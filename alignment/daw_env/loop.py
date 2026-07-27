"""resolve_daw — place → listen → sense → iterate (Mode A / B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from alignment.agentic.belief import Observation, SpanBelief
from alignment.agentic.events import EventLog
from alignment.agentic.policy import Ladder, Mode
from alignment.daw_env.actions import (
    CommitSpan,
    EscalateHuman,
    NudgeSetStart,
    RenderListen,
    SoloLayer,
)
from alignment.daw_env.sense import (
    content_target_set_start,
    sense_listen,
)
from alignment.daw_env.session import DawSession


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
    geom_set_start: float, geom_set_end: float, pad_s: float = 8.0
) -> tuple[float, float]:
    return (max(0.0, geom_set_start - pad_s), geom_set_end + pad_s)


# Content probes that land farther than this from current geom are treated as
# wild (wrong diagonal / cold prior) and must not yank the clip. Keep within
# ~2 capped nudge steps so a high-peak HuBERT miss cannot walk 36s off cue.
_CONTENT_NUDGE_MAX_S = 24.0


def _propose_nudge(belief: SpanBelief, geom_start: float) -> float | None:
    """Nudge toward content sensors (fp/HuBERT/lyrics) when they disagree with geom.

    Onset-only / cue fusion must NOT drive nudges — that rubber-stamped 36s
    drifts on regulars when daw_onset peaks elsewhere in the listen window.
    """
    target = content_target_set_start(belief)
    if target is None:
        return None
    delta = float(target) - geom_start
    if abs(delta) < 1.0:
        return None
    if abs(delta) > _CONTENT_NUDGE_MAX_S:
        return None
    # Cap step so one listen cycle does not teleport.
    return float(max(-12.0, min(12.0, delta)))


def _ensure_live_ctx(session: DawSession, *, with_lyrics: bool = False) -> None:
    if session.live_ctx is not None:
        return
    try:
        from alignment.agentic.live_runners import LiveContext
    except Exception:
        return
    spans = [g.to_timeline_span() for g in session.spans.values()]
    # Attach cue for runners that read cue_anchor_s from span dicts.
    by_slot = {g.slot: g for g in session.spans.values()}
    for s in spans:
        g = by_slot.get(str(s["slot_label"]))
        if g is not None and g.cue_anchor_s is not None:
            s["cue_anchor_s"] = g.cue_anchor_s
    # Whisper JOINT lyrics is cold-start expensive; daw_env defaults to
    # fp + HuBERT only. Opt in with with_lyrics=True.
    # Agentic keeps FP placement shadow-only unless opted in — daw_env is
    # the opt-in experiment, so enable placement for instrumental nudges.
    import os

    prev_lyrics = os.environ.get("AGENTIC_LIVE_SKIP_LYRICS")
    prev_fp = os.environ.get("AGENTIC_LIVE_ENABLE_FP_PLACEMENT")
    if not with_lyrics and prev_lyrics is None:
        os.environ["AGENTIC_LIVE_SKIP_LYRICS"] = "1"
    if prev_fp is None:
        os.environ["AGENTIC_LIVE_ENABLE_FP_PLACEMENT"] = "1"
    try:
        session.live_ctx = LiveContext.from_set(session.set_id, spans)
    except Exception:
        session.live_ctx = None
    finally:
        if not with_lyrics and prev_lyrics is None:
            os.environ.pop("AGENTIC_LIVE_SKIP_LYRICS", None)
        elif prev_lyrics is not None:
            os.environ["AGENTIC_LIVE_SKIP_LYRICS"] = prev_lyrics
        if prev_fp is None:
            os.environ.pop("AGENTIC_LIVE_ENABLE_FP_PLACEMENT", None)
        else:
            os.environ["AGENTIC_LIVE_ENABLE_FP_PLACEMENT"] = prev_fp


def resolve_daw(
    session: DawSession,
    *,
    mode: str = "a",
    log: EventLog | None = None,
    ladder: Ladder | None = None,
    max_steps_per_span: int = 3,
    budget_spans: int | None = None,
    use_live_sensors: bool = True,
    with_lyrics: bool = False,
    stems: frozenset[str] | None = None,
) -> DawResolution:
    """Run Ableton ReAct over session spans (longest-first).

    Mode ``a``: commit when ladder says AUTO or after max steps.
    Mode ``b``: escalate_human when ladder says REVIEW / after max steps without AUTO.
    """
    if mode not in ("a", "b"):
        raise ValueError("mode must be 'a' or 'b'")
    log = log or EventLog()
    ladder = ladder or Ladder()
    if use_live_sensors:
        _ensure_live_ctx(session, with_lyrics=with_lyrics)
    beliefs = {
        slot: SpanBelief(slot, g.recording_id, g.claimed_stem)
        for slot, g in session.spans.items()
    }
    order = sorted(session.spans.values(), key=lambda g: -(g.set_end_s - g.set_start_s))
    if stems is not None:
        order = [g for g in order if g.claimed_stem in stems]
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
        # Cue prior only when it differs from current geom — otherwise it
        # rubber-stamps cue-seed and blocks content-driven nudges.
        if geom0.cue_anchor_s is not None and geom0.cue_anchor_s > 0:
            if abs(float(geom0.cue_anchor_s) - geom0.set_start_s) > 0.5:
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
            for obs in sense_listen(session, g, live_ctx=session.live_ctx):
                belief = log.observe(belief, obs)
                beliefs[slot] = belief
            decision = ladder.mode(belief)
            if decision == Mode.AUTO_COMMIT:
                session.apply(CommitSpan(slot))
                log.append(slot, "commit", {"mode": mode, "via": "auto"})
                out.committed.append(slot)
                break
            nudge = _propose_nudge(belief, g.set_start_s)
            if nudge is not None and decision in (
                Mode.REVIEW,
                Mode.SUGGEST,
                Mode.ESCALATE,
            ):
                session.apply(NudgeSetStart(slot, nudge))
                out.steps += 1
                log.append(
                    slot,
                    "note",
                    {
                        "nudge_set_start_s": nudge,
                        "content_target": content_target_set_start(belief),
                    },
                )
                continue
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
            if mode == "b" and not session.spans[slot].committed:
                session.apply(EscalateHuman(slot))
                log.append(slot, "escalate", {"reason": "max_steps"})
                out.escalated.append(slot)
            elif not session.spans[slot].committed:
                session.apply(CommitSpan(slot))
                log.append(slot, "commit", {"mode": mode, "via": "max_steps"})
                out.committed.append(slot)

    return out
