"""Driver B — agentic: the POMDP loop refines placement over the classical base.

The agentic loop (`agentic/loop.resolve`) does NOT pick identity — it inherits
recording_id per span from the input timeline and refines set_start via
precision-weighted belief fusion + a permission ladder. So this driver takes the
classical timeline as its base and, per span:

  * committed / review / suggest  -> override set_start with the belief's dominant
    cluster centre (and translate the span + its ref_segments by that delta);
  * escalate (belief quality < 0.10, no probe fired usefully) -> keep the
    classical span verbatim (the locked decision: escalation falls back to
    classical, so the timeline is complete and span-for-span comparable).

The autonomy rung is recorded per span (`driver_mode`, `agentic_quality`) so the
"auto-commit clean%" — the pseudo-GT quality the bootstrap flywheel cares about —
survives into the artifact. Ref content mapping (ref_start / ref_end / segment
ref times) is unchanged: this driver owns PLACEMENT, not the ref decode.
"""

from __future__ import annotations

import json
from pathlib import Path

from workspaces.alignment_prototype.agentic.belief import (
    INDEPENDENCE_GROUP,
    SpanBelief,
)
from workspaces.alignment_prototype.agentic.events import EventLog
from workspaces.alignment_prototype.agentic.loop import Resolution, SpanCtx, resolve
from workspaces.alignment_prototype.agentic.policy import Ladder

from .base import SetContext, finalize, gt_stem_by_slot, load_timeline_dict, norm_slot


def pseudo_acceptance(belief: SpanBelief) -> tuple[bool, str | None]:
    top = belief.best()
    if top is None:
        return False, "g0_mode"
    members = tuple(
        obs
        for obs in belief.observations
        if not obs.abstained and obs.probe in top.probes
    )
    if any("[fiber" in obs.detail for obs in members):
        return False, "g3_fiber"
    groups = {INDEPENDENCE_GROUP.get(obs.probe, obs.probe) for obs in members}
    if len(groups) >= 2 or max((obs.precision for obs in members), default=0.0) >= 0.9:
        return True, None
    return False, "g2_independence"


def refine_agentic_spans(
    timeline: dict,
    spans_ctx: list[SpanCtx],
    runners: dict,
    log: EventLog,
    *,
    ladder: Ladder,
    pseudo_safe: bool = False,
) -> tuple[list[dict], Resolution]:
    res = resolve(spans_ctx, runners, log, ladder=ladder)

    mode_of: dict[str, str] = {}
    belief_of: dict[str, SpanBelief] = {}
    for mode, group in (
        ("auto_commit", res.committed),
        ("review", res.review),
        ("suggest", res.suggested),
        ("escalate", res.escalated),
    ):
        for slot, belief in group.items():
            mode_of[slot] = mode
            belief_of[slot] = belief

    out_spans: list[dict] = []
    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        mode = mode_of.get(slot, "escalate")
        belief = belief_of.get(slot)
        top = belief.best() if belief is not None else None
        if mode == "escalate" or top is None:
            out_spans.append({**span, "driver_mode": "escalate->classical"})
            continue

        rejection = None
        if pseudo_safe and mode == "auto_commit":
            accepted, rejection = pseudo_acceptance(belief)
            if not accepted:
                mode = "review"

        delta = float(top.set_start_s) - float(span["set_start_s"])
        new = {
            **span,
            "set_start_s": round(float(top.set_start_s), 3),
            "set_end_s": round(float(span["set_end_s"]) + delta, 3),
            "driver_mode": mode,
            "agentic_quality": round(belief.quality(), 4),
            "start_source": f"agentic:{'+'.join(sorted(top.probes))}",
        }
        if rejection is not None:
            new["pseudo_gate_rejection"] = rejection
        out_spans.append(new)

    return out_spans, res


class AgenticDriver:
    name = "agentic"

    def __init__(self, base_timeline: Path, *, live: bool = True) -> None:
        self.base = Path(base_timeline)
        self.live = live

    def align_set(self, ctx: SetContext) -> Path:
        from workspaces.alignment_prototype.agentic.novelty import (
            surprise_runner_for_set,
        )

        timeline = load_timeline_dict(self.base)
        gt_stems = gt_stem_by_slot(ctx.gt_yaml)

        # Route each span's probe plan by the GT stem, not the stale timeline
        # value (see base.gt_stem_by_slot). Runners read data['claimed_stem'].
        spans_ctx: list[SpanCtx] = []
        for s in timeline["spans"]:
            stem = gt_stems.get(norm_slot(s["slot_label"])) or (
                s.get("claimed_stem") or "regular"
            )
            spans_ctx.append(
                SpanCtx(
                    slot_label=str(s["slot_label"]),
                    recording_id=str(s["recording_id"]),
                    claimed_stem=stem,
                    data={**s, "claimed_stem": stem},
                )
            )

        runners = self._build_runners(ctx, timeline)
        surprise = surprise_runner_for_set(ctx.set_id)
        if surprise is not None:
            runners["surprise"] = surprise

        log_path = ctx.out_dir / "agentic" / f"{ctx.set_id}_driver_events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        out_spans, res = refine_agentic_spans(
            timeline,
            spans_ctx,
            runners,
            EventLog(log_path),
            ladder=Ladder(),
        )
        print(f"  agentic: {res.summary()}")
        n_fallback = sum(
            span["driver_mode"] == "escalate->classical" for span in out_spans
        )
        n_override = len(out_spans) - n_fallback

        print(
            f"  agentic: {n_override} placements overridden, "
            f"{n_fallback} fell back to classical"
        )
        payload = {
            **timeline,
            "spans": out_spans,
            "driver": f"agentic ({'live' if self.live else 'replay'})",
        }
        return finalize(payload, ctx.timeline_path(self.name))

    def _build_runners(self, ctx: SetContext, timeline: dict) -> dict:
        """Live probes (real fp/lyrics/HuBERT/mert) or replay (re-emit the base
        timeline's serialized observations). Live is the real experiment; replay
        needs no audio and only validates the ladder."""
        if self.live:
            from workspaces.alignment_prototype.agentic.live_runners import (
                LiveContext,
                build_live_runners,
            )

            lctx = LiveContext.from_set(ctx.set_id, timeline["spans"])
            if lctx is None:
                raise RuntimeError(
                    f"agentic live: no LiveContext for {ctx.set_id} (missing mix MERT). "
                    "Pass live=False for the replay ladder-validation path."
                )
            runners = build_live_runners(lctx)
            print(f"  agentic live runners: {sorted(runners)}")
            return runners

        # replay: reuse the exact re-emit logic the CLI validates with
        from workspaces.alignment_prototype.agentic.__main__ import _replay_runner

        return {
            name: _replay_runner(name)
            for name in (
                "cue_prior",
                "mert_decode",
                "fp",
                "lyrics",
                "stem_hubert",
                "chroma_refine",
            )
        }
