"""In-memory Ableton-semantic session + action history sidecar."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from workspaces.alignment_prototype.daw_env.actions import (
    CommitSpan,
    DawAction,
    EscalateHuman,
    NudgeRefStart,
    NudgeSetStart,
    PlaceSpan,
    RenderListen,
    SoloLayer,
)

# Output stamp — never write * align.als (seeder provenance rule).
DAW_REACT_SUFFIX = " DAW REACT.als"


@dataclass(frozen=True)
class SpanGeom:
    slot: str
    recording_id: str
    claimed_stem: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    name: str = ""
    cue_anchor_s: float | None = None
    committed: bool = False
    escalated: bool = False

    @property
    def duration_s(self) -> float:
        return max(0.0, self.set_end_s - self.set_start_s)

    def to_timeline_span(self) -> dict[str, Any]:
        return {
            "slot_label": self.slot,
            "recording_id": self.recording_id,
            "claimed_stem": self.claimed_stem,
            "name": self.name,
            "set_start_s": self.set_start_s,
            "set_end_s": self.set_end_s,
            "ref_start_s": self.ref_start_s,
            "start_source": "daw_react",
        }


@dataclass
class DawSession:
    """Working session: span geometry is SoT; ALS path optional sync target."""

    set_id: str
    spans: dict[str, SpanGeom]
    solo_bus: str = "mix"
    als_path: Path | None = None
    out_dir: Path | None = None
    set_dir: Path | None = None
    live_ctx: Any | None = None  # optional agentic LiveContext for fp/HuBERT/lyrics
    history: list[dict[str, Any]] = field(default_factory=list)
    last_listen_wav: Path | None = None
    last_listen_window: tuple[float, float] | None = None

    @classmethod
    def from_timeline(
        cls,
        set_id: str,
        timeline: dict[str, Any],
        *,
        out_dir: Path | None = None,
        als_path: Path | None = None,
        set_dir: Path | None = None,
    ) -> DawSession:
        spans: dict[str, SpanGeom] = {}
        for s in timeline.get("spans") or []:
            slot = str(s.get("slot_label") or s.get("slot") or "")
            if not slot:
                continue
            start = float(s["set_start_s"])
            end = float(
                s.get("set_end_s") or (start + float(s.get("duration_s") or 8.0))
            )
            cue = s.get("cue_anchor_s")
            spans[slot] = SpanGeom(
                slot=slot,
                recording_id=str(s.get("recording_id") or ""),
                claimed_stem=str(s.get("claimed_stem") or "regular"),
                set_start_s=start,
                set_end_s=end,
                ref_start_s=float(s.get("ref_start_s") or 0.0),
                name=str(s.get("name") or ""),
                cue_anchor_s=float(cue) if cue not in (None, "") else None,
            )
        return cls(
            set_id=set_id,
            spans=spans,
            out_dir=out_dir,
            als_path=als_path,
            set_dir=set_dir,
        )

    def apply(self, action: DawAction) -> None:
        """Apply one DAW action; append to history."""
        kind = type(action).__name__
        payload: dict[str, Any] = {"kind": kind}
        if isinstance(action, PlaceSpan):
            g = self.spans[action.slot]
            dur = (
                g.duration_s
                if g.duration_s > 0
                else (action.set_end_s - action.set_start_s)
            )
            end = (
                action.set_end_s
                if action.set_end_s > action.set_start_s
                else action.set_start_s + dur
            )
            self.spans[action.slot] = replace(
                g,
                set_start_s=action.set_start_s,
                set_end_s=end,
                ref_start_s=action.ref_start_s,
            )
            payload.update(
                {
                    "slot": action.slot,
                    "set_start_s": action.set_start_s,
                    "set_end_s": end,
                    "ref_start_s": action.ref_start_s,
                }
            )
            self._sync_als_clip(action.slot)
        elif isinstance(action, NudgeSetStart):
            g = self.spans[action.slot]
            ns = g.set_start_s + action.delta_s
            ne = g.set_end_s + action.delta_s
            self.spans[action.slot] = replace(g, set_start_s=ns, set_end_s=ne)
            payload.update(
                {"slot": action.slot, "delta_s": action.delta_s, "set_start_s": ns}
            )
            self._sync_als_clip(action.slot)
        elif isinstance(action, NudgeRefStart):
            g = self.spans[action.slot]
            self.spans[action.slot] = replace(
                g, ref_start_s=g.ref_start_s + action.delta_s
            )
            payload.update(
                {
                    "slot": action.slot,
                    "delta_s": action.delta_s,
                    "ref_start_s": self.spans[action.slot].ref_start_s,
                }
            )
            self._sync_als_clip(action.slot)
        elif isinstance(action, SoloLayer):
            self.solo_bus = action.bus
            payload["bus"] = action.bus
        elif isinstance(action, RenderListen):
            from workspaces.alignment_prototype.daw_env.render import render_listen

            wav = render_listen(self, action.t0_s, action.t1_s, set_dir=self.set_dir)
            self.last_listen_wav = wav
            self.last_listen_window = (action.t0_s, action.t1_s)
            payload.update({"t0_s": action.t0_s, "t1_s": action.t1_s, "wav": str(wav)})
        elif isinstance(action, CommitSpan):
            g = self.spans[action.slot]
            self.spans[action.slot] = replace(g, committed=True, escalated=False)
            payload["slot"] = action.slot
        elif isinstance(action, EscalateHuman):
            g = self.spans[action.slot]
            self.spans[action.slot] = replace(g, escalated=True, committed=False)
            payload["slot"] = action.slot
            self._write_escalate_locators(action.slot)
        else:
            raise TypeError(f"unknown action {action!r}")
        self.history.append(payload)

    def _sync_als_clip(self, slot: str) -> None:
        if self.als_path is None or not self.als_path.is_file():
            return
        from workspaces.alignment_prototype.daw_env.als_mutate import sync_span_to_als

        sync_span_to_als(self.als_path, self.spans[slot])

    def _write_escalate_locators(self, slot: str) -> None:
        if self.als_path is None or not self.als_path.is_file():
            return
        from workspaces.alignment_prototype.daw_env.als_mutate import (
            write_agent_propose_locator,
        )

        g = self.spans[slot]
        write_agent_propose_locator(self.als_path, g.set_start_s, slot)

    def to_timeline(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "source": "daw_react",
            "spans": [
                g.to_timeline_span()
                for g in sorted(self.spans.values(), key=lambda x: x.set_start_s)
            ],
        }

    def write_history(self, path: Path | None = None) -> Path:
        out = path or (
            (self.out_dir or Path("out/daw_env") / self.set_id) / "action_history.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.history, indent=2, sort_keys=True) + "\n")
        return out

    def daw_react_als_name(self, display: str | None = None) -> str:
        label = display or self.set_id
        return f"{label}{DAW_REACT_SUFFIX}"


def load_timeline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
