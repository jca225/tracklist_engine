"""LLM-as-policy (design tier 2): Opus adjudicates the ambiguous residual.

The heuristic loop auto-commits the easy majority; the spans it leaves in
review/suggest/escalate are exactly the value-of-information residual the
design doc reserves for an LLM policy. In replay mode all probe outputs are
already recorded, so the LLM's action space collapses to ADJUDICATION: given
each span's observations (with calibrated precisions) plus context the
heuristic can't exploit — tracklist ordering and the auto-committed
neighbors' placements — pick a placement or escalate.

Caller: the authenticated Claude Code CLI in headless mode (`claude -p
--model opus`), so no API key handling; swap in the anthropic SDK when a key
lands in .env.

Usage:
    venvs/audio/bin/python -m alignment.agentic.llm_policy \
        --set-id 2nvzlh2k --gt labeling/fixtures/bb11_ground_truth.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.agentic.__main__ import _replay_runner
from alignment.agentic.actions import REGISTRY
from alignment.agentic.belief import SpanBelief
from alignment.agentic.events import EventLog
from alignment.agentic.loop import SpanCtx, resolve
from alignment.agentic.policy import Ladder

OUT_DIR = Path(__file__).resolve().parents[1] / "out"
CHUNK = 25
MODEL = "opus"  # claude CLI alias for claude-opus-4-8

_PROMPT_HEADER = """You are the placement adjudicator for a DJ-mix alignment system.

A DJ set (duration {mix_dur:.0f}s) plays tracks in tracklist order. Each SPAN is one
track's appearance. Its hidden true placement (set_start_s, seconds into the mix)
was measured by several noisy probes. Each probe observation below carries its
measured precision (P(correct | probe fired)); probes may disagree or abstain.

You also get ANCHORS: spans already auto-committed at high confidence. Slot labels
encode tracklist order (e.g. 7 < 7w1 < 8 — 'w' rows are layers over the numbered
host row), and placements are usually monotonic in slot order, so anchors bracket
plausible windows for the spans between them.

For EACH span below, decide:
- "set_start_s": your best placement estimate (a number, seconds)
- "action": "commit" if the evidence supports a placement, "escalate" if evidence
  is contradictory/absent and a human should label it
- "confidence": 0.0-1.0
- "reason": one short clause

Rules: prefer high-precision probes; use anchors + slot ordering to break ties or
reject out-of-window outliers; when probes agree within ~8s, average them; do not
invent placements with no supporting evidence or anchor logic — escalate instead.

Output PURE JSON only — an array of objects, one per span, same order as given,
each: {{"slot_label": ..., "set_start_s": ..., "action": ..., "confidence": ..., "reason": ...}}

ANCHORS (slot_label @ set_start_s):
{anchors}

SPANS TO ADJUDICATE:
{spans}
"""


def _slot_key(slot: str) -> tuple[int, int]:
    m = re.match(r"^(\d+)(?:w(\d+))?$", str(slot))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (10**9, 0)


def _ask_opus(prompt: str) -> list[dict]:
    proc = subprocess.run(
        ["claude", "-p", "--model", MODEL],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=600,
    )
    raw = proc.stdout.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start = raw.find("[")
    if start < 0:
        raise ValueError(f"no JSON array in response: {raw[:200]}")
    return json.loads(raw[start : raw.rfind("]") + 1])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--timeline", type=Path, default=None)
    args = p.parse_args(argv)

    tl_path = args.timeline or OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(tl_path.read_text())
    spans = [
        SpanCtx(
            str(s["slot_label"]),
            str(s["recording_id"]),
            s.get("claimed_stem") or "regular",
            s,
        )
        for s in timeline["spans"]
    ]
    runners = {n: _replay_runner(n) for n in REGISTRY}
    res = resolve(spans, runners, EventLog(), ladder=Ladder())

    residual: dict[str, SpanBelief] = {**res.review, **res.suggested, **res.escalated}
    print(f"heuristic: {res.summary()}")
    print(f"LLM adjudicates {len(residual)} residual spans")

    anchors = sorted(
        ((k, b.best().set_start_s) for k, b in res.committed.items() if b.best()),
        key=lambda kv: _slot_key(kv[0]),
    )
    anchor_txt = "\n".join(f"  {k} @ {v:.1f}" for k, v in anchors)
    mix_dur = max(s.data["set_end_s"] for s in spans)

    span_ctx = {s.slot_label: s for s in spans}
    ordered = sorted(residual, key=_slot_key)
    decisions: dict[str, dict] = {}
    for i in range(0, len(ordered), CHUNK):
        chunk = ordered[i : i + CHUNK]
        lines = []
        for k in chunk:
            b, ctx = residual[k], span_ctx[k]
            obs = "; ".join(
                f"{o.probe}→{'abstain' if o.abstained else f'{o.set_start_s:.1f}s'}"
                f" (prec {o.precision:.2f})"
                for o in b.observations
            )
            lines.append(
                f"  slot {k} [{ctx.claimed_stem}, dur {ctx.data['set_end_s'] - ctx.data['set_start_s']:.0f}s]: {obs}"
            )
        prompt = _PROMPT_HEADER.format(
            mix_dur=mix_dur, anchors=anchor_txt, spans="\n".join(lines)
        )
        for d in _ask_opus(prompt):
            decisions[str(d.get("slot_label"))] = d
        print(f"  chunk {i // CHUNK + 1}: {len(decisions)}/{len(ordered)} adjudicated")

    out_path = OUT_DIR / "agentic" / f"{args.set_id}_llm_adjudication.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decisions, indent=1))

    # ---- score: heuristic residual vs LLM residual, vs GT --------------------
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{args.set_id}.json"
    id_map = json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    gt_by_tid: dict[str, list[dict]] = {}
    for r in yaml.safe_load(args.gt.read_text())["tracks"]:
        tid = str(r.get("track_id") or "")
        if tid and str(r.get("slot_label")) != "mix":
            gt_by_tid.setdefault(id_map.get(tid, tid), []).append(r)

    def err(rec_id: str, start: float | None) -> float | None:
        rows = gt_by_tid.get(rec_id)
        if start is None or not rows:
            return None
        return min(abs(start - r["set_start_s"]) for r in rows)

    heur_e, llm_e, llm_esc = [], [], 0
    for k in ordered:
        b = residual[k]
        top = b.best()
        e1 = err(b.recording_id, top.set_start_s if top else None)
        d = decisions.get(k) or {}
        if d.get("action") == "escalate":
            llm_esc += 1
            e2 = None
        else:
            e2 = err(b.recording_id, d.get("set_start_s"))
        if e1 is not None:
            heur_e.append(e1)
        if e2 is not None:
            llm_e.append(e2)

    def stats(es: list[float]) -> str:
        if not es:
            return "n=0"
        es = sorted(es)
        return (
            f"n={len(es)} median={es[len(es) // 2]:.1f}s "
            f"<=15s {100 * sum(e <= 15 for e in es) // len(es)}% "
            f"<=5s {100 * sum(e <= 5 for e in es) // len(es)}%"
        )

    print(f"\nresidual placement vs GT ({len(ordered)} spans):")
    print(f"  heuristic best-cluster : {stats(heur_e)}")
    print(f"  opus adjudication      : {stats(llm_e)}  (+{llm_esc} escalated to human)")
    print(f"decisions: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
