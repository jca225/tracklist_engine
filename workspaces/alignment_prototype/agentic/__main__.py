"""Replay-mode validation: drive the agentic loop over a real predicted
timeline and score each autonomy rung against GT.

Replay runners re-emit the observations the real chain produced (via the
per-span ``start_source`` provenance + cue anchors); no audio compute. This
validates the LADDER — is the auto-commit rung ≥90% clean, is the escalate
queue honest — not the probes themselves. Known bias: losing probes'
proposals aren't serialized, so cross-probe agreement is underestimated and
spans skew toward review (the safe direction).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.agentic \
        --set-id 2nvzlh2k --gt labeling/fixtures/bb11_ground_truth.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.agentic.actions import REGISTRY
from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief
from workspaces.alignment_prototype.agentic.events import EventLog
from workspaces.alignment_prototype.agentic.loop import Resolution, SpanCtx, resolve
from workspaces.alignment_prototype.agentic.policy import Ladder

OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def _abstain(probe: str) -> Observation:
    return Observation(
        probe=probe,
        set_start_s=None,
        confidence=0.0,
        precision=REGISTRY[probe].precision,
        detail="replay: not recorded / did not fire",
    )


def _replay_runner(probe: str):
    """Emit the recorded observation when this probe produced the span's final
    placement (start_source provenance); otherwise abstain."""

    def _run(data: dict) -> Observation:
        spec = REGISTRY[probe]
        if probe == "cue_prior":
            cue = data.get("cue_anchor_s")
            slot = str(data.get("slot_label") or "")
            if cue is None or (cue == 0.0 and "w" in slot):
                return _abstain(probe)  # scraped cue is 0.0 for every w-row
            return Observation(
                probe=probe,
                set_start_s=float(cue),
                confidence=0.6,
                precision=spec.precision,
                cost=spec.cost,
            )
        if data.get("start_source") == probe:
            conf = float(data.get("confidence") or 0.8)
            return Observation(
                probe=probe,
                set_start_s=float(data["set_start_s"]),
                ref_start_s=data.get("ref_start_s"),
                confidence=max(0.3, min(conf, 1.0)),
                precision=spec.precision,
                cost=spec.cost,
            )
        return _abstain(probe)

    return _run


def _score(beliefs: dict[str, SpanBelief], gt_by_tid: dict[str, list[dict]]) -> dict:
    errs: list[float] = []
    unmatched = 0
    for b in beliefs.values():
        top = b.best()
        rows = gt_by_tid.get(b.recording_id)
        if top is None or not rows:
            unmatched += 1
            continue
        errs.append(min(abs(top.set_start_s - r["set_start_s"]) for r in rows))
    n = len(errs)
    return {
        "n": n,
        "unmatched": unmatched,
        "clean_15s_pct": round(100 * sum(e <= 15 for e in errs) / n) if n else None,
        "clean_5s_pct": round(100 * sum(e <= 5 for e in errs) / n) if n else None,
        "median_s": round(sorted(errs)[n // 2], 1) if n else None,
    }


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
            slot_label=str(s["slot_label"]),
            recording_id=str(s["recording_id"]),
            claimed_stem=s.get("claimed_stem") or "regular",
            data=s,
        )
        for s in timeline["spans"]
    ]
    # replay has no mert values for overridden spans — mert runner abstains there
    runners = {
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

    log_path = OUT_DIR / "agentic" / f"{args.set_id}_events.jsonl"
    if log_path.exists():
        log_path.unlink()  # replay is regenerable; the log of a LIVE run never is
    res: Resolution = resolve(spans, runners, EventLog(log_path), ladder=Ladder())
    print(res.summary())

    # id-normalized GT index (same convention as score_timeline_vs_gt)
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{args.set_id}.json"
    id_map = json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    gt_by_tid: dict[str, list[dict]] = {}
    for r in yaml.safe_load(args.gt.read_text())["tracks"]:
        tid = str(r.get("track_id") or "")
        if tid and str(r.get("slot_label")) != "mix":
            gt_by_tid.setdefault(id_map.get(tid, tid), []).append(r)

    print(f"\nrung-precision vs GT ({args.gt.name}):")
    for name, group in (
        ("auto_commit", res.committed),
        ("review", res.review),
        ("suggest", res.suggested),
        ("escalate", res.escalated),
    ):
        s = _score(group, gt_by_tid)
        print(
            f"  {name:12} n={len(group):3}  scored={s['n']:3}  "
            f"clean<=15s={s['clean_15s_pct']}%  <=5s={s['clean_5s_pct']}%  median={s['median_s']}s"
        )
    print(f"\nevent log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
