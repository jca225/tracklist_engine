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


# timeline provenance vocabulary -> registry action names
_SOURCE_ALIAS = {"mert_decode": "mert"}


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
        src_name = _SOURCE_ALIAS.get(probe, probe)
        proposals = data.get("probe_proposals") or {}
        if probe in proposals:  # rich artifact: every probe's raw proposal
            won = data.get("start_source") == src_name
            conf = float(data.get("confidence") or 0.8) if won else 0.7
            return Observation(
                probe=probe,
                set_start_s=float(proposals[probe]),
                ref_start_s=data.get("ref_start_s") if won else None,
                confidence=max(0.3, min(conf, 1.0)),
                precision=spec.precision,
                cost=spec.cost,
            )
        if data.get("start_source") == src_name:  # legacy artifact: winner only
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
    p.add_argument("--sweep", action="store_true", help="grid-sweep ladder thresholds")
    p.add_argument(
        "--live",
        action="store_true",
        help="call the real DSP/ML probes on demand (live_runners) instead of "
        "re-emitting the serialized chain observations (replay). fp/lyrics/hubert "
        "recompute from the pulled aligning audio; cue/mert read the anchored "
        "priors on the span. Writes a distinct <set>_events_live.jsonl log.",
    )
    args = p.parse_args(argv)

    tl_path = args.timeline or OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(tl_path.read_text())

    # Stem-axis routing MUST come from GT, not the timeline: the timeline inherits
    # `claimed_stem` from the DB, whose materialize dropped row-text (Instrumental)/
    # (Acappella) markers and stored 'regular' (fixed in code, not re-materialized
    # on pi). Measured drift: timeline marks ~2 instrumental spans/set vs ~25 in GT,
    # and undercounts acappella too — so instrumental/acappella spans get mis-routed
    # to the wrong stem probes (vocals miss lyrics/HuBERT) and the stem axis is never
    # measured fairly. Slice it from GT by slot_label (the unique per-span key).
    import re as _re

    def _norm_slot(x) -> str:
        # GT uses zero-padded '001'/'001w1'; the timeline uses '1'/'1w1'. Strip
        # leading zeros from the numeric prefix so the two key the same.
        m = _re.match(r"^0*(\d+)(.*)$", str(x))
        return (m.group(1) + m.group(2)) if m else str(x)

    gt_rows = yaml.safe_load(args.gt.read_text())["tracks"]
    gt_stem_by_slot = {
        _norm_slot(r.get("slot_label")): (r.get("claimed_stem") or "regular")
        for r in gt_rows
        if r.get("slot_label") and str(r.get("slot_label")) != "mix"
    }
    corrected = 0

    def _stem_for(s: dict) -> str:
        tl_stem = s.get("claimed_stem") or "regular"
        gt_stem = gt_stem_by_slot.get(_norm_slot(s["slot_label"]))
        return gt_stem if gt_stem else tl_stem

    spans = []
    for s in timeline["spans"]:
        stem = _stem_for(s)
        if stem != (s.get("claimed_stem") or "regular"):
            corrected += 1
        spans.append(
            SpanCtx(
                slot_label=str(s["slot_label"]),
                recording_id=str(s["recording_id"]),
                claimed_stem=stem,
                data={**s, "claimed_stem": stem},  # runners read data['claimed_stem']
            )
        )
    if corrected:
        print(f"[stem-route] corrected {corrected} spans' claimed_stem from GT")
    if args.live:
        from workspaces.alignment_prototype.agentic.live_runners import (
            LiveContext,
            build_live_runners,
        )

        ctx = LiveContext.from_set(args.set_id, timeline["spans"])
        if ctx is None:
            sys.exit(f"live: could not build LiveContext for {args.set_id} (no MERT)")
        runners = build_live_runners(ctx)
        print(
            f"[live] runners loaded: {sorted(runners)} (channels: {sorted(ctx.loaded)})"
        )
        for name in ("cue_prior", "mert_decode", "fp", "lyrics", "stem_hubert"):
            if name not in runners:
                print(f"[live] {name}: SKIPPED (channel not loaded — abstains)")
    else:
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
    # surprise is live even in replay: its novelty curve comes from the cached
    # mix-MERT artifact (order-free, no GT or chain output involved)
    from workspaces.alignment_prototype.agentic.novelty import surprise_runner_for_set

    surprise = surprise_runner_for_set(args.set_id)
    if surprise is not None:
        runners["surprise"] = surprise
    else:
        print(f"[skip] surprise: no {args.set_id}_mix_mert.npz artifact")

    # a live run's log is not regenerable — keep it off the replay log path
    log_name = (
        f"{args.set_id}_events_live.jsonl"
        if args.live
        else f"{args.set_id}_events.jsonl"
    )
    log_path = OUT_DIR / "agentic" / log_name
    if log_path.exists():
        # each run is a full recompute; start its own log fresh. The distinct
        # *_live.jsonl name keeps a live run from ever clobbering the replay log.
        log_path.unlink()
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
    if args.sweep:
        all_beliefs = {**res.committed, **res.review, **res.suggested, **res.escalated}
        print("\nladder sweep (auto rung: coverage / clean<=15s / median):")
        for auto in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
            grp = {k: b for k, b in all_beliefs.items() if b.quality() >= auto}
            s = _score(grp, gt_by_tid)
            cov = round(100 * len(grp) / max(1, len(all_beliefs)))
            print(
                f"  auto>={auto:.2f}  cov={cov:3}%  n={s['n']:3}  "
                f"clean15={s['clean_15s_pct']}%  clean5={s['clean_5s_pct']}%  med={s['median_s']}s"
            )

    print(f"\nevent log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
