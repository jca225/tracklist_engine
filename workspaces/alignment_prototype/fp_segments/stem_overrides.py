"""Evaluation-only GT stem routing onto timeline spans.

BB11 GT slot labels share the tracklist namespace, so zero-strip matching works.
BB12 GT labels are Ableton clip numbers and collide with unrelated timeline
slots when zero-stripped (e.g. Ableton ``007`` vs tracklist ``7``). The durable
bridge is ``GT.track_id == timeline.recording_id``, with nearest ``set_start_s``
when a recording appears more than once.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml


def norm_slot(value: object) -> str:
    match = re.match(r"^0*(\d+)(.*)$", str(value))
    return match.group(1) + match.group(2) if match else str(value)


def timeline_stem_overrides(
    spans: Sequence[Mapping[str, object]],
    gt_yaml: Path,
) -> dict[str, str]:
    """Map timeline ``slot_label`` (normalized) → GT ``claimed_stem``.

    Only recording-id matches are emitted. Ableton-only GT rows with no
    timeline recording are omitted (true inventory gaps).
    """
    rows = yaml.safe_load(Path(gt_yaml).read_text())["tracks"]
    by_rid: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for span in spans:
        rid = span.get("recording_id")
        if rid is None:
            continue
        by_rid[str(rid)].append(span)

    # (norm_slot -> (stem, time_err)) keep the closer GT match on conflicts.
    chosen: dict[str, tuple[str, float]] = {}
    for row in rows:
        slot = row.get("slot_label")
        if not slot or str(slot) == "mix":
            continue
        tid = row.get("track_id")
        if not tid:
            continue
        stem = str(row.get("claimed_stem") or "regular")
        candidates = by_rid.get(str(tid), [])
        if not candidates:
            continue
        gt_ss = row.get("set_start_s")
        if gt_ss is not None and len(candidates) > 1:
            best = min(
                candidates,
                key=lambda span: abs(
                    float(span.get("set_start_s") or 0.0) - float(gt_ss)
                ),
            )
            err = abs(float(best.get("set_start_s") or 0.0) - float(gt_ss))
        else:
            best = candidates[0]
            err = (
                abs(float(best.get("set_start_s") or 0.0) - float(gt_ss))
                if gt_ss is not None
                else 0.0
            )
        key = norm_slot(best["slot_label"])
        prev = chosen.get(key)
        if prev is None or err < prev[1]:
            chosen[key] = (stem, err)
    return {slot: stem for slot, (stem, _err) in chosen.items()}
