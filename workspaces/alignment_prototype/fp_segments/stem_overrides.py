"""Evaluation-only GT stem routing onto timeline spans.

BB11 GT slot labels share the tracklist namespace, so zero-strip matching works.
BB12 GT labels are Ableton clip numbers and collide with unrelated timeline
slots when zero-stripped (e.g. Ableton ``007`` vs tracklist ``7``). The durable
bridge is ``GT.track_id == timeline.recording_id``.

A recording is often played through more than one stem *form* within a single
slot -- e.g. an instrumental bed that rises into the full/regular track, or an
acappella laid over an instrumental. GT encodes this as multiple rows sharing
``(slot_label, track_id)`` with different ``claimed_stem`` over disjoint time
ranges. The routing map therefore keeps **every** form a slot is played in
(keyed by normalized timeline slot -> list of time-ranged forms); a lane admits
a span when any GT form overlapping the span in time belongs to that lane. The
old ``dict[slot -> single stem]`` collapse dropped whichever form was farther
from the predicted span, silently excluding the other form's decode.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


def norm_slot(value: object) -> str:
    match = re.match(r"^0*(\d+)(.*)$", str(value))
    return match.group(1) + match.group(2) if match else str(value)


@dataclass(frozen=True)
class SlotForm:
    """One stem form a slot is played in, with its GT mix-time window."""

    stem: str
    set_start_s: float
    set_end_s: float


def _f(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def timeline_stem_overrides(
    spans: Sequence[Mapping[str, object]],
    gt_yaml: Path,
) -> dict[str, list[SlotForm]]:
    """Map timeline ``slot_label`` (normalized) -> list of GT stem forms.

    Only recording-id matches are emitted. Ableton-only GT rows with no
    timeline recording are omitted (true inventory gaps). Every stem form a
    matched slot is played in is retained -- multi-form slots are not collapsed.
    """
    rows = yaml.safe_load(Path(gt_yaml).read_text())["tracks"]
    by_rid: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for span in spans:
        rid = span.get("recording_id")
        if rid is None:
            continue
        by_rid[str(rid)].append(span)

    forms: dict[str, list[SlotForm]] = defaultdict(list)
    for row in rows:
        slot = row.get("slot_label")
        if not slot or str(slot) == "mix":
            continue
        tid = row.get("track_id")
        if not tid:
            continue
        candidates = by_rid.get(str(tid), [])
        if not candidates:
            continue
        stem = str(row.get("claimed_stem") or "regular")
        ss = _f(row.get("set_start_s"), 0.0)
        se = _f(row.get("set_end_s"), ss)
        form = SlotForm(stem=stem, set_start_s=ss, set_end_s=se)
        # A recording can back more than one timeline slot; attach the form to
        # every matched slot's normalized key (time overlap disambiguates later).
        for span in candidates:
            key = norm_slot(str(span["slot_label"]))
            if form not in forms[key]:
                forms[key].append(form)
    return dict(forms)


def _overlaps(form: SlotForm, span_start: float, span_end: float) -> bool:
    return form.set_start_s <= span_end and form.set_end_s >= span_start


def lane_stem(
    overrides: Mapping[str, list[SlotForm]],
    span: Mapping[str, object],
    claimed_stems: Collection[str],
) -> str | None:
    """The stem to decode this span under in a lane accepting ``claimed_stems``,
    or ``None`` to skip the span in that lane.

    A matched slot admits the span iff some GT form overlapping the span in time
    is in ``claimed_stems``. When the slot has no GT routing (recording absent
    from the GT, or no ``--stem-overrides`` file), fall back to the span's own
    ``claimed_stem`` -- preserving the pre-override default behavior.
    """
    key = norm_slot(str(span["slot_label"]))
    slot_forms = overrides.get(key)
    if not slot_forms:
        stem = str(span.get("claimed_stem") or "regular")
        return stem if stem in claimed_stems else None
    span_start = _f(span.get("set_start_s"), 0.0)
    span_end = _f(span.get("set_end_s"), span_start)
    for form in slot_forms:
        if form.stem in claimed_stems and _overlaps(form, span_start, span_end):
            return form.stem
    return None
