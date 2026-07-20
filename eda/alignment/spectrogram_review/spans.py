"""Load span_table.csv and enrich with success/cause/window fields."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from eda.alignment.spectrogram_review.classify import (
    binding_cause,
    failure_type,
    is_success,
    success_type,
)
from eda.alignment.spectrogram_review.spectrogram import window_bounds

DEFAULT_SPAN_TABLE = (
    Path(__file__).resolve().parent.parent
    / "failure_analysis"
    / "out"
    / "span_table.csv"
)


@dataclass(frozen=True)
class SpanCard:
    set: str
    set_id: str
    slot: str
    recording_id: str
    name: str
    gt_stem: str
    pred_stem: str
    span_class: str
    gt_set_start_s: float
    gt_set_end_s: float
    pred_set_start_s: float
    pred_set_end_s: float
    gt_ref_start_s: float
    gt_ref_end_s: float
    pred_ref_start_s: float
    pred_ref_end_s: float
    ref_offset_err_s: float | None
    tempo_ratio: float
    identity_hit: int
    set_start_err_s: float
    traj_strict: float
    success: bool
    cause: str
    group_key: str  # failure cause or success_type — gallery section
    win_start_s: float
    win_end_s: float
    ref_win_start_s: float
    ref_win_end_s: float
    sec_lost: float

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d["success"] = int(self.success)
        d["outcome"] = "success" if self.success else "failure"
        return d


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    return float(v)


def _f_opt(row: dict[str, str], key: str) -> float | None:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    return float(v)


def load_span_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def enrich_row(
    row: dict[str, str],
    *,
    pad_s: float = 5.0,
    max_window_s: float = 90.0,
    mix_duration_s: float | None = None,
) -> SpanCard | None:
    """Return a SpanCard for a matched GT span, else None."""
    if not row.get("span_class"):
        return None
    gt_start = _f(row, "gt_set_start_s")
    gt_dur = _f(row, "gt_duration_s")
    gt_end = gt_start + gt_dur
    pred_start = _f(row, "pred_set_start_s")
    pred_end = pred_start + gt_dur
    ratio = _f(row, "tempo_ratio", 1.0) or 1.0
    ref_dur = gt_dur * ratio
    gt_ref = _f(row, "gt_ref_start_s")
    pred_ref = _f(row, "pred_ref_start_s")
    gt_ref_end = gt_ref + ref_dur
    pred_ref_end = pred_ref + ref_dur
    audible = _f(row, "gt_audible_s", gt_dur)
    traj = _f(row, "traj_strict")
    win_lo, win_hi = window_bounds(
        gt_start,
        gt_end,
        pred_start,
        pred_end,
        pad_s=pad_s,
        max_window_s=max_window_s,
        mix_duration_s=mix_duration_s,
    )
    ref_lo, ref_hi = window_bounds(
        gt_ref,
        gt_ref_end,
        pred_ref,
        pred_ref_end,
        pad_s=pad_s,
        max_window_s=max_window_s,
        mix_duration_s=None,
    )
    identity = row.get("identity_hit", "1")
    identity_i = 1 if identity in ("", None) else int(float(identity))
    ok = is_success(row)
    cause = binding_cause(row)
    group = success_type(row) if ok else failure_type(row)
    return SpanCard(
        set=row.get("set") or "",
        set_id=row.get("set_id") or "",
        slot=str(row.get("slot") or ""),
        recording_id=row.get("recording_id") or "",
        name=row.get("name") or "",
        gt_stem=row.get("gt_stem") or "regular",
        pred_stem=row.get("pred_stem") or "regular",
        span_class=row.get("span_class") or "",
        gt_set_start_s=gt_start,
        gt_set_end_s=gt_end,
        pred_set_start_s=pred_start,
        pred_set_end_s=pred_end,
        gt_ref_start_s=gt_ref,
        gt_ref_end_s=gt_ref_end,
        pred_ref_start_s=pred_ref,
        pred_ref_end_s=pred_ref_end,
        ref_offset_err_s=_f_opt(row, "ref_offset_err_s"),
        tempo_ratio=ratio,
        identity_hit=identity_i,
        set_start_err_s=_f(row, "set_start_err_s"),
        traj_strict=traj,
        success=ok,
        cause=cause,
        group_key=group,
        win_start_s=win_lo,
        win_end_s=win_hi,
        ref_win_start_s=max(0.0, ref_lo),
        ref_win_end_s=max(ref_lo + 1.0, ref_hi),
        sec_lost=audible * (1.0 - traj),
    )


def iter_cards(
    rows: Iterable[dict[str, str]],
    *,
    set_id: str | None = None,
    outcome: str = "all",
    cause: str | None = None,
    stem: str | None = None,
    pad_s: float = 5.0,
    max_window_s: float = 90.0,
) -> Iterator[SpanCard]:
    for row in rows:
        if set_id and row.get("set_id") != set_id:
            continue
        card = enrich_row(row, pad_s=pad_s, max_window_s=max_window_s)
        if card is None:
            continue
        if outcome == "success" and not card.success:
            continue
        if outcome == "failure" and card.success:
            continue
        if cause and card.cause != cause:
            continue
        if stem and card.gt_stem != stem:
            continue
        yield card


def sort_cards(
    cards: Iterable[SpanCard], *, worst_first: bool = True
) -> list[SpanCard]:
    return sorted(cards, key=lambda c: c.sec_lost, reverse=worst_first)


def select_cards(
    cards: list[SpanCard],
    *,
    limit: int,
    outcome: str,
    worst_first: bool = True,
) -> list[SpanCard]:
    """Apply limit, diversifying across failure causes / success types."""
    ordered = sort_cards(cards, worst_first=worst_first)
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    if outcome != "all":
        return _diverse_take(ordered, limit)
    fails = [c for c in ordered if not c.success]
    oks = sorted(
        (c for c in cards if c.success),
        key=lambda c: (c.traj_strict, -c.sec_lost),
        reverse=True,
    )
    n_fail = min(len(fails), (limit + 1) // 2)
    n_ok = min(len(oks), limit - n_fail)
    if n_fail + n_ok < limit:
        n_fail = min(len(fails), limit - n_ok)
    picked = _diverse_take(fails, n_fail) + _diverse_take(oks, n_ok)
    return sort_cards(picked, worst_first=worst_first)


def _diverse_take(cards: list[SpanCard], n: int) -> list[SpanCard]:
    """Round-robin across group_key so one cause doesn't dominate the sample."""
    if n <= 0 or not cards:
        return []
    by_g: dict[str, list[SpanCard]] = defaultdict(list)
    for c in cards:
        by_g[c.group_key].append(c)
    keys = sorted(
        by_g.keys(), key=lambda k: sum(x.sec_lost for x in by_g[k]), reverse=True
    )
    out: list[SpanCard] = []
    idx = {k: 0 for k in keys}
    while len(out) < n:
        progressed = False
        for k in keys:
            i = idx[k]
            if i < len(by_g[k]):
                out.append(by_g[k][i])
                idx[k] = i + 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out
