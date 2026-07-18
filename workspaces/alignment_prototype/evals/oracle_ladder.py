"""Acappella oracle→e2e gap decomposition ("oracle ladder").

Attributes the acappella trajectory gap between end-to-end and oracle-placement
to {routing, identity, placement} by re-decoding the SAME acappella population
under progressively more oracle inputs, decoder held fixed at looptrace, scored
against a fixed GT-acappella-row denominator. Design + decision rule:
docs/superpowers/specs/2026-07-18-acappella-oracle-ladder-design.md.
"""

from __future__ import annotations

from typing import Callable

from workspaces.alignment_prototype.score_timeline_vs_gt import norm_slot

RUNGS: tuple[str, ...] = ("R0", "R1", "R2", "R3")


def _by_slot(spans: list[dict]) -> dict[str, dict]:
    return {norm_slot(str(s["slot_label"])): s for s in spans}


def build_rung_timeline(
    rung: str, r0_spans: list[dict], gt_acap_rows: list[dict]
) -> list[dict]:
    """Acappella input-span list for *rung*, oracle-substituted, slot-matched.

    R0: matched r0 spans unchanged. R1: + GT stem (acappella routing).
    R2: + GT recording_id (predicted placement retained; unmatched rows omitted).
    R3: ALL gt rows synthesized with GT recording + GT placement (full oracle).
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r} (expected one of {RUNGS})")
    r0_by_slot = _by_slot(r0_spans)
    out: list[dict] = []
    for g in gt_acap_rows:
        slot = norm_slot(str(g["slot_label"]))
        p = r0_by_slot.get(slot)
        if rung == "R3":
            out.append(
                {
                    "slot_label": str(g["slot_label"]),
                    "recording_id": str(g["track_id"]),
                    "claimed_stem": "acappella",
                    "set_start_s": float(g["set_start_s"]),
                    "set_end_s": float(g["set_end_s"]),
                    "ref_start_s": 0.0,
                    "ref_end_s": 0.0,
                }
            )
            continue
        if p is None:
            continue  # R0/R1/R2: no predicted placement -> row scores 0 downstream
        s = dict(p)
        if rung in ("R1", "R2"):
            s["claimed_stem"] = "acappella"
        if rung == "R2":
            s["recording_id"] = str(g["track_id"])
        s.setdefault("ref_start_s", 0.0)
        out.append(s)
    return out


def summarize_acappella(
    decoded_spans: list[dict],
    gt_acap_rows: list[dict],
    score_fn: Callable[[dict, dict], tuple[float, float]],
) -> dict:
    """Mean strict/fiber over the FIXED GT-acappella-row denominator.

    A GT row with no matching decoded span (by slot) contributes 0 to both
    sums — never dropped (that is the inflation trap of predicted-centric
    scoring). ``score_fn(span, gt_row) -> (strict, fiber)`` is injected so this
    is testable without audio.
    """
    dec_by_slot = _by_slot(decoded_spans)
    n = len(gt_acap_rows)
    n_scored = 0
    strict_sum = 0.0
    fiber_sum = 0.0
    for g in gt_acap_rows:
        slot = norm_slot(str(g["slot_label"]))
        span = dec_by_slot.get(slot)
        if span is None:
            continue  # missing decode -> contributes 0 to both sums
        st, fb = score_fn(span, g)
        strict_sum += float(st)
        fiber_sum += float(fb)
        n_scored += 1
    return {
        "n": n,
        "n_scored": n_scored,
        "strict": strict_sum / n if n else 0.0,
        "fiber": fiber_sum / n if n else 0.0,
    }
