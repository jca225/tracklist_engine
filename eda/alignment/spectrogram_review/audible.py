"""Audible extents for review Truth boxes.

Alignment is about what you can hear in the mix. Silent stem padding and
fader lead-in are not alignment targets — do not draw Truth over them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from eda.alignment.spectrogram_review.ableton_label import _gt_rows, resolve_ableton
from eda.alignment.spectrogram_review.spans import SpanCard


def _gt_match(
    card: SpanCard,
) -> dict[str, Any] | None:
    info = resolve_ableton(
        card.set_id,
        recording_id=card.recording_id,
        gt_set_start_s=card.gt_set_start_s,
        fallback_slot=card.slot,
        fallback_name=card.name,
        gt_stem=card.gt_stem,
        pred_set_start_s=card.pred_set_start_s,
        pred_set_end_s=card.pred_set_end_s,
        pred_stem=card.pred_stem,
    )
    slot = str(info.get("ableton_slot") or "")
    start = float(info.get("mix_start_s") or card.gt_set_start_s)
    for r in _gt_rows(card.set_id):
        if (
            str(r.get("slot_label") or "") == slot
            and abs(float(r["set_start_s"]) - start) < 0.05
        ):
            return r
    # fallback: closest same stem by set_start
    cand = [
        r
        for r in _gt_rows(card.set_id)
        if (r.get("claimed_stem") or "regular").lower()
        == (card.gt_stem or "regular").lower()
    ] or list(_gt_rows(card.set_id))
    if not cand:
        return None
    return min(cand, key=lambda r: abs(float(r["set_start_s"]) - card.gt_set_start_s))


def mix_truth_span(card: SpanCard) -> tuple[float, float]:
    """Mix-timeline Truth: audible_start/end when present, else clip set span."""
    g = _gt_match(card)
    if g is None:
        return card.gt_set_start_s, card.gt_set_end_s
    s0, s1 = float(g["set_start_s"]), float(g["set_end_s"])
    a0 = g.get("audible_start_s")
    a1 = g.get("audible_end_s")
    frac = g.get("audible_frac")
    lo = float(a0) if isinstance(a0, (int, float)) else s0
    hi = float(a1) if isinstance(a1, (int, float)) else s1
    # Only audible_end + frac: treat the missing fraction as silent lead-in.
    if a0 is None and isinstance(a1, (int, float)) and isinstance(frac, (int, float)):
        dur = max(1e-3, s1 - s0)
        lo = float(a1) - float(frac) * dur
        lo = max(s0, min(lo, hi))
    return lo, max(lo + 0.05, hi)


def _mix_to_ref(g: dict[str, Any], mix_t: float) -> float:
    """Map a mix time into ref time via ref_segments, else linear clip map."""
    segs = g.get("ref_segments") or []
    tempo = float(g.get("tempo_ratio") or 1.0) or 1.0
    for seg in segs:
        ms = float(seg["mix_start_s"])
        rs = float(seg["ref_start_s"])
        re = float(seg["ref_end_s"])
        # segment duration in mix ≈ (re-rs)/tempo for warped clips; prefer end hints
        mix_len = (re - rs) / tempo if tempo else (re - rs)
        if mix_t + 1e-6 < ms:
            continue
        if mix_t <= ms + mix_len + 1e-3:
            return rs + (mix_t - ms) * tempo
    s0 = float(g["set_start_s"])
    rs0 = float(g.get("ref_start_s") or 0.0)
    return rs0 + (mix_t - s0) * tempo


def stem_energy_onset_s(
    path: Path,
    *,
    thresh_rms: float = 0.01,
    hop_s: float = 0.25,
    max_scan_s: float = 90.0,
) -> float | None:
    """First time the stem exceeds ``thresh_rms``; None if always quiet/missing."""
    if path is None or not path.is_file():
        return None
    # Decode a mono preview once — clips are short windows, stems may be longer.
    try:
        raw = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-t",
                f"{max_scan_s:.3f}",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "8000",
                "-",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    x = np.frombuffer(raw, dtype=np.float32)
    if x.size == 0:
        return None
    hop = max(1, int(hop_s * 8000))
    for i in range(0, len(x) - hop, hop):
        chunk = x[i : i + hop]
        if float(np.sqrt(np.mean(chunk * chunk))) >= thresh_rms:
            return i / 8000.0
    return None


def ref_truth_span(
    card: SpanCard,
    *,
    source_path: Path | None = None,
) -> tuple[float, float]:
    """Source-timeline Truth: audible mix region mapped to ref, silence trimmed."""
    g = _gt_match(card)
    mix_lo, mix_hi = mix_truth_span(card)
    if g is not None:
        r0 = _mix_to_ref(g, mix_lo)
        r1 = _mix_to_ref(g, mix_hi)
    else:
        # linear fallback from card fields
        ratio = card.tempo_ratio or 1.0
        span = max(1e-3, card.gt_set_end_s - card.gt_set_start_s)
        r0 = card.gt_ref_start_s + (mix_lo - card.gt_set_start_s) / span * (
            card.gt_ref_end_s - card.gt_ref_start_s
        )
        r1 = card.gt_ref_start_s + (mix_hi - card.gt_set_start_s) / span * (
            card.gt_ref_end_s - card.gt_ref_start_s
        )
    if r1 < r0:
        r0, r1 = r1, r0
    onset = stem_energy_onset_s(source_path) if source_path is not None else None
    if onset is not None:
        r0 = max(r0, onset)
        if r1 <= r0:
            r1 = r0 + 0.05
    return float(r0), float(r1)


def ref_view_window(
    card: SpanCard,
    truth: tuple[float, float],
    *,
    pad_s: float = 5.0,
    max_window_s: float = 90.0,
) -> tuple[float, float]:
    """Source spectrogram window around audible Truth + pred — skip silent heads."""
    from eda.alignment.spectrogram_review.spectrogram import window_bounds

    lo, hi = window_bounds(
        truth[0],
        truth[1],
        card.pred_ref_start_s,
        card.pred_ref_end_s,
        pad_s=pad_s,
        max_window_s=max_window_s,
        mix_duration_s=None,
    )
    # window_bounds recentering can pull lo back into stem silence — don't.
    lo = max(lo, max(0.0, truth[0] - pad_s))
    if hi <= lo:
        hi = lo + min(max_window_s, max(1.0, truth[1] - truth[0] + 2 * pad_s))
    return float(lo), float(hi)
