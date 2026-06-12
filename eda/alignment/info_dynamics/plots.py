"""Fig-8 analogue: information signals over time with GT boundaries overlaid.

Emits **dependency-free SVG** rather than using matplotlib: the venv's
Python 3.14 build has a broken ``pyexpat`` (system libexpat symbol mismatch)
that makes ``import matplotlib`` fail. SVG needs no third-party imports, renders
in VSCode / any browser, and keeps the study fully reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import StudyData
from .evaluate import PeakConfig, pick_peaks_seconds, restrict_to_labeled
from .signals import SignalSet, smooth

_W = 1500
_PANEL_H = 150
_PAD_L = 70
_PAD_R = 20
_PAD_T = 26
_GAP = 14


def _x_of(t_min: np.ndarray, t0: float, t1: float) -> np.ndarray:
    span = max(t1 - t0, 1e-6)
    return _PAD_L + (t_min - t0) / span * (_W - _PAD_L - _PAD_R)


def _segments(xs: np.ndarray, ys: np.ndarray) -> list[str]:
    """Polyline point-strings, broken at NaNs."""
    out: list[str] = []
    cur: list[str] = []
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            cur.append(f"{x:.1f},{y:.1f}")
        elif cur:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def save_signal_overlay(
    path: Path,
    sigset: SignalSet,
    data: StudyData,
    *,
    cfg: PeakConfig,
    show_peaks: bool = True,
) -> None:
    names = sigset.names()
    t_min = data.bar_start_s / 60.0
    t0, t1 = float(t_min.min()), float(t_min.max())
    gt_x = _x_of(data.gt_boundary_s / 60.0, t0, t1)
    x = _x_of(t_min, t0, t1)

    total_h = _PAD_T + len(names) * (_PANEL_H + _GAP) + 30
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{total_h}" '
        f'font-family="sans-serif" font-size="11">',
        f'<rect width="{_W}" height="{total_h}" fill="white"/>',
        f'<text x="{_PAD_L}" y="16" font-size="13" font-weight="bold">'
        f'{sigset.model} — info signals vs GT boundaries (red) · peaks (green ▾) · {data.set_id}</text>',
    ]

    for pi, name in enumerate(names):
        top = _PAD_T + pi * (_PANEL_H + _GAP)
        bot = top + _PANEL_H
        sig = smooth(sigset.get(name), window=cfg.smooth_window)
        finite = sig[np.isfinite(sig)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        rng = max(hi - lo, 1e-9)
        ys = bot - (sig - lo) / rng * _PANEL_H

        # panel frame + label
        svg.append(f'<rect x="{_PAD_L}" y="{top}" width="{_W-_PAD_L-_PAD_R}" '
                   f'height="{_PANEL_H}" fill="#fafafa" stroke="#ddd"/>')
        svg.append(f'<text x="6" y="{top+_PANEL_H/2:.0f}" fill="#333">{name}</text>')
        # GT boundary lines
        for gx in gt_x:
            svg.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{bot}" '
                       f'stroke="#dc2626" stroke-opacity="0.16"/>')
        # signal polylines
        for seg in _segments(x, ys):
            svg.append(f'<polyline points="{seg}" fill="none" stroke="#2563eb" stroke-width="0.8"/>')
        # detected peaks
        if show_peaks:
            _, ptimes = pick_peaks_seconds(sigset.get(name), data.bar_start_s, cfg)
            ptimes = restrict_to_labeled(ptimes, data)
            for px in _x_of(ptimes / 60.0, t0, t1):
                svg.append(f'<path d="M{px:.1f},{top+3} l-4,-6 l8,0 z" fill="#16a34a"/>')

    # x-axis ticks (minutes)
    ybase = _PAD_T + len(names) * (_PANEL_H + _GAP)
    for tick in range(0, int(t1) + 1, 5):
        tx = float(_x_of(np.array([tick]), t0, t1)[0])
        svg.append(f'<line x1="{tx:.1f}" y1="{ybase}" x2="{tx:.1f}" y2="{ybase+5}" stroke="#888"/>')
        svg.append(f'<text x="{tx:.1f}" y="{ybase+18}" text-anchor="middle" fill="#555">{tick}m</text>')

    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg))
