"""Mel spectrogram → square RGB canvas + OD-style ACTUAL/PRED boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 22_050
N_FFT = 2048
HOP = 512
N_MELS = 128
DEFAULT_SIZE = 512  # square OD canvas (like DETR training crops)

# Object-detection class colors (cue-detr: GT white / pred magenta)
GT_COLOR = (80, 220, 120)  # green — ACTUAL (ground truth)
PRED_COLOR = (255, 60, 220)  # magenta — PRED
SUCCESS_EDGE = (46, 204, 113)
FAIL_EDGE = (231, 76, 60)
LABEL_FG = (255, 255, 255)


@dataclass(frozen=True)
class TimeBox:
    """Time interval in absolute mix seconds → OD bbox on the square canvas."""

    start_s: float
    end_s: float
    class_name: str  # e.g. "ACTUAL", "PRED"
    color: tuple[int, int, int]
    band: str = "full"  # "full" | "top" | "bottom" — avoid total overlap


def mel_rgb(
    y: np.ndarray,
    *,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP,
    n_mels: int = N_MELS,
) -> Image.Image:
    """Return native-resolution RGB mel spectrogram for mono audio ``y``."""
    import librosa
    from matplotlib import cm

    if y.size == 0:
        return Image.new("RGB", (8, n_mels), color=(0, 0, 0))

    M = librosa.feature.melspectrogram(
        y=y.astype(np.float32),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    M_db = librosa.power_to_db(M, ref=np.max)
    arr = M_db[::-1]  # low freq at bottom
    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_clim(float(arr.min()), float(arr.max()))
    rgba = sm.to_rgba(arr, bytes=True)
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    return Image.fromarray(rgb, mode="RGB")


def to_square(image: Image.Image, size: int = DEFAULT_SIZE) -> Image.Image:
    """Stretch spectrogram to a square OD canvas (time → x, mel → y)."""
    return image.resize((size, size), Image.Resampling.BILINEAR)


def _time_to_x(t_s: float, win_start_s: float, win_end_s: float, width: int) -> int:
    dur = max(1e-6, win_end_s - win_start_s)
    frac = (t_s - win_start_s) / dur
    x = int(round(frac * (width - 1)))
    return max(0, min(width - 1, x))


def _band_y(band: str, height: int) -> tuple[int, int]:
    """Vertical band so ACTUAL vs PRED boxes stay readable when they overlap in time."""
    if band == "top":
        return 4, int(height * 0.48)
    if band == "bottom":
        return int(height * 0.52), height - 4
    return 4, height - 4


def _blend_rect(
    base: Image.Image,
    xy: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    alpha: float = 0.28,
) -> None:
    """Semi-transparent fill (OD highlight) composited onto ``base`` in-place."""
    x0, y0, x1, y1 = xy
    if x1 <= x0 or y1 <= y0:
        return
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle(
        (x0, y0, x1, y1),
        fill=(color[0], color[1], color[2], int(255 * alpha)),
    )
    composed = Image.alpha_composite(base.convert("RGBA"), overlay)
    base.paste(composed.convert("RGB"))


def _label_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None,
) -> None:
    """Filled class-name badge at the top-left of a detection box."""
    pad_x, pad_y = 5, 3
    if font is not None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = 7 * len(text), 11
    x1 = x + tw + 2 * pad_x
    y1 = y + th + 2 * pad_y
    draw.rectangle((x, y, x1, y1), fill=color)
    draw.text((x + pad_x, y + pad_y), text, fill=LABEL_FG, font=font)


def draw_od_boxes(
    image: Image.Image,
    boxes: Sequence[TimeBox],
    *,
    win_start_s: float,
    win_end_s: float,
    edge_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Draw object-detection boxes + class chips on a square spectrogram."""
    out = image.copy()
    W, H = out.size
    try:
        # default bitmap font is tiny; try a larger TrueType if present
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16
                )
            except OSError:
                font = ImageFont.load_default()
    except OSError:
        font = None

    if edge_color is not None:
        draw = ImageDraw.Draw(out)
        draw.rectangle((0, 0, W - 1, H - 1), outline=edge_color, width=4)

    for box in boxes:
        x0 = _time_to_x(box.start_s, win_start_s, win_end_s, W)
        x1 = _time_to_x(box.end_s, win_start_s, win_end_s, W)
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 - x0 < 3:
            x1 = min(W - 1, x0 + 3)
        y0, y1 = _band_y(box.band, H)
        _blend_rect(out, (x0, y0, x1, y1), box.color, alpha=0.30)
        draw = ImageDraw.Draw(out)
        draw.rectangle((x0, y0, x1, y1), outline=box.color, width=3)
        # keep chips short — times live in the HTML card copy
        _label_chip(draw, x0, max(0, y0 - 2), box.class_name, box.color, font)

    return out


def window_bounds(
    gt_start: float,
    gt_end: float,
    pred_start: float,
    pred_end: float,
    *,
    pad_s: float = 5.0,
    max_window_s: float = 90.0,
    mix_duration_s: float | None = None,
) -> tuple[float, float]:
    """Union of GT/pred extents + pad, clamped to max_window and mix length."""
    lo = min(gt_start, pred_start) - pad_s
    hi = max(gt_end, pred_end) + pad_s
    if hi - lo > max_window_s:
        mid = 0.5 * (gt_start + pred_start)
        lo = mid - 0.5 * max_window_s
        hi = lo + max_window_s
    if mix_duration_s is not None:
        lo = max(0.0, lo)
        hi = min(mix_duration_s, hi)
        if hi <= lo:
            hi = min(mix_duration_s, lo + 1.0)
    else:
        lo = max(0.0, lo)
    return float(lo), float(hi)
