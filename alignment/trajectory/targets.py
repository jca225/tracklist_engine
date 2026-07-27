"""GT span -> per-frame trajectory supervision.

The eval (`path_decode.trajectory_acc`) samples ref(mix_t) once per second
with a 2 s tolerance, interpolating GT via `_gt_pieces`/`_ref_at`. Training
must optimize the same quantity, so targets are rasterized with those exact
functions — one label per feature bin:

- a ref position (the second of the reference the mix frame is playing), or
- NULL where the hand-labeled fader (`gain_curve`) has the clip inaudible —
  the clip "plays" in Ableton but the mix does not contain it, so the model
  must learn to predict silence there, or
- ignore (outside the ref, or an `unalignable` row — a positive abstain
  label whose placement loss is masked but which stays a training example).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alignment.path_decode import _gt_pieces, _ref_at, _span_class

MIN_AUDIBLE_GAIN = 0.1

KIND_POSITION = 0
KIND_NULL = 1
KIND_IGNORE = -1


@dataclass(frozen=True)
class TrajectoryTarget:
    """Per-frame supervision for one GT span (all arrays length Tm)."""

    times_s: np.ndarray  # absolute mix time at bin centers
    ref_pos_s: np.ndarray  # GT ref position (valid only where kind==POSITION)
    gain: np.ndarray  # linear fader gain, 1.0 = unity (empty curve)
    kind: np.ndarray  # KIND_POSITION | KIND_NULL | KIND_IGNORE per frame
    abstain: bool  # unalignable row
    span_class: str  # linear | multiseg | loop | oddratio


def gain_at(gain_curve, times_s: np.ndarray) -> np.ndarray:
    """Piecewise-linear fader gain at each time; empty curve = unity."""
    pts = [
        (float(t), float(g))
        for t, g in (gain_curve or ())
        if isinstance(t, (int, float)) and isinstance(g, (int, float))
    ]
    if not pts:
        return np.ones_like(times_s)
    pts.sort()
    xs = np.array([p[0] for p in pts])
    gs = np.array([p[1] for p in pts])
    return np.interp(times_s, xs, gs)


def raster_targets(
    row: dict,
    times_s: np.ndarray,
    ref_dur_s: float | None,
    min_gain: float = MIN_AUDIBLE_GAIN,
) -> TrajectoryTarget:
    """Rasterize one GT yaml row onto the caller's bin-center time grid.

    `times_s` comes from the feature grid (span bins snapped to the global
    bin raster), NOT from set_start_s directly — so labels and features
    can never drift by a partial bin.
    """
    pieces = _gt_pieces(row)
    ref_pos = np.array([_ref_at(pieces, float(t)) for t in times_s])
    gain = gain_at(row.get("gain_curve"), times_s)
    abstain = bool(row.get("unalignable", False))

    kind = np.full(times_s.shape, KIND_POSITION, dtype=np.int64)
    kind[gain < min_gain] = KIND_NULL
    in_ref = ref_pos >= 0.0
    if ref_dur_s is not None:
        in_ref &= ref_pos < ref_dur_s
    kind[(kind == KIND_POSITION) & ~in_ref] = KIND_IGNORE
    if abstain:
        kind[:] = KIND_IGNORE

    return TrajectoryTarget(
        times_s=times_s,
        ref_pos_s=ref_pos,
        gain=gain,
        kind=kind,
        abstain=abstain,
        span_class=_span_class(row),
    )


def frames_to_segments(
    ref_bin: np.ndarray,
    null_mask: np.ndarray,
    bin_s: float,
    jump_bins: int = 2,
) -> list[tuple[float, float, float]]:
    """Collapse a per-frame decode into `[(mix_start_s, ref_start_s, ref_end_s)]`.

    mix_start is RELATIVE to the span start (the `decode_path` convention that
    `trajectory_acc` expects); ref times are absolute. Frames whose ref-offset
    (ref_bin - frame index) drifts by more than `jump_bins` start a new
    segment; NULL frames terminate the current one.
    """
    segs: list[tuple[float, float, float]] = []
    start = None  # (frame0, refbin0)
    prev_off = None
    prev_ref = None
    for t in range(len(ref_bin) + 1):
        ended = t == len(ref_bin) or bool(null_mask[t])
        off = None if ended else int(ref_bin[t]) - t
        if start is not None and (ended or abs(off - prev_off) > jump_bins):
            t0, r0 = start
            segs.append((t0 * bin_s, r0 * bin_s, (prev_ref + 1) * bin_s))
            start = None
        if not ended and start is None:
            start = (t, int(ref_bin[t]))
        if not ended:
            prev_off = off
            prev_ref = int(ref_bin[t])
    return segs
