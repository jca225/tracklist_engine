"""Mix-side landmark matching → placement scores and ``set_fingerprint_hits`` rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .landmark_fp import FHOP, LandmarkFingerprint, SR, fp_offset

HIT_MIN_VOTES = 25
HIT_MIN_SHARPNESS = 1.2


@dataclass(frozen=True)
class MixFpHit:
    mix_start_s: float
    mix_end_s: float
    recording_id: str
    stem: str
    score: float
    votes: int
    sharpness: float


def score_mix_window(
    mix_y: np.ndarray,
    *,
    ref_fp: LandmarkFingerprint,
    ref_y: np.ndarray | None = None,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> tuple[int, float, float]:
    """Return (votes, sharpness, stretch) for one mix excerpt vs one ref."""
    _off, votes, _st, sharp = fp_offset(
        mix_y,
        ref_y,
        ref_fp=ref_fp,
        stretches=stretches,
    )
    return votes, sharp, _st


def scan_band(
    mix_y: np.ndarray,
    *,
    ref_fp: LandmarkFingerprint,
    ref_y: np.ndarray | None,
    lo_s: float,
    hi_s: float,
    win_s: float,
    step_s: float,
    recording_id: str,
    stem: str,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> tuple[MixFpHit, ...]:
    """Slide a fixed window; emit hits where landmark evidence is peaked."""
    dur = len(mix_y) / SR
    lo_s = max(0.0, lo_s)
    hi_s = min(dur, hi_s)
    if hi_s - lo_s < win_s * 0.5:
        return ()

    scores: list[tuple[float, int, float, float]] = []
    t = lo_s
    while t + win_s <= hi_s + 1e-6:
        i0 = int(t * SR)
        i1 = int(min((t + win_s) * SR, len(mix_y)))
        chunk = mix_y[i0:i1]
        if len(chunk) < SR // 2:
            break
        votes, sharp, _st = score_mix_window(
            chunk, ref_fp=ref_fp, ref_y=ref_y, stretches=stretches
        )
        scores.append((t, votes, sharp, win_s))
        t += step_s

    if not scores:
        return ()

    vote_arr = np.array([s[1] for s in scores], dtype=np.float64)
    sharp_arr = np.array([s[2] for s in scores], dtype=np.float64)
    # z-score sharpness relative to the band (peak/second selector from fine_placement_plan)
    mu, sig = sharp_arr.mean(), sharp_arr.std() + 1e-9
    z = (sharp_arr - mu) / sig

    hits: list[MixFpHit] = []
    for (start, votes, sharp, w), zz in zip(scores, z):
        if votes < HIT_MIN_VOTES or sharp < HIT_MIN_SHARPNESS:
            continue
        if zz < 1.0:
            continue
        hits.append(
            MixFpHit(
                mix_start_s=start,
                mix_end_s=start + w,
                recording_id=recording_id,
                stem=stem,
                score=float(zz),
                votes=int(votes),
                sharpness=float(sharp),
            )
        )
    return tuple(hits)


def placement_curve(
    mix_y: np.ndarray,
    *,
    ref_fp: LandmarkFingerprint,
    ref_y: np.ndarray | None,
    measure_mid_s: np.ndarray,
    coarse_start_s: float,
    band_s: float,
    win_s: float = 12.0,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> np.ndarray:
    """Per-measure placement emission scores aligned to ``measure_mid_s``.

    Returns (T,) float64; invalid starts masked to -1e18 (sequence_decode convention).
    """
    from .sequence_decode import NEG

    t = measure_mid_s
    lo = max(0.0, coarse_start_s - band_s)
    hi = min(len(mix_y) / SR, coarse_start_s + band_s + win_s)
    step = max(0.5, float(np.median(np.diff(t))) if len(t) > 1 else 2.0)

    grid_t: list[float] = []
    grid_v: list[float] = []
    cur = lo
    while cur + win_s <= hi + 1e-6:
        i0 = int(cur * SR)
        i1 = int(min((cur + win_s) * SR, len(mix_y)))
        chunk = mix_y[i0:i1]
        if len(chunk) >= SR // 2:
            votes, sharp, _ = score_mix_window(
                chunk, ref_fp=ref_fp, ref_y=ref_y, stretches=stretches
            )
            grid_t.append(cur)
            grid_v.append(float(votes) * float(sharp))
        cur += step

    curve = np.full(len(t), NEG, dtype=np.float64)
    if not grid_t:
        return curve

    gt = np.asarray(grid_t)
    gv = np.asarray(grid_v)
    for i, mid in enumerate(t):
        if mid < lo or mid > hi:
            continue
        j = int(np.argmin(np.abs(gt - mid)))
        curve[i] = gv[j]
    return curve


def load_mix_mono(path: Path, *, sr: int = SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def span_from_offset_votes(
    mix_hashes: dict,
    ref_fp: LandmarkFingerprint,
    *,
    gap_s: float = 6.0,
    tol: int = 1,
) -> tuple[float, float, int, float] | None:
    """(set_start_s, set_end_s, votes, offset_s) from the fingerprint's own
    vote-extent — the placement primitive behind the 2026-06-28 reframe.

    The landmark vote bins are off = ref_frame - mix_frame; the dominant bin is
    the alignment diagonal d. The mix-times voting for d are exactly where the
    ref plays in the mix, so the densest contiguous cluster of them (gap-split at
    ``gap_s``) is the played span [set_start, set_end] — directly, with no
    ref_start, cue, or GT. This is why the ~30s set_start "wall" was illusory:
    the fingerprint localizes the diagonal to ~0.2s and its vote-extent gives
    set_start to ~5.7s median (BB12 regular). Outliers (repeat / weak-fp /
    heavy-crossfade) want a boundary-snap (D2) + fiber handling on top.

    Pass ``mix_hashes`` = landmark_fp.hashes(*constellation(mix)) computed ONCE
    per set and reused across refs.
    """
    votes, pairs = _vote_pairs(mix_hashes, ref_fp)
    if not votes:
        return None
    d = max(votes.items(), key=lambda kv: kv[1])[0]  # dominant diagonal
    return _cluster_at(pairs, d, tol=tol, gap_s=gap_s)


def offset_candidates(
    mix_hashes: dict,
    ref_fp: LandmarkFingerprint,
    *,
    topk: int = 6,
    gap_s: float = 6.0,
    tol: int = 1,
) -> list[tuple[float, float, int, float]]:
    """Top-K alignment-diagonal candidates, each (set_start_s, set_end_s, votes,
    offset_s) from that offset's densest contiguous vote-cluster, ordered by
    total votes.

    Feeds the monotonic placement decode: pass all spans' candidates (in
    tracklist order) to sequence_decode.monotonic_decode so a high-vote but
    out-of-order candidate (a wrong-diagonal / repeat-instance pick) is rejected
    for a lower-vote IN-ORDER one. Measured BB12 regular vs argmax-only:
    outliers>15s 11->9, mean 44->28s, <15s 70->76% — the proper outlier fix the
    post-hoc tricks (cluster-strength, isotonic, boundary-snap) could not do.
    """
    votes, pairs = _vote_pairs(mix_hashes, ref_fp)
    if not votes:
        return []
    cands: list[int] = []
    for off, _ in sorted(votes.items(), key=lambda kv: -kv[1]):
        if all(abs(off - c) > tol for c in cands):
            cands.append(off)
        if len(cands) >= topk:
            break
    out = []
    for c in cands:
        r = _cluster_at(pairs, c, tol=tol, gap_s=gap_s)
        if r:
            out.append(r)
    return out


def _vote_pairs(mix_hashes: dict, ref_fp: LandmarkFingerprint):
    """(offset->count, [(offset, mix_frame)]) for matching landmark hash keys.
    off = ref_frame - mix_frame; the dominant offset is the alignment diagonal."""
    votes: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for key, mts in mix_hashes.items():
        rts = ref_fp.hashes.get(key)
        if not rts:
            continue
        for mt in mts:
            for rt in rts:
                off = rt - mt
                votes[off] = votes.get(off, 0) + 1
                pairs.append((off, mt))
    return votes, pairs


def _cluster_at(
    pairs: list[tuple[int, int]], off: int, *, tol: int, gap_s: float
) -> tuple[float, float, int, float] | None:
    """Densest contiguous mix-time cluster of votes for diagonal ``off`` ->
    (set_start_s, set_end_s, votes_in_cluster, offset_s)."""
    mts = sorted(mt for o, mt in pairs if abs(o - off) <= tol)
    if not mts:
        return None
    ts = np.array(mts, dtype=np.float64) * FHOP / SR
    cluster = max(np.split(ts, np.where(np.diff(ts) > gap_s)[0] + 1), key=len)
    return (
        float(cluster[0]),
        float(cluster[-1]),
        int(len(cluster)),
        float(off * FHOP / SR),
    )


def decode_placements(
    mix_hashes: dict,
    ref_fps: list,
    *,
    mix_dur_s: float,
    dt: float = 2.0,
    topk: int = 6,
    gap_s: float = 6.0,
    tol: int = 1,
    min_step: int = 0,
) -> list[tuple[float, float] | None]:
    """Set-level fingerprint placement: per-span top-K diagonal candidates ->
    monotonic decode over tracklist order. ``ref_fps`` are LandmarkFingerprints
    in tracklist (slot) order; returns [(set_start_s, set_end_s) | None] aligned
    to that order (None where a ref produced no candidates).

    The decode enforces non-decreasing set_start (min_step=0 admits the
    near-simultaneous starts of mashup layers), so a high-vote but out-of-order
    candidate (wrong-diagonal / repeat instance) is rejected for the best in-order
    one — the outlier fix validated on BB12 regular (mean 44->28s, outliers 11->9
    vs argmax-only). Spans whose true diagonal is absent from top-K (genuine
    weak-fp / heavy-crosstalk) remain errors -> per-stem + fibers.
    """
    from .sequence_decode import NEG, monotonic_decode

    cand_lists = [
        offset_candidates(mix_hashes, fp, topk=topk, gap_s=gap_s, tol=tol)
        for fp in ref_fps
    ]
    out: list[tuple[float, float] | None] = [None] * len(ref_fps)
    keep = [i for i, c in enumerate(cand_lists) if c]
    if not keep:
        return out
    T = int(mix_dur_s / dt) + 1
    curves = np.full((len(keep), T), NEG, dtype=np.float64)
    for r, i in enumerate(keep):
        cands = cand_lists[i]
        mx = max(c[2] for c in cands) or 1.0
        for ss, _se, votes, _off in cands:
            b = min(T - 1, int(ss / dt))
            curves[r, b] = max(curves[r, b], votes / mx)
    starts = monotonic_decode(curves, min_step=min_step)
    for r, i in enumerate(keep):
        ss_pred = float(starts[r]) * dt
        best = min(cand_lists[i], key=lambda c: abs(c[0] - ss_pred))
        out[i] = (best[0], best[1])
    return out
