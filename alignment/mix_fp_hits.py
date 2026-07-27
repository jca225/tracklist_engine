"""Mix-side landmark matching → placement scores and ``set_fingerprint_hits`` rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .landmark_fp import FHOP, LandmarkFingerprint, SR, fp_offset

HIT_MIN_VOTES = 25
HIT_MIN_SHARPNESS = 1.2
# Among offset candidates whose votes are ≥ this fraction of the max, prefer
# highest vote-density (votes / cluster duration). Earned on BB11 slot 034
# (decoder_wall bb11_34): vote-argmax was a long false diagonal (+33s) while
# the GT audible-onset cluster was shorter and denser (18/s vs 10/s).
COMPETITIVE_VOTE_FRAC = 0.3


@dataclass(frozen=True)
class MixFpHit:
    mix_start_s: float
    mix_end_s: float
    recording_id: str
    stem: str
    score: float
    votes: int
    sharpness: float


@dataclass(frozen=True)
class FpCandidateEvidence:
    """One ranked FP diagonal with the native evidence needed by an arbiter."""

    rank: int
    set_start_s: float
    set_end_s: float
    offset_s: float
    votes: int
    vote_density: float
    runner_up_ratio: float


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


def candidate_density(cand: tuple[float, float, int, float]) -> float:
    """Votes per second of the vote-extent cluster (set_end - set_start)."""
    ss, se, votes, _off = cand
    return float(votes) / max(float(se - ss), 0.1)


def fp_candidate_evidence(
    candidates: list[tuple[float, float, int, float]],
) -> tuple[FpCandidateEvidence, ...]:
    """Expose ranked diagonal evidence without changing legacy selection.

    ``candidates`` must be in the exact order returned by
    :func:`offset_candidates`. This adapter is intentionally pure so shadow
    candidate materialization and parity tests cannot perturb the decoder.
    """
    out: list[FpCandidateEvidence] = []
    for rank, candidate in enumerate(candidates):
        ss, se, votes, offset = candidate
        others = [
            other[2] for other_rank, other in enumerate(candidates) if other_rank != rank
        ]
        runner_up = max(others) if others else 0
        ratio = float(votes) / float(runner_up) if runner_up else float(votes)
        out.append(
            FpCandidateEvidence(
                rank=rank,
                set_start_s=float(ss),
                set_end_s=float(se),
                offset_s=float(offset),
                votes=int(votes),
                vote_density=candidate_density(candidate),
                runner_up_ratio=ratio,
            )
        )
    return tuple(out)


def pick_dense_competitive(
    cands: list[tuple[float, float, int, float]],
    *,
    frac: float = COMPETITIVE_VOTE_FRAC,
) -> tuple[float, float, int, float] | None:
    """Prefer the densest cluster among candidates with competitive vote mass.

    Vote-argmax alone prefers long false diagonals on multiseg instrumentals
    (bb11_34). Restrict to votes ≥ ``frac`` × max, then maximize density.
    """
    if not cands:
        return None
    mx = max(c[2] for c in cands)
    pool = [c for c in cands if c[2] >= frac * mx]
    return max(pool, key=candidate_density)


def span_from_offset_votes(
    mix_hashes: dict,
    ref_fp: LandmarkFingerprint,
    *,
    gap_s: float = 6.0,
    tol: int = 1,
) -> tuple[float, float, int, float] | None:
    """(set_start_s, set_end_s, votes, offset_s) from the fingerprint's own
    vote-extent — the placement primitive behind the 2026-06-28 reframe.

    The landmark vote bins are off = ref_frame - mix_frame; competitive
    diagonals are clustered on mix-time, then the densest competitive cluster
    wins (see ``pick_dense_competitive``). Pass ``mix_hashes`` =
    landmark_fp.hashes(*constellation(mix)) computed ONCE per set.
    """
    cands = offset_candidates(mix_hashes, ref_fp, topk=6, gap_s=gap_s, tol=tol)
    return pick_dense_competitive(cands)


def offset_candidates(
    mix_hashes: dict,
    ref_fp: LandmarkFingerprint,
    *,
    topk: int = 6,
    gap_s: float = 6.0,
    tol: int = 1,
) -> list[tuple[float, float, int, float]]:
    """Top-K alignment-diagonal candidates, each (set_start_s, set_end_s, votes,
    offset_s) from that offset's densest contiguous vote-cluster.

    Offset bins are still gathered by raw vote count (recall); the returned
    list is ordered densest-competitive first so argmax-style consumers and
    ``decode_placements`` prefer short dense onset clusters over long false
    diagonals (bb11_34).

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
    if not out:
        return []
    # Densest among competitive first; remaining keep relative vote order.
    mx = max(c[2] for c in out)
    competitive = [c for c in out if c[2] >= COMPETITIVE_VOTE_FRAC * mx]
    rest = [c for c in out if c not in competitive]
    competitive.sort(key=candidate_density, reverse=True)
    rest.sort(key=lambda c: c[2], reverse=True)
    return competitive + rest


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
    with_offset: bool = False,
    with_strength: bool = False,
    candidate_evidence_out: list[tuple[FpCandidateEvidence, ...]] | None = None,
) -> list[tuple[float, ...] | None]:
    """Set-level fingerprint placement: per-span top-K diagonal candidates ->
    monotonic decode over tracklist order. ``ref_fps`` are LandmarkFingerprints
    in tracklist (slot) order; returns [(set_start_s, set_end_s) | None] aligned
    to that order (None where a ref produced no candidates). With
    ``with_offset=True`` each entry is (set_start_s, set_end_s, offset_s) instead
    — offset_s = ref_frame-mix_frame in seconds, so ref_start_s = set_start_s +
    offset_s (the part of the song the DJ started on).

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
    if candidate_evidence_out is not None:
        candidate_evidence_out.clear()
        candidate_evidence_out.extend(
            fp_candidate_evidence(candidates) for candidates in cand_lists
        )
    out: list[tuple[float, float] | None] = [None] * len(ref_fps)
    keep = [i for i, c in enumerate(cand_lists) if c]
    if not keep:
        return out
    T = int(mix_dur_s / dt) + 1
    curves = np.full((len(keep), T), NEG, dtype=np.float64)
    for r, i in enumerate(keep):
        cands = cand_lists[i]
        dens = [candidate_density(c) for c in cands]
        mx = max(dens) or 1.0
        for (ss, _se, _votes, _off), d in zip(cands, dens):
            b = min(T - 1, int(ss / dt))
            curves[r, b] = max(curves[r, b], d / mx)
    starts = monotonic_decode(curves, min_step=min_step)
    for r, i in enumerate(keep):
        ss_pred = float(starts[r]) * dt
        best = min(cand_lists[i], key=lambda c: abs(c[0] - ss_pred))
        if with_strength:
            # evidence strength for downstream gates (opinion-audit #1): the
            # chosen diagonal's votes + sharpness vs the strongest OTHER
            # candidate, so a gate can tell an overwhelming diagonal from a
            # coin-flip instead of discarding both.
            others = [c[2] for c in cand_lists[i] if c is not best]
            sharp = float(best[2]) / float(max(others)) if others else float(best[2])
            out[i] = (best[0], best[1], best[3], int(best[2]), sharp)
        elif with_offset:
            out[i] = (best[0], best[1], best[3])
        else:
            out[i] = (best[0], best[1])
    return out
