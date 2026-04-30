"""Viterbi primitives + supporting signals for the SOTA alignment pipeline.

This file holds the pure Viterbi machinery, stem-routing lookups, and the
small indicator maths (MACD, rolling z-score, ATR) that feed the
per-universe emission scorer. `sota.py` is the only caller and composes
these into the end-to-end pipeline.

Nothing in this module writes to the database or knows about a specific
set_id — those concerns live in `sota.py`. The `DB_PATH` module variable
lets `sota.main()` point downstream DB helpers at the right file.

See [docs/SOTA.md](../../docs/SOTA.md) for the pipeline diagram.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------- module-level defaults -------------------------------------------
# `sota.main()` overwrites DB_PATH to the `--db` argument so the few
# connection-opening helpers here (_load_cue_detr_cues etc.) read from the
# right file. No other state leaks out of the module.

DB_PATH: Path = Path("data/db/music_database.db")


# ---------- data model ------------------------------------------------------

@dataclass(frozen=True)
class GtRef:
    """One reference track in the alignment universe.

    Named `GtRef` for historical reasons (the original scorer ran over a
    hand-annotated ground-truth set); the pipeline now loads refs
    dynamically from the tracklist, so `gt_start_s` / `gt_end_s` are
    usually `0.0` and unused outside the eval harness.
    """
    label: str
    track_id: str
    track_audio_id: int
    version_tag: str           # 'full' | 'acappella' | 'instrumental'
    color: str
    cue_s: float
    gt_start_s: float
    gt_end_s: float


# ---------- DB + audio path helpers -----------------------------------------

def _mix_stem_path(conn: sqlite3.Connection, set_audio_id: int, stem_name: str) -> Path | None:
    r = conn.execute(
        "SELECT path FROM set_stems WHERE set_audio_id=? AND stem_name=?",
        (set_audio_id, stem_name),
    ).fetchone()
    return Path(r["path"]) if r else None


def _track_stem_path(conn: sqlite3.Connection, track_audio_id: int, stem_name: str) -> Path | None:
    r = conn.execute(
        "SELECT path FROM track_stems WHERE track_audio_id=? AND stem_name=?",
        (track_audio_id, stem_name),
    ).fetchone()
    return Path(r["path"]) if r else None


def _track_full_path(conn: sqlite3.Connection, track_audio_id: int) -> Path:
    r = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=?", (track_audio_id,),
    ).fetchone()
    return Path(r["path"])


def _stem_routing(tag: str) -> str:
    """Which stem to compare on for a given version_tag."""
    if tag == "instrumental":
        return "instrumental"
    if tag == "acappella":
        return "vocals"
    if tag == "full":
        return "__full__"
    raise ValueError(f"unknown version_tag: {tag}")


def _embed_per_measure(
    audio_path: Path,
    measures: list[tuple[int, float, float, float | None]],
    *,
    duration_s: float | None,
) -> np.ndarray:
    """Cached per-measure MERT embedding for `audio_path`. Thin wrapper over
    `mert_align._cache_measure_embeddings` so callers don't have to know
    about the cache key layout. Writes an `.npz` under `data/cache/mert/`
    on first call and reads from it afterwards."""
    from .mert_align import _cache_measure_embeddings
    r = _cache_measure_embeddings(
        audio_path, measures, offset_s=0.0, duration_s=duration_s,
    )
    if not r.is_ok():
        raise RuntimeError(f"MERT embed failed for {audio_path}: {r.error}")
    return r.value


def _load_cue_detr_cues(
    conn: sqlite3.Connection, track_id: str, track_audio_id: int,
) -> list[float]:
    """Per-song cue-detr cue points (seconds, on the full-song timeline).

    Prefers `canonical_track_cue_points` (keyed by `track_id`, computed
    once on the original/full variant at sensitivity=0.5). Falls back to
    the legacy per-variant `track_analysis.cue_points_json` when
    canonical is missing.
    """
    import json as _json
    r = conn.execute(
        "SELECT cue_points_json FROM canonical_track_cue_points WHERE track_id=?",
        (track_id,),
    ).fetchone()
    if r is None or not r["cue_points_json"]:
        r = conn.execute(
            "SELECT cue_points_json FROM track_analysis WHERE track_audio_id=?",
            (track_audio_id,),
        ).fetchone()
    if r is None or not r["cue_points_json"]:
        return []
    try:
        return sorted(float(x) for x in _json.loads(r["cue_points_json"]))
    except (ValueError, TypeError):
        return []


def _bracket_cue_points(ref_t: float, cue_points_ref: list[float]) -> float | None:
    """Return the ref cue point nearest to `ref_t`. None if no cues."""
    if not cue_points_ref:
        return None
    arr = np.array(cue_points_ref, dtype=np.float64)
    return float(arr[int(np.argmin(np.abs(arr - ref_t)))])


# ---------- small indicator maths -------------------------------------------
# EMA / rolling mean + std and Wilder smoothing are the minimal indicator
# stack needed by the emission scorer. MACD needs EMA; per-ref z-score
# needs SMA + rolling std; ATR uses Wilder smoothing. Everything else
# from the research stack (Bollinger / RSI / ADX / ADXR / DI / MACD
# crossovers / trust gates) was evaluated and dropped — see
# [docs/alignment_archive.md](../../docs/alignment_archive.md).

def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, pandas span convention: alpha=2/(span+1)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average, forward-padded at the boundary."""
    window = max(1, window)
    c = np.cumsum(np.concatenate(([0.0], x.astype(np.float64))))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        seg = x[lo : i + 1]
        out[i] = float(np.std(seg)) if seg.size > 1 else 0.0
    return out


def wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA): alpha=1/period."""
    alpha = 1.0 / period
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd(
    x: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic MACD on a 1D series.
    Returns (macd_line, signal_line, histogram). Units are 'measures' here
    (~2 s @ 120 BPM); MACD(12,26,9) ≈ 24/52/18 second bands."""
    fast_e = ema(x, fast)
    slow_e = ema(x, slow)
    m = fast_e - slow_e
    s = ema(m, signal)
    return m, s, (m - s)


def per_ref_z(x: np.ndarray, window: int = 40) -> np.ndarray:
    """z-score of x against its own rolling mean/std. Kills per-ref
    baseline bias: a ref with a high absolute level can still surprise
    its own history."""
    mu = sma(x, window)
    sd = rolling_std(x, window)
    sd = np.maximum(sd, 1e-6)
    return (x - mu) / sd


def atr(x: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR of a 1D series — Wilder-smoothed absolute first-difference."""
    return wilder_smooth(np.abs(np.diff(x, prepend=x[0])), period)


# ---------- per-ref monotonic ref-position Viterbi -------------------------

def ref_position_viterbi(
    sim: np.ndarray,
    *,
    stay_cost: float = 0.0,
    advance_cost: float = 0.0,
    jump_cost: float = 0.3,
    backward_cost: float = 2.0,
) -> np.ndarray:
    """Monotonic ref-position Viterbi over an (N_ref, N_mix) similarity
    matrix. Returns shape (N_mix,) mapping mix measure → ref measure.

    Subsequence-DTW family:

        states:     0..N_ref-1 (one per ref measure)
        emissions:  1 - sim[ref, mix]   (lower cost = better match)
        transitions:
            delta == 0   (stay)                → stay_cost
            delta == 1   (advance, normal)     → advance_cost
            delta >= 2   (skip, speed up)      → jump_cost * delta
            delta < 0    (backward / loop)     → backward_cost * |delta|

    With near-zero advance/stay and high backward cost, the decoded
    path is monotone-by-construction (with occasional small skips),
    which is exactly what argmax fails to guarantee.
    """
    N_ref, N_mix = sim.shape
    emit = (1.0 - sim).astype(np.float64)
    cost = np.full((N_mix, N_ref), np.inf, dtype=np.float64)
    back = np.full((N_mix, N_ref), -1, dtype=np.int32)

    cost[0] = emit[:, 0]
    RADIUS = 8
    for t in range(1, N_mix):
        prev = cost[t - 1]
        best_prev = np.full(N_ref, -1, dtype=np.int32)
        best_cost = np.full(N_ref, np.inf, dtype=np.float64)
        for r in range(N_ref):
            lo = max(0, r - RADIUS)
            hi = min(N_ref, r + RADIUS + 1)
            candidates = prev[lo:hi].copy()
            for i, rp in enumerate(range(lo, hi)):
                delta = r - rp
                if delta == 0:
                    tc = stay_cost
                elif delta == 1:
                    tc = advance_cost
                elif delta >= 2:
                    tc = jump_cost * delta
                else:
                    tc = backward_cost * (-delta)
                candidates[i] += tc
            j = int(np.argmin(candidates))
            best_prev[r] = lo + j
            best_cost[r] = candidates[j]
        cost[t] = best_cost + emit[:, t]
        back[t] = best_prev

    path = np.empty(N_mix, dtype=np.int32)
    last = int(np.argmin(cost[-1]))
    for t in range(N_mix - 1, -1, -1):
        path[t] = last
        last = int(back[t, last])
    return path


# ---------- per-universe selector Viterbi ----------------------------------

# Emission-score weights (winners picked by cross-sectional z; MACD picks up
# entry/exit events; persistence keeps the score > 0 while the ref is held).
_EMIT_CS_Z_WEIGHT: float = 0.5
_EMIT_MACD_WEIGHT: float = 1.0
_EMIT_PERSIST_WEIGHT: float = 1.5

# Transition / silence costs tuned on BB11. Refs self-loop for free,
# silence is free to stay in, enter/exit are small, cross-ref jumps
# must route through SILENCE.
_SILENCE_EMIT: float = 1.2
_SELF_LOOP_COST: float = 0.0
_SILENCE_STAY_COST: float = 0.0
_ENTER_COST: float = 0.4
_EXIT_COST: float = 0.1
_CROSS_REF_COST: float = 3.0

# Fingerprint-anchor + full-track-exclusion parameters (Phase 5 + 6).
_FP_MIN_SCORE: float = 0.65
_FP_DENSITY_WINDOW_S: float = 10.0
_FP_MIN_DENSITY: int = 2
_FP_ANCHOR_BONUS: float = 1.5
# Full-track exclusion is only safe while the union mask covers a small
# fraction of the mix. On 5-ref BB11 it covers ~15 %; at 119 refs the
# union inflates and starts hard-masking real acap/instr plays, which
# regressed Gnash 0.857 → 0.27. Above this coverage threshold we disable
# full_excl entirely for the set and let the within-universe Viterbi decide.
_FP_FULL_EXCL_COVERAGE_CAP: float = 0.40

# Earliest-run-near-cue cleanup thresholds.
_MERGE_GAP_M: int = 10
_MIN_DURATION_M: int = 5
_CUE_TOLERANCE_S: float = 80.0


def _universe(tag: str) -> str:
    """Mutual-exclusion group for a ref. Acapellas never layer on other
    acapellas; same for instrumentals; full tracks are their own class."""
    return {"acappella": "acapella", "instrumental": "instrumental", "full": "full"}[tag]


def _within_universe_cs_z(
    per_ref: dict[str, np.ndarray], refs_in_u: list[GtRef],
) -> dict[str, np.ndarray]:
    """Cross-sectional z-score computed only over refs in the same
    universe. Using every ref in the set contaminates the score (e.g.
    Bastille's raw MERT sim showing up in the acapella pool)."""
    if len(refs_in_u) == 1:
        # Degenerate one-ref case: fall back to rolling z of the ref
        # against its own history. Goes negative when the ref isn't
        # playing, which is what the Viterbi needs to exit to SILENCE.
        s = per_ref[refs_in_u[0].label]
        return {refs_in_u[0].label: per_ref_z(s, 40)}
    stacked = np.stack([per_ref[r.label] for r in refs_in_u], axis=0)
    mu = stacked.mean(axis=0, keepdims=True)
    sd = np.maximum(stacked.std(axis=0, keepdims=True), 1e-6)
    z = (stacked - mu) / sd
    return {r.label: z[i] for i, r in enumerate(refs_in_u)}


def _emission_score(
    sim: np.ndarray, cs_z: np.ndarray, times: np.ndarray, cue_s: float,
) -> np.ndarray:
    """Per-measure 'is this ref playing' score.

    Composite of:
      1. persistence: `(sim - pre_cue_baseline)` — stays positive for the
         duration of a play, not just at entry.
      2. MACD histogram (ATR-normalised) — sharp entry/exit events.
      3. cross-sectional z within universe — picks the winner.
    """
    _, _, hist = macd(sim)
    a = atr(sim, 14)
    macd_n = np.clip(hist / np.maximum(a, 1e-4), -3.0, 3.0)

    precue = sim[times < cue_s]
    if precue.size >= 3:
        baseline = float(np.mean(precue))
    else:
        baseline = float(np.quantile(sim, 0.1))
    persistence = np.clip((sim - baseline) * 8.0, -2.0, 2.0)

    return (_EMIT_CS_Z_WEIGHT * cs_z
            + _EMIT_MACD_WEIGHT * macd_n
            + _EMIT_PERSIST_WEIGHT * persistence)


def viterbi_universe(
    times: np.ndarray,
    per_ref: dict[str, np.ndarray],
    refs_in_u: list[GtRef],
    *,
    anchors_by_label: dict[str, np.ndarray] | None = None,
    full_exclusion_mask: np.ndarray | None = None,
    mix_bpm: np.ndarray | None = None,                # deprecated, ignored
    ref_bpm_by_label: dict[str, float] | None = None, # deprecated, ignored
) -> np.ndarray:
    """Decode one universe. At each measure, at most one ref is active
    (SILENCE otherwise). Hard left-boundary at each ref's `cue_s`.

    Returns shape (T,) with values in {-1, 0, ..., K-1}, where -1 is
    SILENCE and i ≥ 0 indexes `refs_in_u`.

    `anchors_by_label`: per-ref fingerprint-density mask (Phase 5).
    Reduces emission cost at confirmed measures.
    `full_exclusion_mask`: bool mask on mix measures; in non-full
    universes, forces SILENCE at masked measures (Phase 6). Constructed
    by `sota.py` from the decoded full-universe path in its two-pass
    scheme.

    `mix_bpm` / `ref_bpm_by_label` are deprecated no-ops — the BPM
    penalty experiment was dropped (broke Gnash IoU 0.857 → 0.381 on
    BB11; DJs routinely tempo-shift refs). Kept in the signature so
    callers don't have to update yet.
    """
    del mix_bpm, ref_bpm_by_label    # explicitly unused

    T = len(times)
    K = len(refs_in_u)
    S = K + 1                        # 0..K-1 = refs, K = SILENCE
    silence_idx = K

    cs_z = _within_universe_cs_z(per_ref, refs_in_u)

    # Emission cost matrix (T, S). Cost = -score (min-cost Viterbi).
    emit_cost = np.full((T, S), np.inf, dtype=np.float64)
    emit_cost[:, silence_idx] = -_SILENCE_EMIT
    for i, ref in enumerate(refs_in_u):
        score = _emission_score(per_ref[ref.label], cs_z[ref.label], times, ref.cue_s)
        cost = -score
        mask_active = times >= ref.cue_s
        emit_cost[mask_active, i] = cost[mask_active]

    # Transition cost: leave state a → b. Time-invariant.
    trans = np.full((S, S), _CROSS_REF_COST, dtype=np.float64)
    for i in range(K):
        trans[i, i] = _SELF_LOOP_COST
        trans[i, silence_idx] = _EXIT_COST
        trans[silence_idx, i] = _ENTER_COST
    trans[silence_idx, silence_idx] = _SILENCE_STAY_COST

    # Phase 5 — within-universe fingerprint anchors.
    if anchors_by_label is not None:
        for i, ref in enumerate(refs_in_u):
            mask = anchors_by_label.get(ref.label)
            if mask is None or not mask.any():
                continue
            emit_cost[mask, i] -= _FP_ANCHOR_BONUS

    # Phase 6 — cross-universe full-track exclusion. Only applied to
    # non-full universes (full is the SOURCE of the exclusion, not a
    # subject). When active, forces SILENCE at masked measures.
    if full_exclusion_mask is not None:
        is_full_universe = any(r.version_tag == "full" for r in refs_in_u)
        if not is_full_universe:
            emit_cost[full_exclusion_mask, :K] = 1e6

    # Viterbi forward pass.
    cost = np.full((T, S), np.inf, dtype=np.float64)
    back = np.full((T, S), -1, dtype=np.int32)
    cost[0, silence_idx] = emit_cost[0, silence_idx]
    for t in range(1, T):
        candidates = cost[t - 1, :, None] + trans
        best_prev = candidates.argmin(axis=0)
        cost[t] = candidates.min(axis=0) + emit_cost[t]
        back[t] = best_prev

    # Backtrace.
    path = np.full(T, -1, dtype=np.int32)
    last = int(cost[-1].argmin())
    for t in range(T - 1, -1, -1):
        path[t] = last
        last = int(back[t, last])

    # Remap silence state to -1.
    path = np.where(path == silence_idx, -1, path).astype(np.int32)

    # Per-ref cleanup: merge small silence gaps, keep the earliest
    # qualifying run near each ref's cue.
    cues_by_idx = {i: refs_in_u[i].cue_s for i in range(K)}
    return _clean_path(path, K, times=times, cues_by_idx=cues_by_idx)


# ---------- path cleanup ----------------------------------------------------

def _runs_of(path: np.ndarray, label: int) -> list[tuple[int, int]]:
    """Contiguous runs of `path == label` as `[(start, end_excl), ...]`."""
    mask = (path == label).astype(np.int8)
    if mask.size == 0:
        return []
    padded = np.concatenate(([0], mask, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _clean_path(
    path: np.ndarray,
    K: int,
    *,
    times: np.ndarray,
    cues_by_idx: dict[int, float],
) -> np.ndarray:
    """Post-process a Viterbi path:

      1. PRE-PRUNE: strip runs of any ref j that can't pass its own
         cue-proximity test (run start > _CUE_TOLERANCE_S away from
         cue_j). Those cells are already guaranteed to be wiped by
         j's own cleanup, so treating them as effectively-SILENCE now
         lets ref i's merge step bridge across them. Without this,
         a competing ref's single stray cell in the middle of ref i's
         real play window breaks the merge (requires gap all-SILENCE)
         and ref i gets fragmented into sub-threshold runs.
      2. Per ref, merge runs separated by SILENCE-only gaps ≤ _MERGE_GAP_M
         (where SILENCE is judged after the pre-prune).
      3. Per ref, keep the EARLIEST merged run (≥ _MIN_DURATION_M) whose
         start is within _CUE_TOLERANCE_S of the scraped cue. Later runs →
         spurious re-entry → wiped. Earlier-but-far-from-cue runs also
         wiped (belt-and-braces against the cue gate).
      4. Drop the ref entirely if no run qualifies.

    Never overrides another ref's frames — only fills SILENCE cells or
    clears spurious ref frames back to SILENCE.
    """
    # Precompute, per ref j, the run-starts whose distance to cue_j
    # exceeds _CUE_TOLERANCE_S. These cells are guaranteed to be wiped
    # by j's own cleanup, so when bridging ref i's merge gap we treat
    # them as effectively-SILENCE. We do NOT strip ref i's own cells
    # here — a ref can legitimately have fragmented runs across a wide
    # time range that all belong to one real play (e.g. Bastille loops
    # the same 0:32-1:53 section across the mix at 26s, 106s, 150s).
    doomed_by_ref: dict[int, np.ndarray] = {}
    T_ = len(path)
    for j in range(K):
        cue_j = cues_by_idx[j]
        if cue_j <= 1.0:
            continue
        doomed = np.zeros(T_, dtype=bool)
        for s, e in _runs_of(path, j):
            if abs(float(times[s]) - cue_j) > _CUE_TOLERANCE_S:
                doomed[s:e] = True
        if doomed.any():
            doomed_by_ref[j] = doomed

    out = path.copy()
    for i in range(K):
        # Strip other refs' doomed strays from i's view so that
        # ref i's merge can bridge across them.
        view = path.copy()
        for j, doomed in doomed_by_ref.items():
            if j == i:
                continue
            mask = doomed & (view == j)
            view[mask] = -1

        runs = _runs_of(view, i)
        if not runs:
            continue

        # Merge consecutive runs separated only by SILENCE with gap ≤ threshold.
        merged: list[tuple[int, int]] = []
        cur_s, cur_e = runs[0]
        for s, e in runs[1:]:
            gap = s - cur_e
            gap_cells = view[cur_e:s]
            if gap <= _MERGE_GAP_M and np.all(gap_cells == -1):
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        cue_s = cues_by_idx[i]
        has_cue = cue_s > 1.0  # treat 0 as "no scraped cue available"

        # When a scraped cue is known, prefer the EARLIEST run within
        # ±_CUE_TOLERANCE_S of it (classic earliest-near-cue rule — DJs
        # play a track once, starting near its scraped cue).
        # When NO cue is known (cue_s≈0), that rule degenerates to
        # "must start in the first 80 s of the mix" which wipes every
        # track played later in the set. Fall back to the LONGEST run
        # meeting _MIN_DURATION_M — that's the best we can do without
        # a positional prior.
        chosen: tuple[int, int] | None = None
        if has_cue:
            for s, e in merged:
                if e - s < _MIN_DURATION_M:
                    continue
                run_start_s = float(times[s])
                if abs(run_start_s - cue_s) > _CUE_TOLERANCE_S:
                    continue
                chosen = (s, e)
                break
        else:
            viable = [(s, e) for s, e in merged if e - s >= _MIN_DURATION_M]
            if viable:
                chosen = max(viable, key=lambda r: r[1] - r[0])

        # Reset all cells assigned to ref i back to SILENCE, then stamp
        # the chosen run (only where currently SILENCE).
        out[path == i] = -1
        if chosen is None:
            continue
        for t in range(chosen[0], chosen[1]):
            if out[t] == -1:
                out[t] = i
    return out
