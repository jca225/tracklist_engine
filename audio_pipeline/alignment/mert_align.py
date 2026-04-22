"""Measure-level alignment via MERT embeddings.

Replaces chroma cross-similarity with cosine similarity in MERT
(Music Encoder Representation via Transformers) embedding space.
The case for this change:

* Chroma is a 12-dimensional projection onto pitch classes. It
  encodes "what chord is playing" but nothing about "who is playing
  it." Two different vocalists singing the same chord progression
  — exactly the Big Bootie 11 Call Me Maybe / How to Save a Life
  confusion — produce chroma profiles that are nearly identical
  beat-by-beat.
* MERT-v1 is a 95M-parameter transformer trained with masked
  acoustic modelling on 160k hours of music. Its hidden states
  carry 768-dim vectors per ~13ms frame that encode pitch,
  rhythm, timbre, and voice identity jointly. Two different
  singers at similar pitch sit at very different MERT coordinates.

Granularity: per-measure, pooled from MERT's native ~75Hz frames.
Each measure's embedding is the mean of the frames that land
inside it on the track's `beat_this` measure grid. That gives one
768-dim vector per musical measure, which is the right time-scale
for "which ref measure played at which mix measure" — alignment's
actual question.

Caching: per-track embeddings are expensive (~1-2s/10s audio on
MPS) but deterministic given audio+layer. Store as measure-indexed
.npz under `data/cache/mert/` so the first alignment pays the cost
and subsequent ones are instant. The mix is embedded once per set
and reused across rows.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..result import Err, Ok, Result
from .errors import AlignmentError


# Cache root — same root as features_cache uses, so all pipelines
# have one cache tree to manage.
_CACHE_DIR = Path("data/cache/mert")

# Which MERT layer to use. Layer 6 is near the mid-stack and
# experimentally balances low-level acoustic content (layers 1-3)
# against task-specific features the top layers accumulate. The
# MERT paper shows mid-layers transfer best to music-ID tasks; we
# pick 6 as the centre of the 12-layer stack.
DEFAULT_LAYER: int = 6

# Thresholds for the fragment finder on MERT similarity. MERT cosine
# sits higher than CENS chroma on real matches (the 768-dim space
# has more room for different vectors to point the same direction
# when they're genuinely similar). 0.80 separates real plays (0.85+)
# from spurious cross-track overlap (0.75 or below). Tuned on a
# spike test against the Big Bootie 11 CMM case.
DEFAULT_MIN_LENGTH_MEASURES: int = 2       # 2 bars at 4/4 — shortest DJ cut
DEFAULT_SMOOTHING_MEASURES: int = 2        # 2-measure moving average
DEFAULT_MIN_MEAN_SIMILARITY: float = 0.75   # edge-of-play phrases dip to 0.75-0.80


@dataclass(frozen=True)
class MertMeasureMatch:
    """A contiguous high-similarity diagonal in the MERT
    similarity matrix — same shape as chroma-CCC's DiagonalMatch
    but indexed on measures (not beats) and without the 12-shift
    business (MERT isn't pitch-class-shift-invariant; pitch-shifted
    vocals produce slightly different MERT vectors and the search
    naturally handles that via the similarity threshold)."""
    ref_measure_start: int
    ref_measure_end: int           # inclusive
    mix_measure_start: int
    mix_measure_end: int           # inclusive
    mean_similarity: float

    @property
    def length_measures(self) -> int:
        return self.ref_measure_end - self.ref_measure_start + 1


def _cache_key(audio_path: Path, layer: int) -> str:
    """Stable cache key per (audio_file, layer)."""
    stat = audio_path.stat()
    raw = f"{audio_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|layer={layer}"
    return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.npz"


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


# Lazy, process-lifetime MERT handle. The model weights are ~400MB
# on disk and ~1.5GB resident; loading is expensive so we do it
# once and reuse across rows. Kept private to this module so tests
# can import-bomb the module without triggering the load.
_mert_handle: object | None = None


def _get_handle() -> object:
    """Load MERT lazily and cache on the module.

    Importing this module does not load MERT — callers only pay
    the load cost when they actually compute embeddings. Useful
    for the eval harness and tests which can import the module
    without touching the heavy model.
    """
    global _mert_handle
    if _mert_handle is None:
        from ..analysis.adapters import mert_adapter
        r = mert_adapter.load()
        if not r.is_ok():
            raise RuntimeError(f"MERT load failed: {r.error}")
        _mert_handle = r.value
    return _mert_handle


def compute_frame_embeddings(
    audio_path: Path,
    *,
    offset_s: float = 0.0,
    duration_s: float | None = None,
    layer: int = DEFAULT_LAYER,
    chunk_s: float = 10.0,
) -> Result[tuple[np.ndarray, int], AlignmentError]:
    """Full-file MERT embeddings at the model's native frame rate.

    Returns `(embeddings, frame_rate_hz)` where embeddings has shape
    `(T, D)` and `frame_rate_hz` is the number of embedding frames
    per second — used by the caller to map measure times to frame
    indices. On MERT-v1-95M this is ~75 Hz.

    Audio is loaded at MERT's required 24 kHz, chunked (MERT-v1 was
    trained on 10s clips and quadratic-attention's out past ~30s),
    and chunk embeddings are concatenated along time.
    """
    try:
        from ..analysis.adapters.mert_adapter import MERT_SR
        import librosa
        import torch
    except ImportError as e:
        return Err(AlignmentError(kind="dtw_failed", detail=f"mert runtime: {e}"))

    handle = _get_handle()
    # librosa's audioread fallback path chokes on offset=None, so
    # pass 0.0 explicitly. `duration=None` IS accepted — it means
    # "read to end" — so we keep that branch.
    try:
        y, _ = librosa.load(
            str(audio_path), sr=MERT_SR, mono=True,
            offset=float(offset_s or 0.0),
            duration=float(duration_s) if duration_s else None,
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        return Err(AlignmentError(kind="dtw_failed", detail=f"mert load: {e}"))
    if y.size < MERT_SR // 10:
        return Err(AlignmentError(kind="dtw_failed", detail="mert: audio too short"))

    chunk_size = int(chunk_s * MERT_SR)
    pieces: list[np.ndarray] = []
    try:
        for i in range(0, y.size, chunk_size):
            chunk = y[i:i + chunk_size]
            if chunk.size < MERT_SR // 10:
                continue
            inputs = handle._processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
            inputs = {k: v.to(handle.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = handle._model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer]   # (1, T, D)
            pieces.append(hidden.squeeze(0).to("cpu").to(torch.float32).numpy())
    except (RuntimeError, ValueError) as e:
        return Err(AlignmentError(kind="dtw_failed", detail=f"mert inference: {e}"))

    if not pieces:
        return Err(AlignmentError(kind="dtw_failed", detail="mert: no chunks"))

    arr = np.concatenate(pieces, axis=0)
    # MERT-v1 frame rate: model has stride ~320 at 24 kHz = 75 Hz.
    frame_rate = arr.shape[0] / (y.size / MERT_SR)
    return Ok((arr, int(round(frame_rate))))


def pool_to_measures(
    frame_embeddings: np.ndarray,
    frame_rate_hz: int,
    measures: list[tuple[int, float, float, float | None]],
    *,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Mean-pool frame-level embeddings into per-measure vectors.

    For each measure `[start_s, end_s]` in the supplied grid,
    average the MERT frames whose timestamps fall inside. Measures
    with no frames (past the end of the embedded audio) get zero
    vectors — cosine similarity against them is undefined but
    treated as no-match by the downstream find_matches thresholds.

    Returns shape `(N_measures, D)`, L2-normalised per measure for
    direct cosine via matrix multiplication.
    """
    D = frame_embeddings.shape[1]
    out = np.zeros((len(measures), D), dtype=np.float32)
    base = float(offset_s)
    for i, (_midx, ms, me, _bpm) in enumerate(measures):
        frame_lo = max(0, int((ms - base) * frame_rate_hz))
        frame_hi = min(frame_embeddings.shape[0], int((me - base) * frame_rate_hz))
        if frame_hi <= frame_lo:
            continue
        out[i] = frame_embeddings[frame_lo:frame_hi].mean(axis=0)

    # L2 normalise. Zero rows stay zero (norm ≈ 0, division by ε).
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / (norms + 1e-9)
    return out.astype(np.float32)


def _cache_measure_embeddings(
    audio_path: Path,
    measures: list[tuple[int, float, float, float | None]],
    layer: int = DEFAULT_LAYER,
    *,
    offset_s: float = 0.0,
    duration_s: float | None = None,
) -> Result[np.ndarray, AlignmentError]:
    """Cached compute: (N_measures, D) MERT embeddings for one audio
    file + measure grid + layer combination."""
    # Measures grid is part of the cache key — changing the grid
    # invalidates prior caches.
    grid_hash = hashlib.blake2b(
        np.array([(m[1], m[2]) for m in measures], dtype=np.float64).tobytes(),
        digest_size=8,
    ).hexdigest()
    key = f"{_cache_key(audio_path, layer)}_grid={grid_hash}_off={offset_s:.2f}_dur={duration_s or 0:.2f}"
    path = _cache_path(key)
    if path.exists():
        try:
            npz = np.load(path)
            return Ok(npz["emb"].astype(np.float32))
        except (OSError, ValueError, KeyError):
            pass   # cache corruption — re-compute

    frames_r = compute_frame_embeddings(
        audio_path, offset_s=offset_s, duration_s=duration_s, layer=layer,
    )
    if not frames_r.is_ok():
        return frames_r
    frames, fr_hz = frames_r.value
    embeddings = pool_to_measures(frames, fr_hz, measures, offset_s=offset_s)
    try:
        tmp = path.with_suffix(".npz.tmp")
        np.savez_compressed(tmp, emb=embeddings)
        tmp.replace(path)
    except OSError:
        pass   # cache write failure — compute worked, just non-persistent
    return Ok(embeddings)


def _cross_similarity(
    ref_emb: np.ndarray, mix_emb: np.ndarray,
) -> np.ndarray:
    """Cosine similarity matrix between (N_ref, D) and (N_mix, D)
    measure-pooled MERT embeddings. Returns `(N_ref, N_mix)`. Rows
    and columns are already L2-normalised by `pool_to_measures` so
    this is just a matmul."""
    return (ref_emb @ mix_emb.T).astype(np.float32, copy=False)


def _runs_above(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Find contiguous True runs of at least `min_length`. Returns
    `[(start, end_exclusive), ...]`."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]


def find_fragments(
    ref_emb: np.ndarray,
    mix_emb: np.ndarray,
    *,
    min_length_measures: int = DEFAULT_MIN_LENGTH_MEASURES,
    smoothing_measures: int = DEFAULT_SMOOTHING_MEASURES,
    min_mean_similarity: float = DEFAULT_MIN_MEAN_SIMILARITY,
) -> list[MertMeasureMatch]:
    """Find all high-similarity diagonal matches in the (ref, mix)
    MERT similarity matrix — same pattern as
    `correlate.find_matches` but at measure granularity and
    without the 12-shift search (MERT is roughly pitch-invariant
    compared to chroma, so rolling the embedding axis makes no
    sense)."""
    N_ref = ref_emb.shape[0]
    N_mix = mix_emb.shape[0]
    if N_ref < min_length_measures or N_mix < min_length_measures:
        return []

    S = _cross_similarity(ref_emb, mix_emb)
    kernel = np.ones(smoothing_measures, dtype=np.float32) / max(1, smoothing_measures)
    shift = max(0, (smoothing_measures - 1) // 2)

    matches: list[MertMeasureMatch] = []
    d_min = -(N_ref - min_length_measures)
    d_max = N_mix - min_length_measures
    for d in range(d_min, d_max + 1):
        i_start = max(0, -d)
        i_end = min(N_ref, N_mix - d)
        if i_end - i_start < min_length_measures:
            continue

        diag = np.diagonal(S, offset=d)
        if diag.size < smoothing_measures:
            continue
        smoothed = np.convolve(diag, kernel, mode="valid")
        above = smoothed >= min_mean_similarity
        if not above.any():
            continue

        for run_start, run_end in _runs_above(above, min_length_measures - max(0, smoothing_measures - 1)):
            diag_start = run_start + shift
            diag_end = run_end + shift - 1
            ref_s = i_start + diag_start
            ref_e = i_start + diag_end
            mix_s = ref_s + d
            mix_e = ref_e + d
            matches.append(MertMeasureMatch(
                ref_measure_start=int(ref_s),
                ref_measure_end=int(ref_e),
                mix_measure_start=int(mix_s),
                mix_measure_end=int(mix_e),
                mean_similarity=float(smoothed[run_start:run_end].mean()),
            ))
    return matches


def select_best_non_overlapping(
    matches: list[MertMeasureMatch],
    min_separation_measures: int = 2,
) -> list[MertMeasureMatch]:
    """Greedy non-overlapping selection on the mix axis, keeping
    the highest-similarity match first."""
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: m.mean_similarity, reverse=True)
    kept: list[MertMeasureMatch] = []
    for m in ordered:
        overlap = False
        for k in kept:
            if (m.mix_measure_start - min_separation_measures <= k.mix_measure_end
                    and k.mix_measure_start - min_separation_measures <= m.mix_measure_end):
                overlap = True
                break
        if not overlap:
            kept.append(m)
    return sorted(kept, key=lambda m: m.mix_measure_start)


def compute_track_measure_embeddings(
    audio_path: Path,
    measures: list[tuple[int, float, float, float | None]],
    layer: int = DEFAULT_LAYER,
) -> Result[np.ndarray, AlignmentError]:
    """Public wrapper for the per-track measure embedding cache.

    Intended call pattern: ref tracks are embedded once (full audio,
    full measure grid) and the result cached for reuse across every
    row that aligns to them. The mix is also embedded once per set.
    """
    return _cache_measure_embeddings(audio_path, measures, layer=layer)


# --- per-row orchestration helpers (used by orchestrator.align_set_mert) ---


@dataclass(frozen=True)
class RowMertResult:
    """Compact per-row MERT output for persistence and reporting."""
    matches: tuple[MertMeasureMatch, ...]
    primary_cluster: tuple[MertMeasureMatch, ...]
    # Mix measure indices (global into set_measures list) mapped to
    # the local embedding rows — used by the row dict builder so
    # measure_alignment rows reference the real set_measure_idx not
    # the windowed sub-index.
    mix_measure_idx_of_local: tuple[int, ...]


def align_row_mert(
    *,
    ref_emb: np.ndarray,              # (N_ref_measures, D), from cache
    mix_emb_full: np.ndarray,         # (N_set_measures, D), from cache
    ref_measures: list[tuple[int, float, float, float | None]],
    set_measures: list[tuple[int, float, float, float | None]],
    set_window_start_s: float,
    set_window_end_s: float,
    cue_anchor_s: float | None = None,
    cue_radius_s: float = 60.0,
    min_length_measures: int = DEFAULT_MIN_LENGTH_MEASURES,
    smoothing_measures: int = DEFAULT_SMOOTHING_MEASURES,
    min_mean_similarity: float = DEFAULT_MIN_MEAN_SIMILARITY,
    ref_cluster_gap_measures: int = 16,     # ~16 bars = a section
) -> RowMertResult:
    """Align one tracklist row to its ref via MERT measure-fragments.

    Slices the mix embedding to the cue-anchored window, finds all
    measure-diagonal fragments above threshold, filters by cue
    proximity, clusters by ref neighbourhood, and picks the cluster
    closest to the scraped cue anchor.
    """
    # Window the mix side to [set_window_start_s, set_window_end_s]
    # by selecting the set_measures whose start_s falls inside.
    mix_indices: list[int] = [
        i for i, (_idx, s, _e, _bpm) in enumerate(set_measures)
        if set_window_start_s <= s < set_window_end_s
    ]
    if not mix_indices:
        return RowMertResult((), (), ())
    mix_emb = mix_emb_full[mix_indices]

    matches = find_fragments(
        ref_emb, mix_emb,
        min_length_measures=min_length_measures,
        smoothing_measures=smoothing_measures,
        min_mean_similarity=min_mean_similarity,
    )
    if not matches:
        return RowMertResult((), (), tuple(mix_indices))

    # Cue hard gate on the mix-measure-start timestamp.
    if cue_anchor_s is not None:
        gated: list[MertMeasureMatch] = []
        for m in matches:
            mix_global = mix_indices[min(m.mix_measure_start, len(mix_indices) - 1)]
            mix_s = float(set_measures[mix_global][1])
            if abs(mix_s - float(cue_anchor_s)) <= cue_radius_s:
                gated.append(m)
        matches = gated
    if not matches:
        return RowMertResult((), (), tuple(mix_indices))

    # Dedup via greedy non-overlapping selection on the mix axis.
    deduped = select_best_non_overlapping(matches)

    # Cluster by ref-axis proximity.
    clustered = _cluster_by_ref_measure(deduped, gap=ref_cluster_gap_measures)
    primary = _pick_primary_cluster(
        clustered, mix_indices, set_measures, cue_anchor_s,
    )

    return RowMertResult(
        matches=tuple(deduped),
        primary_cluster=tuple(sorted(primary, key=lambda m: m.mix_measure_start)),
        mix_measure_idx_of_local=tuple(mix_indices),
    )


def _cluster_by_ref_measure(
    matches: list[MertMeasureMatch],
    *,
    gap: int = 16,
) -> list[list[MertMeasureMatch]]:
    """Group matches whose ref ranges overlap or sit within `gap`
    measures of each other. Same idea as the chroma-path ref clusters
    but at measure granularity."""
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: m.ref_measure_start)
    clusters: list[list[MertMeasureMatch]] = [[ordered[0]]]
    for m in ordered[1:]:
        cluster_max = max(c.ref_measure_end for c in clusters[-1])
        if m.ref_measure_start - cluster_max <= gap:
            clusters[-1].append(m)
        else:
            clusters.append([m])
    return clusters


def _pick_primary_cluster(
    clusters: list[list[MertMeasureMatch]],
    mix_indices: list[int],
    set_measures: list[tuple[int, float, float, float | None]],
    cue_anchor_s: float | None,
) -> list[MertMeasureMatch]:
    """Select the cluster whose members sit closest to the cue on
    the mix axis. Ties broken by aggregate similarity-weighted
    length.

    The cue-proximity filter is the main defence against MERT
    pulling in spurious high-similarity fragments from unrelated
    parts of the track — chroma noise coincidences are rare in MERT
    space but not zero, so we still respect the cue as ground truth.
    """
    if not clusters:
        return []
    if cue_anchor_s is None:
        return max(clusters, key=lambda cl: sum(m.mean_similarity * m.length_measures ** 0.5 for m in cl))

    def _dist(cl: list[MertMeasureMatch]) -> float:
        return min(
            abs(
                float(set_measures[mix_indices[min(m.mix_measure_start, len(mix_indices) - 1)]][1])
                - float(cue_anchor_s)
            )
            for m in cl
        )

    def _score(cl: list[MertMeasureMatch]) -> float:
        return sum(m.mean_similarity * m.length_measures ** 0.5 for m in cl)

    return min(clusters, key=lambda cl: (_dist(cl), -_score(cl)))


def matches_to_measure_alignment_rows(
    result: RowMertResult,
    *,
    ref_track_id: str,
    set_measures: list[tuple[int, float, float, float | None]],
    ref_measures: list[tuple[int, float, float, float | None]],
    stem_hint: str | None = None,
) -> list[dict]:
    """Flatten primary-cluster matches into measure_alignment rows.

    Each (mix_measure, ref_measure) pair on the diagonal produces
    one row with pitch_shift=0 (MERT alignment is pitch-invariant —
    downstream render resolves pitch via a separate chroma shift
    detector when needed) and tempo_ratio derived from the actual
    measure durations on each side.
    """
    if not result.primary_cluster:
        return []

    stem_mask_list = _stem_mask_for_version(stem_hint)
    rows: list[dict] = []
    for m in result.primary_cluster:
        # Walk the measure diagonal, emit one row per measure pair.
        length = m.ref_measure_end - m.ref_measure_start + 1
        for k in range(length):
            rm_local = m.ref_measure_start + k
            mm_local = m.mix_measure_start + k
            if rm_local >= len(ref_measures):
                break
            if mm_local >= len(result.mix_measure_idx_of_local):
                break
            rm_global = ref_measures[rm_local][0]
            mm_global = result.mix_measure_idx_of_local[mm_local]
            set_m = set_measures[mm_global] if mm_global < len(set_measures) else None
            ref_m = ref_measures[rm_local]
            if set_m is None:
                continue
            # tempo_ratio = ref_dur / mix_dur on this specific pair.
            r_dur = max(1e-6, ref_m[2] - ref_m[1])
            m_dur = max(1e-6, set_m[2] - set_m[1])
            tempo_ratio = r_dur / m_dur
            rows.append({
                "set_measure_idx": int(set_m[0]),
                "ref_track_id": ref_track_id,
                "ref_measure_idx": int(rm_global),
                "pitch_shift_semi": 0,
                "tempo_ratio": float(tempo_ratio),
                "stem_mask": stem_mask_list,
                "gain_db": None,
                "confidence": float(m.mean_similarity),
            })
    return rows


def _stem_mask_for_version(stem_hint: str | None) -> list[str]:
    """Version tag → stem-mask list stored in measure_alignment rows."""
    if stem_hint == "acappella":
        return ["vocals"]
    if stem_hint == "instrumental":
        return ["drums", "bass", "other"]
    return ["full"]
