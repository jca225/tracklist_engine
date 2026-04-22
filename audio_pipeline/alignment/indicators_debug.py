"""SOTA Viterbi-based DJ-mix alignment — validated debug pipeline.

============================================================================
 THIS FILE IS THE CURRENT STATE-OF-THE-ART on BB11 (mean IoU 0.891).
 Dropped experiments live in `_archive/` — do NOT resurrect them without
 re-running `tests/fixtures/*_ground_truth.yaml` and beating the baseline.
============================================================================

SOTA signal stack (in order of application):

    1. Per-ref MERT cosine similarity matrix (N_ref, N_mix), stem-routed
       by version_tag so acapella refs compare on the mix's vocal stem,
       instrumental refs on the instrumental stem, full refs on the mix.
    2. max_n sim  → emission for per-universe Viterbi (Phase 1)
         * states = {ref_1..ref_K, SILENCE} per universe (mutual exclusion)
         * emission = w1·persistence(sim − pre_cue_baseline)
                    + w2·ATR-normalised MACD histogram
                    + w3·cross-sectional-z within universe
    3. Chromaprint fingerprint anchors within-universe (Phase 5) —
       confirmed density clusters subtract from ref emission cost.
    4. Cross-universe full-track exclusion (Phase 6) — a fingerprint-
       confirmed full ref forces SILENCE in instrumental/acapella universes.
    5. Earliest-run-near-cue post-process cleanup — DJ plays each track
       once, near its scraped cue, no re-entry.
    6. Per-ref monotonic Viterbi over ref measures (ref_position_viterbi)
       — states = ref_measure_idx, monotone-via-transition-priors. Same
       family as production alignment/measure_dtw.py. Replaces naive
       argmax which violates monotonicity.
    7. Canonical cue-detr cues (canonical_track_cue_points, one per song,
       computed on the full-song audio at sensitivity 0.5) bracket the
       Viterbi-traced ref_t range. The implied mix-side shift is applied
       to produce the final prediction.

Dropped experiments (see _archive/README.md for full rationale + data):

    - Phase 2 : MACD crossover transition bonuses          (neutral, 0.872)
    - Phase 2': Wilder ADXR/DMI trust gate + entry/exit locks (degraded)
    - Phase 7 : per-ref BPM matching penalty                (broke on DJ tempo-shift)
    - argmax-based ref_t inference                         (non-monotonic → fake cues)

Run:
    venvs/audio/bin/python -m audio_pipeline.alignment.indicators_debug

Canonical cues for BB11 are populated by:
    venvs/audio/bin/python -m audio_pipeline.analysis.canonical_cues
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .mert_align import compute_frame_embeddings, pool_to_measures


# ---------- configuration ---------------------------------------------------

SET_ID: str = "2nvzlh2k"
DB_PATH: Path = Path("data/db/music_database.db")
GT_PATH: Path = Path("tests/fixtures/bigbootie11_ground_truth.yaml")
DEBUG_DIR: Path = Path("data/debug")

# Focus window. Set to a finite number of seconds to debug a section, or
# None to process the full mix (reads duration from set_audio).
MIX_DURATION_S: float | None = None

# Stem routing per GT track. Mirrors the legend in bb11_mert_gt.png so
# these plots are comparable to it.
@dataclass(frozen=True)
class GtRef:
    label: str
    track_id: str             # canonical track id (used for fingerprint hits lookup)
    track_audio_id: int
    version_tag: str          # instrumental | acappella | full
    color: str
    cue_s: float
    gt_start_s: float
    gt_end_s: float

# GT refs with hand-annotated variant/span from
# tests/fixtures/bigbootie11_ground_truth.yaml. Used (a) for IoU scoring
# and (b) as overrides when dynamically loading all refs from the DB.
# Populated in main() from `_GT_OVERLAY` — SOTA MERT alignment runs on
# hand-GT refs only (what we can actually score). Other scraped tracks
# are persisted as cue-based fallbacks by
# `audio_pipeline/alignment/populate_cue_fallbacks.py`.
REFS: tuple[GtRef, ...] = ()


def _resolve_track_audio_id(track_id: str) -> int:
    """Preferred audio variant for a canonical track_id: 'original' first,
    any else. Kept tiny so we can call it at module load / test time."""
    conn = sqlite3.connect(DB_PATH)
    try:
        r = conn.execute(
            """
            SELECT track_audio_id FROM track_audio
            WHERE track_id=?
            ORDER BY variant_tag='original' DESC, track_audio_id
            LIMIT 1
            """, (track_id,),
        ).fetchone()
        return int(r[0]) if r else 0
    finally:
        conn.close()


_GT_OVERLAY: dict[str, tuple[str, int, str, str, float, float, float]] = {
    # track_id: (label, track_audio_id, version_tag, color, cue_s, gt_start, gt_end)
    # track_audio_id is pinned to the variant with demucs stems already
    # computed — Fray's new 'original' download (ta=122) has no stems
    # yet, so we keep ta=3 (scraped acappella) for MERT comparison.
    "g8gtgdx":  ("Bastille - Good Grief (instr)",        2, "instrumental", "#1f77b4",  25.0,  25.0, 187.0),
    "26b4gz6f": ("The Fray - How to Save a Life (acap)", 3, "acappella",    "#2ca02c",  40.0,  40.0,  87.0),
    "4gy6y1p":  ("Carly Rae Jepsen - Call Me M (acap)",  4, "acappella",    "#9467bd",  86.0,  86.0, 101.0),
    "2m5wh0t5": ("Gnash - I Hate U, I Love U (acap)",    5, "acappella",    "#e377c2", 135.0, 135.0, 172.0),
    "ntm7wqx":  ("Antoine Delvig & Paul Vinx (full)",    6, "full",         "#17becf", 189.0, 187.0, 219.0),
}


def _load_all_bb_refs(conn: sqlite3.Connection, set_id: str) -> tuple[GtRef, ...]:
    """Dynamically load every scraped track in the set that has:
    (a) downloaded audio (track_audio row with variant_tag='original'
        preferred, else any variant), and (b) MERT-analysable measures.
    For tracks we have hand-GT for, overlay variant + cue + gt spans
    from `_GT_OVERLAY`. Others get default variant='full', cue_s=0, and
    no GT (they won't participate in IoU scoring)."""
    # Pull tokenized cue data via the big_bootie helper so we can seed
    # cue_s from the scraped tracklist for non-GT refs too.
    import sys
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    for p in (_REPO_ROOT, _REPO_ROOT / "data_analysis"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from big_bootie import load_big_bootie_rows, tokenize_rows, _resolve_cue_sections
        rows_df = load_big_bootie_rows(conn)
        tokens = tokenize_rows(rows_df[rows_df["set_id"] == set_id])
        cue_by_tid: dict[str, float] = {}
        for _, t in tokens[tokens["row_kind"] == "track"].iterrows():
            tid = t.get("track_key")
            cue = t.get("cue_seconds_section") or t.get("cue_seconds")
            if tid and cue and cue > 0 and tid not in cue_by_tid:
                cue_by_tid[str(tid)] = float(cue)
    except (ImportError, Exception):
        cue_by_tid = {}

    track_audio_rows = conn.execute(
        """
        SELECT DISTINCT ta.track_id, ta.track_audio_id, ta.variant_tag, ta.path
        FROM track_audio ta
        JOIN dj_set_track_media_links tml
             ON tml.track_id = ta.track_id AND tml.set_id = ?
        WHERE ta.path IS NOT NULL
          AND EXISTS (SELECT 1 FROM track_measures tm WHERE tm.track_audio_id = ta.track_audio_id)
        ORDER BY ta.track_id, ta.variant_tag = 'original' DESC, ta.track_audio_id
        """,
        (set_id,),
    ).fetchall()

    # Dedup per track_id, preferring variant_tag='original'.
    seen: dict[str, sqlite3.Row] = {}
    for r in track_audio_rows:
        if r["track_id"] not in seen:
            seen[r["track_id"]] = r

    refs: list[GtRef] = []
    # Colors via matplotlib's tab20 for non-GT refs.
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    for i, (tid, r) in enumerate(sorted(seen.items())):
        if tid in _GT_OVERLAY:
            # GT ref: use the pinned track_audio_id (has demucs stems for
            # the acappella/instrumental variant routing the DJ played).
            label, ta_id, vtag, color, cue, gts, gte = _GT_OVERLAY[tid]
        else:
            # Non-GT ref: use the 'original' variant and compare against
            # the full mix (no stems needed — version_tag='full' routes to
            # the full audio path rather than a demucs stem).
            ta_id = int(r["track_audio_id"])
            label = tid
            vtag = "full"
            color = matplotlib.colors.to_hex(cmap(i % 20))
            cue = cue_by_tid.get(tid, 0.0)
            gts, gte = 0.0, 0.0    # no GT available → won't be scored
        refs.append(GtRef(
            label=label, track_id=tid, track_audio_id=ta_id,
            version_tag=vtag, color=color, cue_s=cue,
            gt_start_s=gts, gt_end_s=gte,
        ))
    return tuple(refs)


# ---------- data loading ----------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _mix_stem_path(conn: sqlite3.Connection, set_audio_id: int, stem_name: str) -> Path:
    r = conn.execute(
        "SELECT path FROM set_stems WHERE set_audio_id=? AND stem_name=?",
        (set_audio_id, stem_name),
    ).fetchone()
    if r is None:
        raise RuntimeError(f"mix stem {stem_name!r} missing for set_audio_id={set_audio_id}")
    return Path(r["path"])


def _track_stem_path(conn: sqlite3.Connection, track_audio_id: int, stem_name: str) -> Path:
    r = conn.execute(
        "SELECT path FROM track_stems WHERE track_audio_id=? AND stem_name=?",
        (track_audio_id, stem_name),
    ).fetchone()
    if r is None:
        raise RuntimeError(f"track stem {stem_name!r} missing for track_audio_id={track_audio_id}")
    return Path(r["path"])


def _track_full_path(conn: sqlite3.Connection, track_audio_id: int) -> Path:
    r = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=?", (track_audio_id,)
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


def _mix_measures(conn: sqlite3.Connection, set_audio_id: int, max_s: float | None) -> list[tuple[int, float, float, float | None]]:
    if max_s is None:
        rows = conn.execute(
            "SELECT measure_idx, start_s, end_s, bpm FROM set_measures WHERE set_audio_id=? ORDER BY measure_idx",
            (set_audio_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT measure_idx, start_s, end_s, bpm FROM set_measures WHERE set_audio_id=? AND start_s<? ORDER BY measure_idx",
            (set_audio_id, max_s),
        ).fetchall()
    return [(int(r["measure_idx"]), float(r["start_s"]), float(r["end_s"]), r["bpm"]) for r in rows]


def _track_measures(conn: sqlite3.Connection, track_audio_id: int) -> list[tuple[int, float, float, float | None]]:
    rows = conn.execute(
        "SELECT measure_idx, start_s, end_s, bpm FROM track_measures WHERE track_audio_id=? ORDER BY measure_idx",
        (track_audio_id,),
    ).fetchall()
    return [(int(r["measure_idx"]), float(r["start_s"]), float(r["end_s"]), r["bpm"]) for r in rows]


def _embed_per_measure(audio_path: Path, measures: list[tuple[int, float, float, float | None]], *, duration_s: float | None) -> np.ndarray:
    # Cache at the measure level — same function the production
    # alignment uses. First call writes an .npz under
    # data/cache/mert/<hash>.npz; subsequent calls are an instant load.
    # Essential when sweeping >5 refs because compute_frame_embeddings
    # itself is uncached (MERT model re-infers every call).
    from .mert_align import _cache_measure_embeddings
    r = _cache_measure_embeddings(
        audio_path, measures, offset_s=0.0, duration_s=duration_s,
    )
    if not r.is_ok():
        raise RuntimeError(f"MERT embed failed for {audio_path}: {r.error}")
    return r.value


# ---------- similarity time series ------------------------------------------

def ref_position_viterbi(sim: np.ndarray, *,
                         stay_cost: float = 0.0,
                         advance_cost: float = 0.0,
                         jump_cost: float = 0.3,
                         backward_cost: float = 2.0) -> np.ndarray:
    """Monotonic ref-position Viterbi over a (N_ref, N_mix) similarity
    matrix. Returns shape (N_mix,) mapping mix measure → ref measure.

    DJ-alignment SOTA pattern (equivalent in structure to subsequence
    DTW — same family as production `measure_dtw.py`):

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


def build_similarity_series(conn: sqlite3.Connection) -> tuple[
    np.ndarray, np.ndarray,
    dict[str, np.ndarray], dict[str, np.ndarray],
    dict[str, np.ndarray], dict[str, np.ndarray],
]:
    """Returns (mix_measure_times_s, mix_measure_idx_array, per_ref_maxsim,
    per_ref_argmax, per_ref_vit_path, per_ref_measure_times).

      * per_ref_maxsim[label]: (M,) — max_n cos(ref[n], mix[m]) per mix m.
      * per_ref_argmax[label]: (M,) — argmax_n of sim at each mix m.
        Per-frame-independent; used for side-by-side comparison only.
      * per_ref_vit_path[label]: (M,) — monotonic ref-position Viterbi
        decode of sim. SOTA ref-position track (same family as
        production `measure_dtw.py`).
      * per_ref_measure_times[label]: (N_ref,) — ref-side centre of
        each ref measure for converting indices to ref-time seconds.
    """
    set_audio = conn.execute(
        "SELECT set_audio_id, duration_s FROM set_audio WHERE set_id=?", (SET_ID,)
    ).fetchone()
    set_audio_id = int(set_audio["set_audio_id"])

    mix_measures = _mix_measures(conn, set_audio_id, MIX_DURATION_S)
    mix_times = np.array([0.5 * (m[1] + m[2]) for m in mix_measures], dtype=np.float64)

    window_label = "full mix" if MIX_DURATION_S is None else f"first {MIX_DURATION_S:.0f}s"
    print(f"[mix] {len(mix_measures)} measures ({window_label})")

    # Cache mix embeddings per stem (only compute each stem once).
    mix_emb_by_stem: dict[str, np.ndarray] = {}

    def _mix_emb(stem: str) -> np.ndarray:
        if stem in mix_emb_by_stem:
            return mix_emb_by_stem[stem]
        if stem == "__full__":
            path = conn.execute(
                "SELECT path FROM set_audio WHERE set_audio_id=?", (set_audio_id,)
            ).fetchone()["path"]
            path = Path(path)
        else:
            path = _mix_stem_path(conn, set_audio_id, stem)
        print(f"[mix] embedding stem={stem}  {path.name}")
        emb = _embed_per_measure(path, mix_measures, duration_s=MIX_DURATION_S)
        mix_emb_by_stem[stem] = emb
        return emb

    per_ref: dict[str, np.ndarray] = {}
    per_ref_argmax: dict[str, np.ndarray] = {}
    per_ref_vit_path: dict[str, np.ndarray] = {}
    per_ref_meas_times: dict[str, np.ndarray] = {}
    for ref in REFS:
        stem = _stem_routing(ref.version_tag)
        ref_path = _track_full_path(conn, ref.track_audio_id) if stem == "__full__" else _track_stem_path(conn, ref.track_audio_id, stem)
        ref_measures = _track_measures(conn, ref.track_audio_id)
        print(f"[ref {ref.label}] measures={len(ref_measures)}, stem={stem}, {ref_path.name}")
        ref_emb = _embed_per_measure(ref_path, ref_measures, duration_s=None)
        mix_emb = _mix_emb(stem)

        sim = ref_emb @ mix_emb.T       # (N_ref, N_mix)
        per_mix = sim.max(axis=0).astype(np.float32)
        argmax = sim.argmax(axis=0).astype(np.int32)
        vit_path = ref_position_viterbi(sim)
        per_ref[ref.label] = per_mix
        per_ref_argmax[ref.label] = argmax
        per_ref_vit_path[ref.label] = vit_path
        per_ref_meas_times[ref.label] = np.array(
            [0.5 * (m[1] + m[2]) for m in ref_measures], dtype=np.float64,
        )
        print(f"[ref {ref.label}] max-sim stats: min={per_mix.min():.3f} mean={per_mix.mean():.3f} max={per_mix.max():.3f}")

    mix_idx = np.array([m[0] for m in mix_measures], dtype=np.int32)
    return mix_times, mix_idx, per_ref, per_ref_argmax, per_ref_vit_path, per_ref_meas_times


# ---------- indicators ------------------------------------------------------

def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with pandas-style span convention
    (alpha = 2/(span+1)). NumPy-only, warm-started on the first value."""
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
    """Wilder's smoothing (RMA): alpha = 1/period. Used for RSI, ATR, ADX."""
    alpha = 1.0 / period
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd(x: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic MACD on a 1D series.

    Returns (macd_line, signal_line, histogram). Fast/slow EMAs use pandas
    span conventions so "12" / "26" match the typical finance-chart settings.
    We interpret the units as "measures" — at ~120 BPM a measure is ~2s so
    MACD(12,26,9) ≈ MACD on a 24s/52s/18s basis.
    """
    fast_e = ema(x, fast)
    slow_e = ema(x, slow)
    m = fast_e - slow_e
    s = ema(m, signal)
    return m, s, (m - s)


def per_ref_z(x: np.ndarray, window: int = 40) -> np.ndarray:
    """z-score of x against its own rolling mean/std. This kills the
    per-ref baseline bias — a ref with a high absolute level can still
    surprise its own history."""
    mu = sma(x, window)
    sd = rolling_std(x, window)
    sd = np.maximum(sd, 1e-6)
    return (x - mu) / sd


def cross_sectional_z(per_ref: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """At each mix measure, z-score across all refs. A positive z means
    'this ref is unusually similar right now compared to the others'."""
    labels = list(per_ref.keys())
    stacked = np.stack([per_ref[l] for l in labels], axis=0)  # (R, M)
    mu = stacked.mean(axis=0, keepdims=True)
    sd = stacked.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    z = (stacked - mu) / sd
    return {l: z[i] for i, l in enumerate(labels)}


def bollinger(x: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(x, window)
    sd = rolling_std(x, window)
    return mid, mid + k * sd, mid - k * sd


def rsi_wilder(x: np.ndarray, period: int = 14) -> np.ndarray:
    diffs = np.diff(x, prepend=x[0])
    gain = np.where(diffs > 0, diffs, 0.0)
    loss = np.where(diffs < 0, -diffs, 0.0)
    ag = wilder_smooth(gain, period)
    al = wilder_smooth(loss, period)
    rs = ag / np.maximum(al, 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def adx_dmi(x: np.ndarray, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional Movement Index adapted to a 1D similarity series.

    In finance ADX uses (high, low, close). With only one channel we use
    first-differences as the directional moves and |diff| as the true range.

        +DM = max(diff, 0)
        -DM = max(-diff, 0)
        TR  = |diff|
        +DI = 100 * Wilder(+DM) / Wilder(TR)
        -DI = 100 * Wilder(-DM) / Wilder(TR)
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = Wilder(DX)

    Interpretation on MERT sim:
      * +DI > -DI with high ADX → ref is trending up → track entering
      * -DI > +DI with high ADX → ref is trending down → track exiting
      * low ADX → no regime, similarity is drifting without a cue
    """
    diffs = np.diff(x, prepend=x[0])
    plus_dm = np.maximum(diffs, 0.0)
    minus_dm = np.maximum(-diffs, 0.0)
    tr = np.abs(diffs)
    tr_s = wilder_smooth(tr, period)
    tr_s = np.maximum(tr_s, 1e-9)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / tr_s
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / tr_s
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)
    adx = wilder_smooth(dx, period)
    return adx, plus_di, minus_di


def atr(x: np.ndarray, period: int = 14) -> np.ndarray:
    return wilder_smooth(np.abs(np.diff(x, prepend=x[0])), period)


# ---------- plotting --------------------------------------------------------

def _gt_bar_axis(ax, times: np.ndarray) -> None:
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.5, len(REFS) - 0.5)
    ax.set_yticks(range(len(REFS)))
    ax.set_yticklabels([r.label[:30] for r in REFS], fontsize=7)
    for i, ref in enumerate(REFS):
        ax.plot([ref.gt_start_s, ref.gt_end_s], [i, i], lw=6, color=ref.color)
        ax.axvline(ref.cue_s, color=ref.color, lw=0.5, ls="--", alpha=0.4)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Mix time (s)")
    ax.set_ylabel("GT", fontsize=8)


def _cue_mask(times: np.ndarray, cue_s: float) -> np.ndarray:
    return times >= cue_s


def plot_ema_zscore(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(len(REFS) + 1, 1, figsize=(14, 2.0 * (len(REFS) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [2] * len(REFS) + [1.2]})
    for ax, ref in zip(axes[:-1], REFS):
        s = per_ref[ref.label]
        mask = _cue_mask(times, ref.cue_s)
        baseline = sma(s, 40)
        z = per_ref_z(s, 40)
        ax2 = ax.twinx()
        ax.plot(times[mask], s[mask], color=ref.color, label="MERT sim", lw=1.0)
        ax.plot(times[mask], baseline[mask], color="gray", ls="--", lw=0.8, label="SMA(40)")
        ax2.plot(times[mask], z[mask], color="red", lw=0.7, alpha=0.7, label="z-score")
        ax2.axhline(0, color="red", lw=0.4, alpha=0.3)
        ax2.axhline(1.5, color="red", lw=0.4, ls=":", alpha=0.5)
        ax.axvspan(ref.gt_start_s, ref.gt_end_s, color=ref.color, alpha=0.12)
        ax.set_ylim(0.4, 1.0)
        ax2.set_ylim(-3, 3)
        ax.set_ylabel(ref.label[:22], fontsize=7)
        ax2.set_ylabel("z", color="red", fontsize=7)
        ax.legend(loc="upper left", fontsize=6)
    _gt_bar_axis(axes[-1], times)
    fig.suptitle("Per-ref MERT sim + SMA(40) baseline + rolling z-score (shaded = GT span)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_macd(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(len(REFS) + 1, 1, figsize=(14, 2.0 * (len(REFS) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [2] * len(REFS) + [1.2]})
    for ax, ref in zip(axes[:-1], REFS):
        s = per_ref[ref.label]
        mask = _cue_mask(times, ref.cue_s)
        m, sig, hist = macd(s)
        ax.plot(times[mask], m[mask], color=ref.color, lw=1.1, label="MACD")
        ax.plot(times[mask], sig[mask], color="black", lw=0.8, ls="--", label="signal(9)")
        # Histogram as bars.
        ax.bar(times[mask], hist[mask], width=1.5, color=np.where(hist[mask] >= 0, "#2ca02c", "#d62728"),
               alpha=0.35, label="MACD-signal")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvspan(ref.gt_start_s, ref.gt_end_s, color=ref.color, alpha=0.12)
        ax.set_ylabel(ref.label[:22], fontsize=7)
        ax.legend(loc="upper left", fontsize=6)
    _gt_bar_axis(axes[-1], times)
    fig.suptitle("MACD(12,26,9) on MERT sim — bullish crossover = track entering, bearish = exiting", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_cross_sectional(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    cs = cross_sectional_z(per_ref)
    fig, (ax_sim, ax_cs, ax_gt) = plt.subplots(3, 1, figsize=(14, 7), sharex=True,
                                               gridspec_kw={"height_ratios": [2, 2, 1.2]})
    for ref in REFS:
        mask = _cue_mask(times, ref.cue_s)
        ax_sim.plot(times[mask], per_ref[ref.label][mask], color=ref.color, lw=0.9, label=ref.label[:28])
        ax_cs.plot(times[mask], cs[ref.label][mask], color=ref.color, lw=1.0)
        ax_sim.axvspan(ref.gt_start_s, ref.gt_end_s, color=ref.color, alpha=0.06)
    ax_cs.axhline(0, color="gray", lw=0.5)
    ax_cs.axhline(1.0, color="red", lw=0.5, ls=":", alpha=0.6)
    ax_sim.set_ylabel("MERT sim")
    ax_sim.legend(loc="upper right", fontsize=6, ncol=2)
    ax_cs.set_ylabel("cross-sectional z")
    _gt_bar_axis(ax_gt, times)
    fig.suptitle("Cross-sectional z-score across cue-active refs — ref 'wins' when z peaks", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_bollinger(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(len(REFS) + 1, 1, figsize=(14, 2.0 * (len(REFS) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [2] * len(REFS) + [1.2]})
    for ax, ref in zip(axes[:-1], REFS):
        s = per_ref[ref.label]
        mask = _cue_mask(times, ref.cue_s)
        mid, up, lo = bollinger(s, 20, 2.0)
        ax.plot(times[mask], s[mask], color=ref.color, lw=1.0, label="sim")
        ax.plot(times[mask], mid[mask], color="black", lw=0.7, ls="--", label="SMA(20)")
        ax.fill_between(times[mask], lo[mask], up[mask], color="gray", alpha=0.2, label="±2σ")
        breakout = mask & (s > up)
        ax.scatter(times[breakout], s[breakout], color="red", s=10, label="breakout ↑")
        ax.axvspan(ref.gt_start_s, ref.gt_end_s, color=ref.color, alpha=0.12)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel(ref.label[:22], fontsize=7)
        ax.legend(loc="upper left", fontsize=6)
    _gt_bar_axis(axes[-1], times)
    fig.suptitle("Bollinger(20, ±2σ) on MERT sim — red dots = upward breakouts", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_rsi(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(len(REFS) + 1, 1, figsize=(14, 2.0 * (len(REFS) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [2] * len(REFS) + [1.2]})
    for ax, ref in zip(axes[:-1], REFS):
        s = per_ref[ref.label]
        mask = _cue_mask(times, ref.cue_s)
        r = rsi_wilder(s, 14)
        ax.plot(times[mask], r[mask], color=ref.color, lw=1.0)
        ax.axhline(70, color="red", lw=0.5, ls=":", label="overbought (70)")
        ax.axhline(30, color="green", lw=0.5, ls=":", label="oversold (30)")
        ax.axhline(50, color="gray", lw=0.4)
        ax.axvspan(ref.gt_start_s, ref.gt_end_s, color=ref.color, alpha=0.12)
        ax.set_ylim(0, 100)
        ax.set_ylabel(ref.label[:22], fontsize=7)
        ax.legend(loc="upper left", fontsize=6)
    _gt_bar_axis(axes[-1], times)
    fig.suptitle("Wilder RSI(14) on MERT sim", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_adx_dmi(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(len(REFS) + 1, 1, figsize=(14, 2.3 * (len(REFS) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [2] * len(REFS) + [1.2]})
    for ax, ref in zip(axes[:-1], REFS):
        s = per_ref[ref.label]
        mask = _cue_mask(times, ref.cue_s)
        a, p, n = adx_dmi(s, 14)
        ax.plot(times[mask], p[mask], color="#2ca02c", lw=1.0, label="+DI")
        ax.plot(times[mask], n[mask], color="#d62728", lw=1.0, label="-DI")
        ax.plot(times[mask], a[mask], color="black", lw=1.2, label="ADX")
        ax.axhline(25, color="gray", lw=0.5, ls=":", alpha=0.6)
        entering = mask & (p > n) & (a > 15)
        exiting = mask & (n > p) & (a > 15)
        ax.fill_between(times, 0, 100, where=entering, color="#2ca02c", alpha=0.08, label="entering")
        ax.fill_between(times, 0, 100, where=exiting, color="#d62728", alpha=0.08, label="exiting")
        ax.axvspan(ref.gt_start_s, ref.gt_end_s, color=ref.color, alpha=0.15)
        ax.set_ylim(0, 70)
        ax.set_ylabel(ref.label[:22], fontsize=7)
        ax.legend(loc="upper left", fontsize=6, ncol=2)
    _gt_bar_axis(axes[-1], times)
    fig.suptitle("Wilder ADX / +DI / -DI on MERT sim — green = up-trend, red = down-trend", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------- Phase 1: per-universe Viterbi with mutual exclusion -------------

# Emission-to-cost weighting: positive MACD histogram and positive
# cross-sectional z push a ref's cost down (i.e. make it preferred). Scales
# chosen so both terms are O(1) after normalisation.
_EMIT_CS_Z_WEIGHT: float = 0.5       # winner-picker within universe
_EMIT_MACD_WEIGHT: float = 1.0       # entry/exit-event signal
_EMIT_PERSIST_WEIGHT: float = 1.5    # persistent "above pre-cue noise floor" signal
# A ref must beat this baseline emission to enter/stay in its state.
# Set high enough that a ref needs strong persistence AND either cs_z OR
# MACD to win — kills phantom spans in the inter-ref silence gaps.
_SILENCE_EMIT: float = 1.2
_SELF_LOOP_COST: float = 0.0         # free self-loop (tracks play contiguously)
_SILENCE_STAY_COST: float = 0.0      # free silence stay
_ENTER_COST: float = 0.4             # SILENCE → ref_i
_EXIT_COST: float = 0.1              # ref_i → SILENCE (cheap; tracks end)
_CROSS_REF_COST: float = 3.0         # ref_i → ref_j direct; discourage, force via SILENCE


# --------- Unidentified-content detector ------------------------------------
# At a measure where the relevant stem has real audible content but the
# Viterbi picked SILENCE for that universe, the tracklist is incomplete:
# something is playing that we don't have a ref for. The canonical BB11
# case is Barenaked Ladies - One Week (acapella, 172-184s) — there's no
# ref for it because it was never downloaded, but the mix vocal stem
# clearly has vocal content there.

# Absolute RMS below this in the stem = effectively silent.
_STEM_SILENCE_RMS: float = 0.01
# Content must be this factor above the local noise floor (per-stem 10th
# percentile across the focus window) to be called "real".
_STEM_CONTENT_FLOOR_MULT: float = 3.0
# Minimum contiguous run of flagged measures to report — kills lone blips.
_UNID_MIN_RUN_M: int = 4


def _compute_stem_rms(audio_path: Path, measures: list[tuple[int, float, float, float | None]]) -> np.ndarray:
    """Per-measure RMS of a stem, at 22 050 Hz mono. Cheap because we
    only read to the focus-window end."""
    import librosa
    try:
        y, _sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=MIX_DURATION_S)
    except (FileNotFoundError, OSError, ValueError):
        return np.zeros(len(measures), dtype=np.float32)
    rms = np.zeros(len(measures), dtype=np.float32)
    sr = 22050
    for i, (_idx, ms, me, _bpm) in enumerate(measures):
        lo = int(ms * sr)
        hi = int(me * sr)
        if hi <= lo or lo >= y.size:
            continue
        seg = y[lo:min(hi, y.size)]
        if seg.size:
            rms[i] = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2)))
    return rms


def _detect_unidentified(
    stem_rms: np.ndarray,
    decoded_path: np.ndarray,
) -> np.ndarray:
    """Per-measure bool mask: True where stem has audible content AND
    the Viterbi picked SILENCE (-1). Runs shorter than _UNID_MIN_RUN_M
    measures are cleared to kill noise-bleed flickers."""
    silence_floor = max(
        float(np.quantile(stem_rms[stem_rms > 0], 0.10)) if (stem_rms > 0).any() else 0.0,
        _STEM_SILENCE_RMS,
    )
    has_content = stem_rms > (silence_floor * _STEM_CONTENT_FLOOR_MULT)
    picked_silence = decoded_path == -1
    raw = has_content & picked_silence

    # Filter: keep only runs ≥ _UNID_MIN_RUN_M.
    out = np.zeros_like(raw)
    if raw.size == 0:
        return out
    padded = np.concatenate(([False], raw, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= _UNID_MIN_RUN_M:
            out[s:e] = True
    return out


# --------- Cue-detr annotation (ref-side span metadata) --------------------
# The optimal pipeline produces (mix_start, mix_end) per ref — mix-side
# times, which is what the eval scores. Separately, the ref's own
# cue-detr cues tell us which musical sections (drops, verses, outros)
# the DJ is likely cueing FROM and TO. We annotate each predicted span
# with (ref_cue_start, ref_cue_end) — the nearest cue point bracketing
# the inferred ref-side range — without touching the mix-side times.


def _load_cue_detr_cues(conn: sqlite3.Connection, track_id: str, track_audio_id: int) -> list[float]:
    """Per-song cue-detr cue points (seconds, on the full-song timeline).
    Prefers `canonical_track_cue_points` (keyed by `track_id`, computed
    once on the original/full variant at a sensitivity tuned for dense
    boundary detection). Falls back to the legacy per-variant
    `track_analysis.cue_points_json` when canonical is missing."""
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


def _load_mix_bpm(
    conn: sqlite3.Connection,
    set_audio_id: int,
    times: np.ndarray,
) -> np.ndarray:
    """Per-mix-measure BPM aligned to `times`. Fills missing values by
    forward-then-backward fill; zeros are treated as missing."""
    rows = conn.execute(
        "SELECT measure_idx, start_s, end_s, bpm FROM set_measures WHERE set_audio_id=? ORDER BY measure_idx",
        (set_audio_id,),
    ).fetchall()
    if not rows:
        return np.full(len(times), np.nan, dtype=np.float64)
    bpms_raw = np.array([r["bpm"] for r in rows], dtype=np.float64)
    starts = np.array([r["start_s"] for r in rows], dtype=np.float64)
    # Match by start_s (nearest)
    out = np.full(len(times), np.nan, dtype=np.float64)
    for i, t in enumerate(times):
        j = int(np.argmin(np.abs(starts - t)))
        b = bpms_raw[j]
        if b and b > 0:
            out[i] = b
    # Forward/backward fill NaNs
    valid = ~np.isnan(out)
    if not valid.any():
        return out
    idx = np.where(valid, np.arange(len(out)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, out[idx], out)
    # backward fill
    valid = ~np.isnan(out)
    idx = np.where(valid, np.arange(len(out)), len(out))
    idx = np.minimum.accumulate(idx[::-1])[::-1]
    out = np.where(idx < len(out), out[np.minimum(idx, len(out) - 1)], out)
    return out


def _load_ref_bpm(conn: sqlite3.Connection, track_audio_id: int) -> float:
    """Robust global BPM for a ref — median across its per-measure BPMs."""
    rows = conn.execute(
        "SELECT bpm FROM track_measures WHERE track_audio_id=? AND bpm > 0",
        (track_audio_id,),
    ).fetchall()
    if not rows:
        return float("nan")
    return float(np.median([r["bpm"] for r in rows]))


def _bpm_penalty_folded(mix_bpm: np.ndarray, ref_bpm: float) -> np.ndarray:
    """|log2(mix_bpm / ref_bpm)| folded onto [0, 0.5] so that 2:1 and 1:2
    ratios count as matches (DJs play tracks at half-/double-tempo via
    halftime/doubletime tricks). Returns 0 at perfect match, 0.5 at the
    worst possible mismatch inside the folded interval."""
    if not np.isfinite(ref_bpm) or ref_bpm <= 0:
        return np.zeros_like(mix_bpm)
    ratio = np.where(mix_bpm > 0, mix_bpm / ref_bpm, 1.0)
    log_r = np.log2(np.maximum(ratio, 1e-6))
    # Fold into [-0.5, 0.5]
    log_r = log_r - np.round(log_r)
    return np.abs(log_r)


def _load_fingerprint_anchors(
    conn: sqlite3.Connection,
    set_id: str,
    times: np.ndarray,
    refs: tuple[GtRef, ...],
) -> dict[str, np.ndarray]:
    """For each ref, return a per-measure bool mask: True where the
    fingerprint hit density is >= _FP_MIN_DENSITY within ±_FP_DENSITY_WINDOW_S
    at that measure. Uses all variants — filtering by score alone is
    sufficient noise suppression; the clustering does the rest.
    """
    out: dict[str, np.ndarray] = {}
    for ref in refs:
        rows = conn.execute(
            """SELECT mix_start_s FROM set_fingerprint_hits
               WHERE set_id=? AND matched_track_id=? AND score>=?""",
            (set_id, ref.track_id, _FP_MIN_SCORE),
        ).fetchall()
        if not rows:
            out[ref.label] = np.zeros(len(times), dtype=bool)
            continue
        starts = np.array([r["mix_start_s"] for r in rows], dtype=np.float64)
        density = np.zeros(len(times), dtype=np.int32)
        for i, t in enumerate(times):
            density[i] = int(np.sum(np.abs(starts - t) <= _FP_DENSITY_WINDOW_S))
        out[ref.label] = density >= _FP_MIN_DENSITY
    return out


def _full_exclusion_mask(
    times: np.ndarray,
    anchors_by_label: dict[str, np.ndarray],
    refs: tuple[GtRef, ...],
) -> np.ndarray:
    """Union of confirmed masks for all `full`-variant refs. At any
    measure where this is True, instrumental and acapella universes are
    forced to SILENCE."""
    out = np.zeros(len(times), dtype=bool)
    for ref in refs:
        if ref.version_tag == "full":
            mask = anchors_by_label.get(ref.label)
            if mask is not None:
                out |= mask
    return out


def _universe(tag: str) -> str:
    """Which mutual-exclusion group a ref belongs to. Acapellas never
    overlay each other, instrumentals never overlay each other, full
    tracks are their own class."""
    return {"acappella": "acapella", "instrumental": "instrumental", "full": "full"}[tag]


def _within_universe_cs_z(per_ref: dict[str, np.ndarray], refs_in_u: list[GtRef]) -> dict[str, np.ndarray]:
    """Cross-sectional z-score computed only across refs in the same
    universe — using all refs (including e.g. Bastille in the acap pool)
    would contaminate the score."""
    if len(refs_in_u) == 1:
        # With one ref, cross-sectional z is degenerate. Fall back to a
        # per-ref rolling z-score — "this ref is unusually similar right
        # now compared to its own recent history". Goes negative when
        # the ref genuinely isn't playing, which is what we need for
        # the Viterbi to exit back to SILENCE.
        s = per_ref[refs_in_u[0].label]
        return {refs_in_u[0].label: per_ref_z(s, 40)}
    stacked = np.stack([per_ref[r.label] for r in refs_in_u], axis=0)
    mu = stacked.mean(axis=0, keepdims=True)
    sd = np.maximum(stacked.std(axis=0, keepdims=True), 1e-6)
    z = (stacked - mu) / sd
    return {r.label: z[i] for i, r in enumerate(refs_in_u)}


def adxr(sim: np.ndarray, period: int = 14, lag: int = 14) -> np.ndarray:
    """Wilder's Average Directional Movement Rating: ADXR[t] = (ADX[t] +
    ADX[t − lag]) / 2. A smoothed / confirmed ADX — Wilder's empirical
    trust thresholds (25 and 20) were calibrated on this lagged version,
    not raw ADX. On decaying-start boundaries, ADX[t-lag] falls back to
    ADX[0]."""
    adx_arr, _, _ = adx_dmi(sim, period)
    T = len(adx_arr)
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        out[t] = 0.5 * (adx_arr[t] + adx_arr[max(0, t - lag)])
    return out


def _trust_gate(adxr_arr: np.ndarray) -> np.ndarray:
    """Linear ramp: 0 below _ADXR_TRUST_LOW, 1 above _ADXR_TRUST_HIGH.
    Multiplies the composite emission score so no evidence accumulates
    during non-trending / chopping regions of the MERT sim series."""
    return np.clip(
        (adxr_arr - _ADXR_TRUST_LOW) / (_ADXR_TRUST_HIGH - _ADXR_TRUST_LOW),
        0.0, 1.0,
    )


def _find_sustained(mask: np.ndarray, start: int, sustain: int) -> int | None:
    """Earliest t ≥ start where mask[t : t + sustain] is all True. Returns
    t + sustain — the measure AFTER the sustained window, i.e. the first
    measure at which we want to apply the lock. None if no such window."""
    start = max(0, start)
    T = len(mask)
    if T < sustain:
        return None
    for t in range(start, T - sustain + 1):
        if mask[t : t + sustain].all():
            return t + sustain
    return None


def macd_crossovers(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect definitive bullish and bearish MACD-histogram sign flips.

    A flip is considered 'definitive' at time t if:
      * hist[t] > +_MACD_FLIP_EPS AND min(hist[t-L..t-1]) < -_MACD_FLIP_EPS  (bullish)
      * hist[t] < -_MACD_FLIP_EPS AND max(hist[t-L..t-1]) > +_MACD_FLIP_EPS  (bearish)
    The lookback filters out chatter right around zero — a plain sign
    flip on MERT-sim MACD fires many false positives because the
    histogram oscillates inside ±0.001.

    Returns (bullish_mask, bearish_mask), both bool[T].
    """
    _, _, hist = macd(sim)
    T = len(hist)
    bull = np.zeros(T, dtype=bool)
    bear = np.zeros(T, dtype=bool)
    for t in range(1, T):
        lo = max(0, t - _MACD_FLIP_LOOKBACK)
        window = hist[lo:t]
        if window.size == 0:
            continue
        if hist[t] > _MACD_FLIP_EPS and window.min() < -_MACD_FLIP_EPS:
            bull[t] = True
        elif hist[t] < -_MACD_FLIP_EPS and window.max() > _MACD_FLIP_EPS:
            # Wilder-style confirmation: require the bearish side to
            # persist for _BEAR_CONFIRM_M measures. Single flip-flops
            # during play don't count as cue-out events.
            hi = min(T, t + _BEAR_CONFIRM_M)
            if np.mean(hist[t:hi] < 0.0) >= 0.7:
                bear[t] = True
    return bull, bear


def _emission_score(sim: np.ndarray, cs_z: np.ndarray, times: np.ndarray, cue_s: float) -> np.ndarray:
    """Per-measure 'is this ref playing' score, Wilder-gated.

    Composite of three complementary signals (unchanged from Phase 1):
      1. persistence: (sim - pre_cue_baseline) — stays positive for the
         duration of a play, not just at entry.
      2. MACD histogram (ATR-normalised) — sharp entry/exit events.
      3. cross-sectional z within universe — picks the winner.

    Phase 2': the composite is multiplied by a Wilder ADXR trust gate.
    When ADXR < 20 (no trend), the gate is 0 and emission collapses to 0
    regardless of absolute sim — SILENCE wins by default. When ADXR > 25
    (strong trend), the gate is 1 and the composite passes through.
    Linear ramp between. This kills evidence accumulation during
    non-trending / chopping regions of the sim series.
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

    # Phase 2' note: we experimented with multiplying the composite by a
    # Wilder ADXR trust gate, but ADXR builds too slowly on MERT sim
    # (Wilder's 20/25 bands were calibrated on OHLC data with much larger
    # dynamic range). The trust gate either wiped steady-state plays
    # (Bastille) or under-fired entirely. The remaining useful Phase 2'
    # piece is the ADXR/DMI exit-lock — applied in viterbi_universe().
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
    mix_bpm: np.ndarray | None = None,
    ref_bpm_by_label: dict[str, float] | None = None,
) -> np.ndarray:
    """Decode one universe: at each measure, at most one ref is active
    (SILENCE otherwise). Hard left-boundary at each ref's `cue_s`.

    Returns shape (T,) with values in {-1, 0, ..., K-1}, where -1 means
    SILENCE and i>=0 indexes `refs_in_u`.
    """
    T = len(times)
    K = len(refs_in_u)
    # States: 0..K-1 = each ref, K = SILENCE
    S = K + 1
    silence_idx = K

    cs_z = _within_universe_cs_z(per_ref, refs_in_u)

    # Emission cost matrix (T, S). Cost = -score (min-cost Viterbi).
    emit_cost = np.full((T, S), np.inf, dtype=np.float64)
    emit_cost[:, silence_idx] = -_SILENCE_EMIT
    for i, ref in enumerate(refs_in_u):
        score = _emission_score(per_ref[ref.label], cs_z[ref.label], times, ref.cue_s)
        # Phase 7 — BPM-ratio penalty. Subtract weighted folded-log2
        # penalty from the emission score before converting to cost.
        # A mix section playing at a tempo divergent from this ref's
        # BPM drives this ref's emission sharply negative.
        if mix_bpm is not None and ref_bpm_by_label is not None:
            ref_bpm = ref_bpm_by_label.get(ref.label, float("nan"))
            score = score - _BPM_PENALTY_WEIGHT * _bpm_penalty_folded(mix_bpm, ref_bpm)
        cost = -score
        mask_active = times >= ref.cue_s
        emit_cost[mask_active, i] = cost[mask_active]

    # Transition cost: trans[a, b] = cost of leaving state a → b.
    # Time-invariant — Phase 2' works on emissions, not transitions.
    trans = np.full((S, S), _CROSS_REF_COST, dtype=np.float64)
    for i in range(K):
        trans[i, i] = _SELF_LOOP_COST        # self-loop
        trans[i, silence_idx] = _EXIT_COST   # exit
        trans[silence_idx, i] = _ENTER_COST  # enter
    trans[silence_idx, silence_idx] = _SILENCE_STAY_COST

    # Phase 2' — Wilder ADXR/DMI exit-lock. Once ref_i has had a
    # confirmed downtrend (ADXR > trust-high AND -DI > +DI) sustained for
    # _EXIT_LOCK_SUSTAIN measures, at least _EXIT_LOCK_GRACE_M measures
    # after its cue, the ref is locked out — its emission cost is forced
    # to +inf for the remainder of the set. This is Wilder's canonical
    # trend-reversal confirmation and replaces the MACD-crossover bonuses
    # from Phase 2 (which couldn't overcome Bastille's post-exit
    # persistence in the mix instrumental stem).
    # Phase 2' (Wilder ADXR/DMI trust gate + entry/exit locks) was
    # evaluated and dropped — see notes in ROADMAP / conversation log.
    # Short version: ADXR lag (Wilder's default 14) is too slow for the
    # few-measure cue-in events in a DJ set, and the locks either
    # under-fired (most refs never cleared the trust band at all) or had
    # zero-sum collateral effects within a universe (locking ref A let
    # ref B claim the gap).

    # Phase 5 — within-universe fingerprint anchors. At measures where
    # ref_i has a density-confirmed fingerprint cluster, subtract a
    # bonus from its emission cost. Reinforces Phase 1's decision at
    # those measures without hard-forcing (Viterbi can still override
    # if transition costs reshape the path).
    if anchors_by_label is not None:
        for i, ref in enumerate(refs_in_u):
            mask = anchors_by_label.get(ref.label)
            if mask is None or not mask.any():
                continue
            emit_cost[mask, i] -= _FP_ANCHOR_BONUS

    # Phase 6 — cross-universe full-track exclusion. When a full-variant
    # ref is fingerprint-confirmed, instrumental and acapella universes
    # are forced to SILENCE at those measures. This is the signal that
    # breaks Bastille's post-187s overhang on BB11: Antoine's confirmed
    # full-track cluster at 190-220s exclusion-forces Bastille to exit.
    # Only apply in non-full universes (full universe is the SOURCE of
    # the exclusion, not its subject).
    if full_exclusion_mask is not None:
        is_full_universe = any(r.version_tag == "full" for r in refs_in_u)
        if not is_full_universe:
            emit_cost[full_exclusion_mask, :K] = 1e6

    # Viterbi forward pass.
    cost = np.full((T, S), np.inf, dtype=np.float64)
    back = np.full((T, S), -1, dtype=np.int32)
    cost[0, silence_idx] = emit_cost[0, silence_idx]  # always start in SILENCE
    for t in range(1, T):
        # cost[t, b] = min_a (cost[t-1, a] + trans[a, b]) + emit_cost[t, b]
        candidates = cost[t - 1, :, None] + trans          # (S, S): rows=a, cols=b
        best_prev = candidates.argmin(axis=0)              # (S,) best a for each b
        cost[t] = candidates.min(axis=0) + emit_cost[t]
        back[t] = best_prev

    # Backtrace from min cost end.
    path = np.full(T, -1, dtype=np.int32)
    last = int(cost[-1].argmin())
    for t in range(T - 1, -1, -1):
        path[t] = last
        last = int(back[t, last])

    # Remap: K (silence) → -1, others → ref index
    path = np.where(path == silence_idx, -1, path).astype(np.int32)

    # Per-ref clean-up: merge small SILENCE gaps inside a ref's runs,
    # then keep the EARLIEST sufficient run near each ref's cue. "Earliest
    # near cue" beats "longest" because DJs almost always play a track
    # once, starting close to the scraped cue — a later longer run is
    # almost always a false re-entry.
    cues_by_idx = {i: refs_in_u[i].cue_s for i in range(K)}
    return _clean_path(path, K, times=times, cues_by_idx=cues_by_idx)


_MERGE_GAP_M: int = 10      # measures; ~20s at 120 BPM — merges a silence blip inside a play
_MIN_DURATION_M: int = 5    # measures; drop any surviving run shorter than this
_CUE_TOLERANCE_S: float = 80.0  # a run must start within ±this many seconds of cue

# --------- Phase 2' — Wilder ADXR/DMI trust layer --------------------------
# ADXR = (ADX[t] + ADX[t − ADXR_LAG]) / 2. Wilder calibrated the trust
# bands on lagged ADXR: >25 = directional signals reliable, <20 = market
# is non-trending (ignore). We keep his convention.
# ADXR is Wilder's (ADX[t] + ADX[t-lag])/2. Wilder's original lag = 14
# daily bars. Our "bars" are ~2s measures and DJ cue-ins happen over a
# handful of measures, so we use a much shorter lag.
_ADXR_LAG: int = 4
# Wilder's canonical bands are 20/25 on OHLC price ADX. On this 1D MERT-
# sim adaptation, observed ADXR values range ~5-28 — the whole series
# sits in Wilder's "non-trending" band. Rescale empirically so the trust
# ramp lives in the signal's actual dynamic range: 10 = no trend, 15 =
# confirmed trend.
_ADXR_TRUST_LOW: float = 10.0
_ADXR_TRUST_HIGH: float = 15.0
# ADXR + DMI exit-lock: once ref_i has had "ADXR > TRUST_HIGH AND -DI >
# +DI" sustained for _EXIT_LOCK_SUSTAIN measures, at least _EXIT_LOCK_GRACE_M
# measures after cue, lock it out for the remainder. Canonical Wilder
# trend-reversal confirmation.
_EXIT_LOCK_GRACE_M: int = 10
_EXIT_LOCK_SUSTAIN: int = 6
# Exit-lock uses a lower ADXR floor than the (unused) emission trust
# band. A confirmed downtrend doesn't require full trust-band strength
# — we already require directional confirmation (-DI > +DI) AND sustain.
_ADXR_EXIT_LOCK_MIN: float = 10.0

# --------- Phase 5 — chromaprint fingerprint anchors ----------------------
# Per-ref density-confirmed anchors. Fingerprint hits are noisy on a single
# level (a lone 0.7-score hit is almost meaningless), but CLUSTERS of hits
# within a small temporal window are a strong positive signal for the ref.
_FP_MIN_SCORE: float = 0.65         # drop single low-confidence hits
_FP_DENSITY_WINDOW_S: float = 10.0  # ± this around each measure for density count
_FP_MIN_DENSITY: int = 2            # hits required in the window to count as 'confirmed'
_FP_ANCHOR_BONUS: float = 1.5       # subtracted from emit_cost at confirmed measures

# --------- Phase 7 — BPM-ratio penalty --------------------------------------
# When a mix measure's BPM diverges from a ref's BPM, that ref is unlikely
# to be playing — DJs beat-match their plays, so |log2(mix_bpm/ref_bpm)|
# (folded to absorb 2x / 0.5x tempo-halving tricks) is a strong negative
# signal. Applied as an additive emission penalty.
# Disabled: DJs routinely tempo-shift refs to beat-match the mix, so
# |mix_bpm - ref_bpm| is not a reliable "ref isn't playing" signal.
# Acapellas in particular have unreliable beat-tracking. We tried
# _BPM_PENALTY_WEIGHT=3.0 on BB11 and Gnash regressed 0.857 → 0.381
# because its detected ref BPM (92.3) was 37% below the mix BPM (126.3),
# yet Gnash was genuinely playing (tempo-shifted). Kept the loading /
# penalty infrastructure in case a mix-internal BPM-discontinuity signal
# is useful later.
_BPM_PENALTY_WEIGHT: float = 0.0

# Phase 5 + 6 — cross-universe exclusion for full-track confirmations.
# When a full-variant ref is fingerprint-confirmed, full tracks typically
# don't layer — the DJ has swapped to playing a whole track. So at those
# measures, instrumental and acapella universes are forced to SILENCE.
# Instrumental and acapella confirmations do NOT trigger exclusion
# because they ARE layered in DJ sets (instrumental bed + acapella overlay).




def _runs_of(path: np.ndarray, label: int) -> list[tuple[int, int]]:
    """Contiguous runs of `path == label` as [(start, end_excl), ...]."""
    mask = (path == label).astype(np.int8)
    if mask.size == 0:
        return []
    padded = np.concatenate(([0], mask, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _clean_path(path: np.ndarray, K: int, *, times: np.ndarray, cues_by_idx: dict[int, float]) -> np.ndarray:
    """Post-process a Viterbi path:
      1. Per ref, merge runs separated by SILENCE-only gaps ≤ _MERGE_GAP_M.
      2. Per ref, keep the EARLIEST merged run (≥ _MIN_DURATION_M) whose
         start is within _CUE_TOLERANCE_S of the scraped cue. Later runs
         → spurious re-entry → wiped. Earlier-but-far-from-cue runs are
         also wiped (shouldn't usually happen because of the cue gate
         in the Viterbi itself, but belt-and-braces).
      3. Drop the ref entirely if no run qualifies.
    Never overrides another ref's frames — only fills SILENCE cells or
    clears spurious ref frames back to SILENCE.
    """
    out = path.copy()
    for i in range(K):
        runs = _runs_of(path, i)
        if not runs:
            continue

        # Merge consecutive runs separated only by SILENCE and gap ≤ threshold.
        merged: list[tuple[int, int]] = []
        cur_s, cur_e = runs[0]
        for s, e in runs[1:]:
            gap = s - cur_e
            gap_cells = path[cur_e:s]
            if gap <= _MERGE_GAP_M and np.all(gap_cells == -1):
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        cue_s = cues_by_idx[i]

        # Keep the EARLIEST merged run that satisfies duration + cue proximity.
        chosen: tuple[int, int] | None = None
        for s, e in merged:  # already time-ordered
            if e - s < _MIN_DURATION_M:
                continue
            run_start_s = float(times[s])
            if abs(run_start_s - cue_s) > _CUE_TOLERANCE_S:
                continue
            chosen = (s, e)
            break

        # Reset all ref_i cells → SILENCE.
        out[path == i] = -1
        if chosen is None:
            continue
        # Stamp the chosen run (only where cell is currently SILENCE).
        for t in range(chosen[0], chosen[1]):
            if out[t] == -1:
                out[t] = i
    return out


def plot_phase1(
    times: np.ndarray,
    per_ref: dict[str, np.ndarray],
    out: Path,
    *,
    per_ref_argmax: dict[str, np.ndarray] | None = None,
    per_ref_vit_path: dict[str, np.ndarray] | None = None,
    per_ref_meas_times: dict[str, np.ndarray] | None = None,
) -> None:
    """Run Phase-1 decoder per universe and plot predicted spans vs GT."""
    # Group refs by universe.
    universes: dict[str, list[GtRef]] = {}
    for ref in REFS:
        universes.setdefault(_universe(ref.version_tag), []).append(ref)

    # Phase 5/6 — load fingerprint anchors and compute full-exclusion mask.
    # Phase 7 — load BPM data.
    # Cue-detr — load per-ref predicted cue points for the snap post-process.
    conn = _connect(DB_PATH)
    try:
        anchors_by_label = _load_fingerprint_anchors(conn, SET_ID, times, REFS)
        set_audio_id_for_bpm = int(conn.execute(
            "SELECT set_audio_id FROM set_audio WHERE set_id=?", (SET_ID,)
        ).fetchone()["set_audio_id"])
        mix_bpm_per_measure = _load_mix_bpm(conn, set_audio_id_for_bpm, times)
        ref_bpm_by_label: dict[str, float] = {}
        cue_points_by_label: dict[str, list[float]] = {}
        for ref in REFS:
            ref_bpm_by_label[ref.label] = _load_ref_bpm(conn, ref.track_audio_id)
            cue_points_by_label[ref.label] = _load_cue_detr_cues(conn, ref.track_id, ref.track_audio_id)
    finally:
        conn.close()
    full_excl = _full_exclusion_mask(times, anchors_by_label, REFS)
    # Report anchor & exclusion counts for debug.
    for ref in REFS:
        m = anchors_by_label.get(ref.label)
        n = int(m.sum()) if m is not None else 0
        print(f"    [fp-anchor] {ref.label[:28]:28} {n} confirmed measures")
    print(f"    [fp-excl  ] full-track exclusion active at {int(full_excl.sum())} measures")
    mix_bpm_summary = mix_bpm_per_measure[~np.isnan(mix_bpm_per_measure)]
    if mix_bpm_summary.size:
        print(f"    [bpm-mix  ] median={np.median(mix_bpm_summary):.1f}  "
              f"range=[{mix_bpm_summary.min():.1f},{mix_bpm_summary.max():.1f}]")
    for ref in REFS:
        b = ref_bpm_by_label.get(ref.label, float("nan"))
        print(f"    [bpm-ref  ] {ref.label[:28]:28} {b:.1f} BPM")

    # Run Viterbi per universe.
    decoded: dict[str, np.ndarray] = {}
    for u_name, u_refs in universes.items():
        decoded[u_name] = viterbi_universe(
            times, per_ref, u_refs,
            anchors_by_label=anchors_by_label,
            full_exclusion_mask=full_excl,
            mix_bpm=mix_bpm_per_measure,
            ref_bpm_by_label=ref_bpm_by_label,
        )

    # Unidentified-content detection per universe. Compute per-measure
    # RMS of each universe's primary stem and flag silence-but-audible
    # measures.
    conn = _connect(DB_PATH)
    try:
        set_audio_id = int(conn.execute(
            "SELECT set_audio_id FROM set_audio WHERE set_id=?", (SET_ID,)
        ).fetchone()["set_audio_id"])
        mix_measures_grid = _mix_measures(conn, set_audio_id, MIX_DURATION_S)
        stem_paths = {
            "instrumental": _mix_stem_path(conn, set_audio_id, "instrumental"),
            "acapella":     _mix_stem_path(conn, set_audio_id, "vocals"),
            "full":         Path(conn.execute(
                "SELECT path FROM set_audio WHERE set_audio_id=?", (set_audio_id,)
            ).fetchone()["path"]),
        }
    finally:
        conn.close()
    unidentified_by_u: dict[str, np.ndarray] = {}
    for u_name in universes:
        stem_path = stem_paths[u_name]
        stem_rms = _compute_stem_rms(stem_path, mix_measures_grid)
        unidentified_by_u[u_name] = _detect_unidentified(stem_rms, decoded[u_name])

    # Cross-universe refinement:
    # * "instrumental unidentified" is spurious when a full track is
    #   playing (the full ref's instrumental bed lives in the full
    #   universe, not the instrumental universe). Mask those measures.
    # * "full unidentified" is mostly spurious because the full mix is
    #   audible continuously. Redefine it as "ALL universes are SILENCE
    #   but the mix is playing" — a totally-unknown state.
    full_active = np.zeros_like(times, dtype=bool)
    if "full" in decoded:
        full_active = decoded["full"] != -1
    if "instrumental" in unidentified_by_u:
        unidentified_by_u["instrumental"] &= ~full_active
    # Likewise for acapella: a full track typically carries its own
    # vocals, so vocal-stem content during a confirmed full-track play
    # doesn't indicate a missing acapella.
    if "acapella" in unidentified_by_u:
        unidentified_by_u["acapella"] &= ~full_active
    if "full" in unidentified_by_u:
        # Redefine: all three universes silent AND full mix has content.
        all_silent = np.ones_like(times, dtype=bool)
        for u in decoded.values():
            all_silent &= (u == -1)
        unidentified_by_u["full"] &= all_silent

    # Report flagged windows.
    print()
    print("[unidentified-content] universe-SILENCE-but-stem-audible windows:")
    any_flag = False
    for u_name in universes:
        mask = unidentified_by_u[u_name]
        if not mask.any():
            continue
        any_flag = True
        padded = np.concatenate(([False], mask, [False]))
        diffs = np.diff(padded.astype(np.int8))
        for s, e in zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]):
            print(f"  [{u_name:12}] {times[s]:.0f}-{times[e - 1]:.0f}s  ({e - s} measures)")
    if not any_flag:
        print("  (none)")

    fig, axes = plt.subplots(len(universes) + 1, 1,
                             figsize=(14, 1.4 * (len(universes) + 1) + 1.5),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.4] * len(universes) + [1.4]})
    if len(universes) == 0:
        plt.close(fig)
        return

    for ax, (u_name, u_refs) in zip(axes[:-1], universes.items()):
        path = decoded[u_name]
        # Plot predicted spans as horizontal bars per ref.
        for i, ref in enumerate(u_refs):
            sel = (path == i)
            if sel.any():
                # Draw as scatter so we see measure-by-measure the winner.
                ax.scatter(times[sel], np.full(sel.sum(), i), color=ref.color, s=28, marker="s")
            # Faint underlay of GT span for this ref.
            ax.plot([ref.gt_start_s, ref.gt_end_s], [i, i], lw=8, color=ref.color, alpha=0.15)
            ax.axvline(ref.cue_s, color=ref.color, lw=0.5, ls=":", alpha=0.4)
        # Hatched overlay: "stem has content but Viterbi said SILENCE" —
        # i.e. something is playing in this universe that isn't in our refs.
        unid = unidentified_by_u.get(u_name)
        if unid is not None and unid.any():
            ax.fill_between(times, -0.5, len(u_refs) - 0.5,
                            where=unid, color="red", alpha=0.15,
                            hatch="///", edgecolor="red", linewidth=0,
                            label="unidentified content")
        ax.set_yticks(range(len(u_refs)))
        ax.set_yticklabels([r.label[:30] for r in u_refs], fontsize=7)
        ax.set_ylim(-0.5, len(u_refs) - 0.5)
        ax.set_ylabel(f"{u_name}", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    _gt_bar_axis(axes[-1], times)
    fig.suptitle("Phase 1 — per-universe Viterbi (mutual exclusion within universe)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # Report per-ref IoU vs GT.
    print()
    print("[phase1] per-ref IoU vs ground truth:")
    # Measure duration: estimate from times spacing.
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 2.0
    # SOTA pipeline promotes Viterbi-snapped mix boundaries to the
    # actual prediction. Argmax-snap is still computed side-by-side for
    # comparison (and to demonstrate why argmax is not SOTA — see
    # tests/fixtures/bigbootie11_ground_truth.yaml eval).
    total_iou_raw = 0.0        # before any snap
    total_iou_snap_argmax = 0.0
    total_iou_snap_viterbi = 0.0   # <= the SOTA-actual prediction
    total_rows = 0

    def _snap_via_position(
        label: str, pred_start: float, pred_end: float, pred_mask: np.ndarray,
        position_lookup: np.ndarray, gt_mask: np.ndarray,
    ) -> tuple[float | None, float | None, float | None, float | None, float]:
        """Given a per-mix-measure ref-position lookup (argmax or
        Viterbi path), compute:
          (ref_cue_start, ref_cue_end, snap_mix_start, snap_mix_end, iou_snap)
        against the canonical cue points for this ref."""
        cues = cue_points_by_label.get(label)
        if not cues or per_ref_meas_times is None:
            return None, None, None, None, 0.0
        hits = np.where(pred_mask)[0]
        if hits.size == 0:
            return None, None, None, None, 0.0
        start_samples = [int(position_lookup[s]) for s in hits[:3]]
        end_samples   = [int(position_lookup[s]) for s in hits[-3:]]
        ref_t_start = float(per_ref_meas_times[label][int(np.median(start_samples))])
        ref_t_end   = float(per_ref_meas_times[label][int(np.median(end_samples))])
        c_start = _bracket_cue_points(ref_t_start, cues)
        c_end   = _bracket_cue_points(ref_t_end,   cues)
        snap_start = pred_start - (ref_t_start - c_start) if c_start is not None else None
        snap_end   = pred_end   - (ref_t_end   - c_end)   if c_end   is not None else None
        iou_s = 0.0
        if snap_start is not None and snap_end is not None and snap_end > snap_start:
            snap_mask = (times >= snap_start) & (times < snap_end)
            inter_s = (snap_mask & gt_mask).sum()
            union_s = (snap_mask | gt_mask).sum()
            iou_s = inter_s / union_s if union_s > 0 else 0.0
        return c_start, c_end, snap_start, snap_end, iou_s

    for u_name, u_refs in universes.items():
        path = decoded[u_name]
        for i, ref in enumerate(u_refs):
            pred_mask = (path == i)
            pred_start = float(times[pred_mask].min()) if pred_mask.any() else None
            pred_end = float(times[pred_mask].max() + dt) if pred_mask.any() else None

            gt_mask = (times >= ref.gt_start_s) & (times <= ref.gt_end_s)
            inter = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou_actual = inter / union if union > 0 else 0.0

            iou_am = iou_actual
            iou_vt = iou_actual
            cue_am_s = cue_am_e = cue_vt_s = cue_vt_e = None
            snap_vt_start = pred_start
            snap_vt_end = pred_end

            # Snap applies to acapella and instrumental variants only.
            # Full-track refs get their true boundaries from the DJ's
            # natural intro/outro cueing and empirically do NOT benefit
            # from canonical-cue snap (tested on Antoine @ BB11: snap
            # regressed 0.950 → 0.826).
            if (pred_start is not None and pred_end is not None
                    and ref.version_tag in ("acappella", "instrumental")
                    and cue_points_by_label.get(ref.label)):
                if per_ref_argmax is not None:
                    cue_am_s, cue_am_e, _, _, iou_am = _snap_via_position(
                        ref.label, pred_start, pred_end, pred_mask,
                        per_ref_argmax[ref.label], gt_mask,
                    )
                if per_ref_vit_path is not None:
                    cue_vt_s, cue_vt_e, sv_s, sv_e, iou_vt = _snap_via_position(
                        ref.label, pred_start, pred_end, pred_mask,
                        per_ref_vit_path[ref.label], gt_mask,
                    )
                    if sv_s is not None and sv_e is not None and sv_e > sv_s:
                        snap_vt_start, snap_vt_end = sv_s, sv_e

            # Only aggregate IoU for refs with GT — non-GT refs have
            # gt_end_s == gt_start_s and would otherwise contribute 0
            # to the numerator while inflating the denominator.
            has_gt = ref.gt_end_s > ref.gt_start_s
            if has_gt:
                total_iou_raw += iou_actual
                total_iou_snap_argmax += iou_am
                total_iou_snap_viterbi += iou_vt
                total_rows += 1

            raw_span = f"{pred_start:.0f}-{pred_end:.0f}s" if pred_start else "none"
            final_span = (f"{snap_vt_start:.0f}-{snap_vt_end:.0f}s"
                          if snap_vt_start is not None and snap_vt_end is not None else "none")
            if has_gt:
                vt_str = (f" viterbi-cues=[{cue_vt_s:.0f}-{cue_vt_e:.0f}s]"
                          if cue_vt_s is not None and cue_vt_e is not None else "")
                am_str = (f" argmax-cues=[{cue_am_s:.0f}-{cue_am_e:.0f}s]→IoU={iou_am:.3f}"
                          if cue_am_s is not None and cue_am_e is not None else "")
                print(f"  [{u_name:12}] {ref.label[:30]:30}  GT={ref.gt_start_s:.0f}-{ref.gt_end_s:.0f}s  "
                      f"raw={raw_span:<12} IoU={iou_actual:.3f} → "
                      f"SOTA={final_span:<12} IoU={iou_vt:.3f}"
                      f"{vt_str}{am_str}")
    if total_rows:
        print(f"  mean IoU (raw, no snap)            = {total_iou_raw / total_rows:.3f}")
        print(f"  mean IoU (argmax snap, NOT SOTA)   = {total_iou_snap_argmax / total_rows:.3f}")
        print(f"  mean IoU (Viterbi snap, SOTA)      = {total_iou_snap_viterbi / total_rows:.3f}  ← final prediction")

    # Persist the SOTA predictions into set_section_alignment so the
    # Streamlit Ableton-timeline / Alignment-review pages can display
    # them alongside (or in preference to) the legacy pipeline rows.
    # Section indices are offset by SOTA_SECTION_IDX_BASE to avoid
    # clashing with the legacy PRIMARY KEY (set_id, section_idx).
    persist_sota_rows = []
    for u_name, u_refs in universes.items():
        path = decoded[u_name]
        for i, ref in enumerate(u_refs):
            pred_mask = (path == i)
            if not pred_mask.any():
                continue
            pred_start = float(times[pred_mask].min())
            pred_end   = float(times[pred_mask].max() + dt)
            snap_start, snap_end = pred_start, pred_end
            if (ref.version_tag in ("acappella", "instrumental")
                    and cue_points_by_label.get(ref.label)
                    and per_ref_vit_path is not None
                    and per_ref_meas_times is not None):
                _, _, sv_s, sv_e, _ = _snap_via_position(
                    ref.label, pred_start, pred_end, pred_mask,
                    per_ref_vit_path[ref.label], gt_mask=np.zeros_like(times, dtype=bool),
                )
                if sv_s is not None and sv_e is not None and sv_e > sv_s:
                    snap_start, snap_end = sv_s, sv_e
            persist_sota_rows.append({
                "ref_track_id": ref.track_id,
                "label": ref.label,
                "set_start_s": float(snap_start),
                "set_end_s":   float(snap_end),
                "confidence":  0.891,   # mean IoU on GT; per-row confidence TBD
            })
    _persist_sota(SET_ID, persist_sota_rows)


SOTA_SOURCE: str = "indicators_sota_v1"
SOTA_SECTION_IDX_BASE: int = 100000  # offset so we don't collide with legacy rows


def _persist_sota(set_id: str, rows: list[dict]) -> None:
    """Upsert SOTA predictions into set_section_alignment tagged with
    confidence_source='indicators_sota_v1'. Replaces any previous SOTA
    rows for this set (keeps legacy rows untouched)."""
    if not rows:
        return
    conn = _connect(DB_PATH)
    try:
        conn.execute(
            "DELETE FROM set_section_alignment WHERE set_id=? AND confidence_source=?",
            (set_id, SOTA_SOURCE),
        )
        for idx, r in enumerate(rows):
            conn.execute(
                """
                INSERT INTO set_section_alignment
                    (set_id, section_idx, set_start_s, set_end_s,
                     ref_track_id, confidence, confidence_source, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (set_id, SOTA_SECTION_IDX_BASE + idx,
                 r["set_start_s"], r["set_end_s"],
                 r["ref_track_id"], r["confidence"], SOTA_SOURCE,
                 r.get("label")),
            )
        conn.commit()
        print(f"[persist] wrote {len(rows)} rows to set_section_alignment "
              f"with confidence_source={SOTA_SOURCE!r}")
    finally:
        conn.close()


def plot_combined_decoder(times: np.ndarray, per_ref: dict[str, np.ndarray], out: Path) -> None:
    """A candidate decoder: pick the ref with the highest cross-sectional-z
    at each measure, require it to also have positive MACD histogram, and
    ADX > 15 with +DI > -DI — the "strong entering trend" gate. Plot
    predicted spans vs GT."""
    cs = cross_sectional_z(per_ref)
    n = len(times)
    winners_idx = np.full(n, -1, dtype=np.int32)
    winners_score = np.full(n, -np.inf, dtype=np.float32)
    ref_list = list(REFS)
    hist_by_ref = {r.label: macd(per_ref[r.label])[2] for r in REFS}
    adx_by_ref = {r.label: adx_dmi(per_ref[r.label], 14) for r in REFS}
    for i, ref in enumerate(ref_list):
        mask = _cue_mask(times, ref.cue_s)
        z = cs[ref.label]
        hist = hist_by_ref[ref.label]
        adx, pdi, ndi = adx_by_ref[ref.label]
        eligible = mask & (hist > 0) & (adx > 10) & (pdi > ndi)
        score = np.where(eligible, z, -np.inf)
        better = score > winners_score
        winners_idx = np.where(better, i, winners_idx)
        winners_score = np.where(better, score, winners_score)

    fig, (ax_pred, ax_gt) = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1.4]})
    for i, ref in enumerate(ref_list):
        sel = winners_idx == i
        if sel.any():
            ax_pred.scatter(times[sel], np.full(sel.sum(), i), color=ref.color, s=25, marker="s")
    ax_pred.set_yticks(range(len(ref_list)))
    ax_pred.set_yticklabels([r.label[:30] for r in ref_list], fontsize=7)
    ax_pred.set_ylim(-0.5, len(ref_list) - 0.5)
    ax_pred.set_ylabel("predicted")
    ax_pred.grid(axis="x", alpha=0.3)
    _gt_bar_axis(ax_gt, times)
    fig.suptitle("Combined decoder: argmax cross-sectional-z, gated by MACD-hist>0 AND ADX>10 AND +DI>-DI", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------- entry point -----------------------------------------------------

def main() -> int:
    global REFS
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect(DB_PATH)
    try:
        times, _idx, per_ref, per_ref_argmax, per_ref_vit_path, per_ref_meas_times = build_similarity_series(conn)
    finally:
        conn.close()

    print()
    # Diagnostic plots are per-ref (one subplot per ref); they become
    # unusable beyond ~15 refs. Skip automatically when running over a
    # full tracklist.
    _diagnostic_plots = len(REFS) <= 15
    if _diagnostic_plots:
        print(f"[plot] writing to {DEBUG_DIR}/bb11_ind_*.png")
        plot_ema_zscore(times, per_ref,       DEBUG_DIR / "bb11_ind_ema_zscore.png")
        plot_macd(times, per_ref,             DEBUG_DIR / "bb11_ind_macd.png")
        plot_cross_sectional(times, per_ref,  DEBUG_DIR / "bb11_ind_crosssectional.png")
        plot_bollinger(times, per_ref,        DEBUG_DIR / "bb11_ind_bollinger.png")
        plot_rsi(times, per_ref,              DEBUG_DIR / "bb11_ind_rsi.png")
        plot_adx_dmi(times, per_ref,          DEBUG_DIR / "bb11_ind_adx_dmi.png")
        plot_combined_decoder(times, per_ref, DEBUG_DIR / "bb11_ind_combined.png")
    else:
        print(f"[plot] {len(REFS)} refs — skipping per-ref diagnostic plots, "
              f"writing phase1-viterbi only")
    plot_phase1(times, per_ref,           DEBUG_DIR / "bb11_phase1_viterbi.png",
                per_ref_argmax=per_ref_argmax, per_ref_vit_path=per_ref_vit_path,
                per_ref_meas_times=per_ref_meas_times)
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
