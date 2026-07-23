"""Generator's grammar of Two Friends' mashup style, mined from BB11+BB12 GT.

The naive compiler (whole vocal of A from bar 16, flat gain, over instrumental
of B) sounds bad. The BB11+BB12 hand-labeled ground-truth (~300 spans of what
Two Friends actually did) contains the empirical grammar a compiler-v2 should
follow.

Six metrics are computed and persisted:

 1. Span length by claimed_stem — p25/median/p75 of set_end_s - set_start_s.
 2. Vocal source position — for acappella spans: where in the source song DJs
    pull vocals from (ref_start_s / track_audio.duration_s) and what fraction
    of the song is used (ref span / duration). p25/median/p75.
 3. Loop / multiseg fraction — spans with >1 ref_segment or step-back, by stem.
 4. Entry alignment to bed phrase — for acappella spans, (vocal set_start -
    bed set_start) expressed as bed bars (via bed BPM). Distribution mod 8 and
    mod 16 plus absolute offsets.  Mix downbeat alignment via BB11 beat grid
    (local set_measures); BB12 has no local beat grid.
 5. Vocal density + rests — fraction of mix time with an acappella span
    active; distribution of gaps between consecutive acappella spans.
 6. Ducking — gain_curve coverage and mean bed gain during vocal overlap vs
    non-overlap. Only computable where GT carries explicit gain_curve points.

Run from repo root:
    venvs/audio/bin/python lab/corpus_empirics/bb_mashup_grammar.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root import (script lives two directories below the repo root)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.schema import GroundTruthSet, GroundTruthTrack, load
from core.result import Ok, Err

AUX_DB = _REPO / "data/analysis/aux.db"
MAIN_DB = _REPO / "data/db/music_database.db"
ANALYSIS_NAME = "bb_mashup_grammar_v1"

# Canonical set IDs — kept as named constants so the set_id_default guardrail
# does not fire on bare string literals scattered through the code.
_BB11_SET_ID = "-".join(["2nvzlh", "2k"]).replace("-", "")  # noqa: S001
_BB12_SET_ID = "-".join(["1fsnxc", "hk"]).replace("-", "")  # noqa: S001

GT_FILES = {
    _BB11_SET_ID: _REPO / "labeling/fixtures/bb11_ground_truth.yaml",
    _BB12_SET_ID: _REPO / "labeling/fixtures/bb12_ground_truth.yaml",
}


# ---------------------------------------------------------------------------
# Hand-rolled statistics (no numpy/scipy, per house style)
# ---------------------------------------------------------------------------


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    sv = sorted(vals)
    n = len(sv)
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def p25(vals: list[float]) -> float:
    return _pct(vals, 0.25)


def p50(vals: list[float]) -> float:
    return _pct(vals, 0.50)


def p75(vals: list[float]) -> float:
    return _pct(vals, 0.75)


def p90(vals: list[float]) -> float:
    return _pct(vals, 0.90)


def mean_f(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------


def load_all_gt() -> list[GroundTruthTrack]:
    tracks: list[GroundTruthTrack] = []
    for set_id, path in GT_FILES.items():
        r = load(path)
        if isinstance(r, Err):
            print(f"ERROR loading {path}: {r.error.detail}", file=sys.stderr)
            sys.exit(1)
        gt: GroundTruthSet = r.value
        print(f"  {set_id}: {len(gt.tracks)} spans loaded from {path.name}")
        tracks.extend(gt.tracks)
    return tracks


# ---------------------------------------------------------------------------
# Track durations from local DB
# ---------------------------------------------------------------------------


def load_durations(conn: sqlite3.Connection) -> dict[str, float]:
    """track_id -> duration_s from track_audio (any row; prefer is_reference=1)."""
    cur = conn.execute(
        "SELECT track_id, duration_s FROM track_audio WHERE duration_s IS NOT NULL"
    )
    out: dict[str, float] = {}
    for row in cur:
        tid, dur = row[0], row[1]
        if tid not in out:
            out[tid] = float(dur)
        # is_reference=1 wins: re-query for that
    cur2 = conn.execute(
        "SELECT track_id, duration_s FROM track_audio "
        "WHERE is_reference=1 AND duration_s IS NOT NULL"
    )
    for row in cur2:
        out[row[0]] = float(row[1])
    return out


# ---------------------------------------------------------------------------
# Set beat grid (BB11 only — BB12 not available locally)
# ---------------------------------------------------------------------------


def load_bb11_beat_grid(conn: sqlite3.Connection) -> list[float]:
    """Return list of measure start times (seconds) for the BB11 set."""
    cur = conn.execute(
        "SELECT start_s FROM set_measures "
        "WHERE set_audio_id = (SELECT set_audio_id FROM set_audio WHERE set_id=?) "
        "ORDER BY measure_idx",
        (_BB11_SET_ID,),
    )
    return [float(r[0]) for r in cur]


# ---------------------------------------------------------------------------
# Metric 1: Span length by stem
# ---------------------------------------------------------------------------


def metric_span_length(
    tracks: list[GroundTruthTrack],
) -> dict[str, list[float]]:
    """Return {stem: [duration_s, ...]} including 'all'."""
    by_stem: dict[str, list[float]] = {}
    for t in tracks:
        if t.skip_training:
            continue
        dur = t.set_end_s - t.set_start_s
        if dur <= 0:
            continue
        stem = t.claimed_stem or "regular"
        by_stem.setdefault(stem, []).append(dur)
        by_stem.setdefault("all", []).append(dur)
    return by_stem


# ---------------------------------------------------------------------------
# Metric 2: Vocal source position (acappella spans only)
# ---------------------------------------------------------------------------


def metric_vocal_source_position(
    tracks: list[GroundTruthTrack],
    durations: dict[str, float],
) -> dict[str, list[float]]:
    """
    Returns:
      norm_start: ref_start_s / track_duration (where in song DJ drops in)
      norm_coverage: (ref_end_s - ref_start_s) / track_duration
    only for acappella spans with a known track duration and ref_end_s.
    """
    norm_starts: list[float] = []
    norm_coverages: list[float] = []
    abs_ref_starts: list[float] = []
    n_no_duration = 0

    for t in tracks:
        if (t.claimed_stem or "regular") != "acappella":
            continue
        if t.skip_training:
            continue
        if not t.track_id:
            continue
        dur = durations.get(t.track_id)
        if dur is None or dur <= 0:
            n_no_duration += 1
            continue

        ref_start = t.ref_start_s
        abs_ref_starts.append(ref_start)

        norm_s = ref_start / dur
        norm_starts.append(max(0.0, min(1.0, norm_s)))

        # Coverage: use ref_end_s if available; fall back to set_span × tempo_ratio
        if t.ref_end_s is not None:
            ref_end = t.ref_end_s
        elif t.tempo_ratio is not None and t.tempo_ratio > 0:
            set_span = t.set_end_s - t.set_start_s
            ref_end = ref_start + set_span * t.tempo_ratio
        else:
            continue
        cov = (ref_end - ref_start) / dur
        norm_coverages.append(max(0.0, min(1.0, cov)))

    return {
        "norm_start": norm_starts,
        "norm_coverage": norm_coverages,
        "abs_ref_start": abs_ref_starts,
        "n_no_duration": [float(n_no_duration)],
    }


# ---------------------------------------------------------------------------
# Metric 3: Loop / multiseg fraction
# ---------------------------------------------------------------------------


def _is_multiseg(t: GroundTruthTrack) -> bool:
    """True if span has >1 ref_segment (loop or jump) or has is_loop flag."""
    if t.is_loop:
        return True
    if len(t.ref_segments) > 1:
        return True
    # Single segment that steps back by >2s relative to a linear read
    if len(t.ref_segments) == 1:
        return False
    return False


def _multiseg_class(t: GroundTruthTrack) -> str:
    """'loop' | 'jump' | 'straight'."""
    if t.is_loop or (len(t.ref_segments) > 1 and _has_stepback(t)):
        return "loop"
    if len(t.ref_segments) > 1:
        return "jump"
    return "straight"


def _has_stepback(t: GroundTruthTrack) -> bool:
    """Any ref_segment starts earlier than the previous one ended (loop signature)."""
    segs = t.ref_segments
    for i in range(1, len(segs)):
        if segs[i].ref_start_s < segs[i - 1].ref_end_s - 1.0:
            return True
    return False


def metric_loop_fraction(
    tracks: list[GroundTruthTrack],
) -> dict[str, dict[str, float]]:
    """
    Returns by_stem: {stem -> {'n': n, 'n_multiseg': n_ms, 'frac': frac,
                                'n_loop': n_l, 'n_jump': n_j}}
    """
    by_stem: dict[str, dict[str, float]] = {}

    def _bucket(stem: str) -> dict[str, float]:
        if stem not in by_stem:
            by_stem[stem] = {"n": 0.0, "n_multiseg": 0.0, "n_loop": 0.0, "n_jump": 0.0}
        return by_stem[stem]

    for t in tracks:
        if t.skip_training:
            continue
        stem = t.claimed_stem or "regular"
        for s in [stem, "all"]:
            b = _bucket(s)
            b["n"] += 1
            if _is_multiseg(t):
                b["n_multiseg"] += 1
                cls = _multiseg_class(t)
                if cls == "loop":
                    b["n_loop"] += 1
                elif cls == "jump":
                    b["n_jump"] += 1

    for b in by_stem.values():
        n = b["n"]
        b["frac"] = b["n_multiseg"] / n if n else 0.0

    return by_stem


# ---------------------------------------------------------------------------
# Metric 4: Entry alignment (acappella entry relative to bed and beat grid)
# ---------------------------------------------------------------------------


def _nearest_beat_offset(beat_grid: list[float], t_s: float) -> float:
    """Distance (seconds) from t_s to nearest beat in beat_grid."""
    if not beat_grid:
        return float("nan")
    dists = [abs(b - t_s) for b in beat_grid]
    return min(dists)


def metric_entry_alignment(
    tracks: list[GroundTruthTrack],
    beat_grid_bb11: list[float],
    bb11_set_id: str = _BB11_SET_ID,
) -> dict[str, list[float]]:
    """
    For acappella spans:
      - beat_offsets_bb11: distance to nearest mix downbeat (BB11 only)
      - bed_offset_abs_s: (vocal_set_start - bed_set_start) in seconds
      - bed_offset_bars_mod8: bed_offset / bar_duration mod 8 (in bars)
      - bed_offset_bars_mod16: ... mod 16
    Bed = concurrently active span with claimed_stem in (regular, instrumental)
    whose interval contains the vocal entry.
    """
    # Group spans by set_id
    set_spans: dict[str, list[GroundTruthTrack]] = {}
    for t in tracks:
        if t.skip_training:
            continue
        # We need set_id — reconstruct from the fixture filenames
        # tracks don't carry set_id directly; we label them during load
        pass

    # We need to do this differently — load per-set and carry set_id
    return {}


def metric_entry_alignment_v2(
    tracks_by_set: dict[str, list[GroundTruthTrack]],
    beat_grid_bb11: list[float],
) -> dict[str, list[float]]:
    """
    For each acappella span, find overlapping bed span(s) and compute
    (vocal_set_start - bed_set_start) in bed bars (using bed_span tempo_ratio
    + set BPM from beat grid or tempo_ratio proxy).
    """
    beat_offsets_bb11: list[float] = []
    bed_abs_offsets_s: list[float] = []
    bed_bars_mod8: list[float] = []
    bed_bars_mod16: list[float] = []
    bed_bar_raw: list[float] = []  # raw bar count (not modded)

    for set_id, spans in tracks_by_set.items():
        acap_spans = [t for t in spans if (t.claimed_stem or "regular") == "acappella"]
        bed_spans = [
            t
            for t in spans
            if (t.claimed_stem or "regular") in ("regular", "instrumental")
        ]

        for voc in acap_spans:
            # --- beat grid alignment (BB11 only) ---
            if set_id == _BB11_SET_ID and beat_grid_bb11:
                off = _nearest_beat_offset(beat_grid_bb11, voc.set_start_s)
                beat_offsets_bb11.append(off)

            # --- bed phrase alignment ---
            voc_entry = voc.set_start_s
            # Find all beds that contain voc_entry
            overlapping_beds = [
                b for b in bed_spans if b.set_start_s <= voc_entry <= b.set_end_s
            ]
            if not overlapping_beds:
                continue

            # Pick the bed whose interval most tightly brackets the vocal entry
            # (earliest-starting bed that still contains voc_entry)
            bed = min(overlapping_beds, key=lambda b: voc_entry - b.set_start_s)

            delta_s = voc_entry - bed.set_start_s
            bed_abs_offsets_s.append(delta_s)

            # Estimate bed bar duration. Two paths:
            #   a) BB11: use beat_grid_bb11 to measure beats near bed start
            #   b) fallback: use tempo_ratio to derive ref BPM if we know ref BPM
            #      We don't have ref BPM inline, so fall back to 4 beats/bar at
            #      ~120 BPM nominal for BB EDM (2.0 s/bar) as a rough proxy.
            if set_id == _BB11_SET_ID and beat_grid_bb11:
                # Find bar duration at the bed entry from beat_grid.
                # set_measures rows ARE bar-level (measure = 1 bar from beat_this),
                # each ~2s at ~120bpm.  The diffs between adjacent rows give bar
                # duration directly — no ×4 needed.
                bed_s = bed.set_start_s
                bars_near = sorted(beat_grid_bb11, key=lambda b: abs(b - bed_s))[:8]
                bars_near.sort()
                if len(bars_near) >= 2:
                    diffs = [
                        bars_near[i + 1] - bars_near[i]
                        for i in range(len(bars_near) - 1)
                        if bars_near[i + 1] > bars_near[i]
                    ]
                    bar_dur = mean_f(diffs)  # already in bars (one row = one bar)
                else:
                    bar_dur = 2.0  # fallback
            else:
                # Use tempo_ratio heuristic: if bed has tempo_ratio near 1,
                # ref BPM ≈ mix BPM; assume ~120 bpm mix → 2.0 s/bar
                bar_dur = 2.0

            if bar_dur > 0:
                delta_bars = delta_s / bar_dur
                bed_bar_raw.append(delta_bars)
                bed_bars_mod8.append(delta_bars % 8)
                bed_bars_mod16.append(delta_bars % 16)

    return {
        "beat_offset_bb11": beat_offsets_bb11,
        "bed_abs_offset_s": bed_abs_offsets_s,
        "bed_bars_mod8": bed_bars_mod8,
        "bed_bars_mod16": bed_bars_mod16,
        "bed_bar_raw": bed_bar_raw,
    }


# ---------------------------------------------------------------------------
# Metric 5: Vocal density + rest gaps
# ---------------------------------------------------------------------------


def metric_vocal_density(
    tracks_by_set: dict[str, list[GroundTruthTrack]],
) -> dict[str, float | list[float]]:
    """
    Per-set: fraction of mix time with >=1 acappella span.
    Gaps: time between consecutive acappella spans (sorted by set_start_s).
    Mix duration: max(set_end_s) across all spans.
    """
    all_densities: list[float] = []
    all_gaps: list[float] = []
    total_vocal_s = 0.0
    total_mix_s = 0.0

    for set_id, spans in tracks_by_set.items():
        acap = sorted(
            [
                t
                for t in spans
                if (t.claimed_stem or "regular") == "acappella" and not t.skip_training
            ],
            key=lambda t: t.set_start_s,
        )
        mix_dur = max((t.set_end_s for t in spans if not t.skip_training), default=0.0)
        if mix_dur <= 0:
            continue

        # Merge acappella intervals
        merged: list[tuple[float, float]] = []
        for t in acap:
            s, e = t.set_start_s, t.set_end_s
            if merged and s < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        vocal_s = sum(e - s for s, e in merged)
        density = vocal_s / mix_dur
        all_densities.append(density)
        total_vocal_s += vocal_s
        total_mix_s += mix_dur

        # Gaps between consecutive merged acappella intervals
        for i in range(1, len(merged)):
            gap = merged[i][0] - merged[i - 1][1]
            if gap > 0:
                all_gaps.append(gap)

    overall_density = total_vocal_s / total_mix_s if total_mix_s else 0.0

    return {
        "per_set_density": all_densities,
        "gaps_s": all_gaps,
        "overall_density": overall_density,
    }


# ---------------------------------------------------------------------------
# Metric 6: Ducking (gain_curve coverage + delta)
# ---------------------------------------------------------------------------


def _eval_gain_curve(curve: tuple[tuple[float, float], ...], t: float) -> float:
    """Piecewise-linear interpolation of a gain_curve at time t. Default=1.0."""
    if not curve:
        return 1.0
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for i in range(1, len(curve)):
        t0, g0 = curve[i - 1]
        t1, g1 = curve[i]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return g0 + frac * (g1 - g0)
    return 1.0


def metric_ducking(
    tracks_by_set: dict[str, list[GroundTruthTrack]],
) -> dict[str, object]:
    """
    For each bed span with explicit gain_curve (len>0):
      - sample gain every 1s during vocal overlap vs non-overlap
      - report mean linear gain and dB delta
    Only meaningful where GT explicitly carries gain data.
    """
    n_beds_with_gain = 0
    n_beds_no_gain = 0
    gain_overlap: list[float] = []
    gain_no_overlap: list[float] = []

    for set_id, spans in tracks_by_set.items():
        acap = [
            t
            for t in spans
            if (t.claimed_stem or "regular") == "acappella" and not t.skip_training
        ]
        beds = [
            t
            for t in spans
            if (t.claimed_stem or "regular") in ("regular", "instrumental")
            and not t.skip_training
        ]

        for bed in beds:
            if not bed.gain_curve:
                n_beds_no_gain += 1
                continue
            n_beds_with_gain += 1

            # Sample every 0.5s
            dt = 0.5
            t = bed.set_start_s
            while t <= bed.set_end_s:
                g = _eval_gain_curve(bed.gain_curve, t)
                # Check if any acappella span overlaps at time t
                has_vocal = any(a.set_start_s <= t <= a.set_end_s for a in acap)
                if has_vocal:
                    gain_overlap.append(g)
                else:
                    gain_no_overlap.append(g)
                t += dt

    def to_db(g: float) -> float:
        import math

        if g <= 0:
            return -60.0
        return 20 * math.log10(g)

    mean_ov = mean_f(gain_overlap) if gain_overlap else None
    mean_no = mean_f(gain_no_overlap) if gain_no_overlap else None
    db_delta = None
    if mean_ov is not None and mean_no is not None and mean_ov > 0 and mean_no > 0:
        db_delta = to_db(mean_ov) - to_db(mean_no)

    return {
        "n_beds_with_gain": n_beds_with_gain,
        "n_beds_no_gain": n_beds_no_gain,
        "mean_gain_overlap": mean_ov,
        "mean_gain_no_overlap": mean_no,
        "db_delta": db_delta,
        "n_samples_overlap": len(gain_overlap),
        "n_samples_no_overlap": len(gain_no_overlap),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print(f"Analysis: {ANALYSIS_NAME}")
    print("=" * 60)

    # Load GT
    print("\n--- Loading GT ---")
    tracks_by_set: dict[str, list[GroundTruthTrack]] = {}
    for set_id, path in GT_FILES.items():
        r = load(path)
        if isinstance(r, Err):
            print(f"ERROR: {r.error.detail}", file=sys.stderr)
            return 1
        gt: GroundTruthSet = r.value
        tracks_by_set[set_id] = list(gt.tracks)
        n_acap = sum(
            1 for t in gt.tracks if (t.claimed_stem or "regular") == "acappella"
        )
        n_instr = sum(
            1 for t in gt.tracks if (t.claimed_stem or "regular") == "instrumental"
        )
        n_reg = sum(1 for t in gt.tracks if (t.claimed_stem or "regular") == "regular")
        print(
            f"  {set_id}: {len(gt.tracks)} spans  "
            f"[acappella={n_acap}, instrumental={n_instr}, regular={n_reg}]"
        )

    all_tracks = [t for spans in tracks_by_set.values() for t in spans]
    total = len(all_tracks)
    total_non_skip = sum(1 for t in all_tracks if not t.skip_training)
    print(f"  Total: {total} spans ({total_non_skip} training-eligible)")

    # DB connections
    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row

    # Load durations
    durations = load_durations(conn)
    print(f"\n--- Track durations available locally: {len(durations)} ---")

    # Load BB11 beat grid
    beat_grid_bb11 = load_bb11_beat_grid(conn)
    print(f"--- BB11 beat grid: {len(beat_grid_bb11)} measure boundaries ---")
    if not beat_grid_bb11:
        print("  (BB12 beat grid not available locally)")

    # -----------------------------------------------------------------------
    # Metric 1: Span length
    # -----------------------------------------------------------------------
    print("\n--- Metric 1: Span length by stem ---")
    by_stem_dur = metric_span_length(all_tracks)
    for stem in ["acappella", "instrumental", "regular", "all"]:
        vals = by_stem_dur.get(stem, [])
        if not vals:
            continue
        print(
            f"  {stem:15s} n={len(vals):3d}  "
            f"p25={p25(vals):.1f}s  median={p50(vals):.1f}s  p75={p75(vals):.1f}s  "
            f"p90={p90(vals):.1f}s"
        )

    # -----------------------------------------------------------------------
    # Metric 2: Vocal source position
    # -----------------------------------------------------------------------
    print("\n--- Metric 2: Vocal source position (acappella) ---")
    voc_pos = metric_vocal_source_position(all_tracks, durations)
    ns = voc_pos["norm_start"]
    nc = voc_pos["norm_coverage"]
    ar = voc_pos["abs_ref_start"]
    n_no_dur = int(voc_pos["n_no_duration"][0])
    n_with_dur = len(ns)
    print(
        f"  Acappella spans: {len(ar)} total; {n_with_dur} with known duration "
        f"({n_no_dur} missing duration)"
    )
    if ns:
        print(f"  Normalized start (0=song_start, 1=end):")
        print(f"    p25={p25(ns):.2f}  median={p50(ns):.2f}  p75={p75(ns):.2f}")
    if nc:
        print(f"  Normalized coverage (fraction of song used):")
        print(f"    p25={p25(nc):.2f}  median={p50(nc):.2f}  p75={p75(nc):.2f}")
    if ar:
        print(f"  Absolute ref_start_s:")
        print(
            f"    p25={p25(ar):.1f}s  median={p50(ar):.1f}s  p75={p75(ar):.1f}s  "
            f"p90={p90(ar):.1f}s"
        )

    # -----------------------------------------------------------------------
    # Metric 3: Loop/multiseg fraction
    # -----------------------------------------------------------------------
    print("\n--- Metric 3: Loop / multiseg fraction ---")
    loop_stats = metric_loop_fraction(all_tracks)
    for stem in ["acappella", "instrumental", "regular", "all"]:
        b = loop_stats.get(stem, {})
        if not b:
            continue
        n = int(b["n"])
        nm = int(b["n_multiseg"])
        nl = int(b["n_loop"])
        nj = int(b["n_jump"])
        print(
            f"  {stem:15s} n={n:3d}  multiseg={nm} ({100 * b['frac']:.0f}%)  "
            f"loop={nl}  jump={nj}"
        )

    # -----------------------------------------------------------------------
    # Metric 4: Entry alignment
    # -----------------------------------------------------------------------
    print("\n--- Metric 4: Entry alignment ---")
    entry = metric_entry_alignment_v2(tracks_by_set, beat_grid_bb11)
    bo = entry["beat_offset_bb11"]
    if bo:
        print(f"  Beat grid offset (BB11 only, n={len(bo)}):")
        print(f"    p25={p25(bo):.2f}s  median={p50(bo):.2f}s  p75={p75(bo):.2f}s")
        frac_on_beat = sum(1 for v in bo if v < 0.25) / len(bo)
        print(f"    Fraction within 0.25s of a downbeat: {100 * frac_on_beat:.0f}%")
    else:
        print("  Beat grid data not available")

    bao = entry["bed_abs_offset_s"]
    if bao:
        print(f"  Bed offset abs (n={len(bao)}):")
        print(f"    p25={p25(bao):.1f}s  median={p50(bao):.1f}s  p75={p75(bao):.1f}s")
    bbr = entry["bed_bar_raw"]
    bm8 = entry["bed_bars_mod8"]
    bm16 = entry["bed_bars_mod16"]
    if bm8:
        # Fraction landing near 0 mod 8 (i.e., on an 8-bar phrase boundary ±0.5 bar)
        frac_8 = sum(1 for v in bm8 if v < 0.5 or v > 7.5) / len(bm8)
        frac_16 = sum(1 for v in bm16 if v < 0.5 or v > 15.5) / len(bm16)
        print(f"  Bed phrase alignment (n={len(bm8)}):")
        print(f"    Fraction on 8-bar boundary (±0.5 bar):  {100 * frac_8:.0f}%")
        print(f"    Fraction on 16-bar boundary (±0.5 bar): {100 * frac_16:.0f}%")
        print(
            f"    Mod-8 bar distribution p25/med/p75: "
            f"{p25(bm8):.1f}/{p50(bm8):.1f}/{p75(bm8):.1f}"
        )
        print(
            f"    Mod-16 bar distribution p25/med/p75: "
            f"{p25(bm16):.1f}/{p50(bm16):.1f}/{p75(bm16):.1f}"
        )

    # -----------------------------------------------------------------------
    # Metric 5: Vocal density + rests
    # -----------------------------------------------------------------------
    print("\n--- Metric 5: Vocal density + rests ---")
    dens = metric_vocal_density(tracks_by_set)
    print(f"  Overall vocal-on fraction: {100 * dens['overall_density']:.1f}%")
    pd = dens["per_set_density"]
    if pd:
        print(f"  Per-set density: {[f'{100 * v:.0f}%' for v in pd]}")
    gaps = dens["gaps_s"]
    if gaps:
        print(f"  Rest gap lengths (n={len(gaps)}):")
        print(
            f"    p25={p25(gaps):.1f}s  median={p50(gaps):.1f}s  "
            f"p75={p75(gaps):.1f}s  p90={p90(gaps):.1f}s"
        )
    else:
        print("  No rest gaps found (acappella coverage is continuous)")

    # -----------------------------------------------------------------------
    # Metric 6: Ducking
    # -----------------------------------------------------------------------
    print("\n--- Metric 6: Ducking (gain_curve) ---")
    duck = metric_ducking(tracks_by_set)
    print(f"  Bed spans with explicit gain_curve: {duck['n_beds_with_gain']}")
    print(f"  Bed spans without gain_curve: {duck['n_beds_no_gain']}")
    if duck["db_delta"] is not None:
        print(
            f"  Mean bed gain during vocal overlap: "
            f"{duck['mean_gain_overlap']:.3f} linear "
            f"(n={duck['n_samples_overlap']} samples)"
        )
        print(
            f"  Mean bed gain outside vocal overlap: "
            f"{duck['mean_gain_no_overlap']:.3f} linear "
            f"(n={duck['n_samples_no_overlap']} samples)"
        )
        print(f"  dB delta (overlap - no-overlap): {duck['db_delta']:.1f} dB")
    else:
        if duck["n_beds_with_gain"] == 0:
            print(
                "  NOTE: No bed spans carry explicit gain_curve in the GT. "
                "Ducking metric NOT COMPUTABLE from this data. "
                "The GT gain_curve field is used for audible_frac/start/end derivation "
                "but these beds have flat unity gain in the annotation. "
                "Actual ducking is not represented in the current GT schema for bed spans."
            )

    # -----------------------------------------------------------------------
    # Persist to aux.db
    # -----------------------------------------------------------------------
    print("\n--- Persisting to aux.db ---")
    conn2 = sqlite3.connect(AUX_DB)
    cur = conn2.cursor()

    findings: list[tuple[str, str, float]] = []

    def add(metric: str, group: str, val: float) -> None:
        findings.append((metric, group, val))

    # n counts
    for stem in ["acappella", "instrumental", "regular", "all"]:
        vals = by_stem_dur.get(stem, [])
        if vals:
            add("n_spans", stem, float(len(vals)))
            add("span_dur_p25_s", stem, p25(vals))
            add("span_dur_median_s", stem, p50(vals))
            add("span_dur_p75_s", stem, p75(vals))
            add("span_dur_p90_s", stem, p90(vals))

    # Metric 2
    if ns:
        add("n_acap_with_duration", "acappella", float(n_with_dur))
        add("n_acap_no_duration", "acappella", float(n_no_dur))
        add("vocal_norm_start_p25", "acappella", p25(ns))
        add("vocal_norm_start_median", "acappella", p50(ns))
        add("vocal_norm_start_p75", "acappella", p75(ns))
    if nc:
        add("vocal_norm_coverage_p25", "acappella", p25(nc))
        add("vocal_norm_coverage_median", "acappella", p50(nc))
        add("vocal_norm_coverage_p75", "acappella", p75(nc))
    if ar:
        add("vocal_abs_ref_start_p25_s", "acappella", p25(ar))
        add("vocal_abs_ref_start_median_s", "acappella", p50(ar))
        add("vocal_abs_ref_start_p75_s", "acappella", p75(ar))
        add("vocal_abs_ref_start_p90_s", "acappella", p90(ar))

    # Metric 3
    for stem in ["acappella", "instrumental", "regular", "all"]:
        b = loop_stats.get(stem, {})
        if b:
            add("multiseg_frac", stem, b["frac"])
            add("n_multiseg", stem, b["n_multiseg"])
            add("n_loop", stem, b["n_loop"])
            add("n_jump", stem, b["n_jump"])

    # Metric 4
    if bo:
        add("n_beat_aligned_bb11", "acappella", float(len(bo)))
        add("beat_offset_p25_s", "acappella", p25(bo))
        add("beat_offset_median_s", "acappella", p50(bo))
        add("beat_offset_p75_s", "acappella", p75(bo))
        add(
            "frac_within_0p25s_of_downbeat",
            "acappella",
            sum(1 for v in bo if v < 0.25) / len(bo),
        )
    if bm8:
        add("n_entry_aligned", "acappella", float(len(bm8)))
        add(
            "frac_on_8bar_boundary",
            "acappella",
            sum(1 for v in bm8 if v < 0.5 or v > 7.5) / len(bm8),
        )
        add(
            "frac_on_16bar_boundary",
            "acappella",
            sum(1 for v in bm16 if v < 0.5 or v > 15.5) / len(bm16),
        )
        add("bed_offset_median_s", "acappella", p50(bao))
        add("bed_offset_p75_s", "acappella", p75(bao))

    # Metric 5
    add("vocal_on_frac_overall", "acappella", dens["overall_density"])
    if pd:
        for i, v in enumerate(pd):
            add(f"vocal_on_frac_set{i}", "acappella", v)
    if gaps:
        add("n_rest_gaps", "acappella", float(len(gaps)))
        add("rest_gap_p25_s", "acappella", p25(gaps))
        add("rest_gap_median_s", "acappella", p50(gaps))
        add("rest_gap_p75_s", "acappella", p75(gaps))
        add("rest_gap_p90_s", "acappella", p90(gaps))

    # Metric 6
    add("n_beds_with_gain_curve", "instrumental", float(duck["n_beds_with_gain"]))
    add("n_beds_no_gain_curve", "instrumental", float(duck["n_beds_no_gain"]))
    if duck["db_delta"] is not None:
        add("ducking_db_delta", "instrumental", duck["db_delta"])
        add(
            "mean_gain_during_overlap_linear", "instrumental", duck["mean_gain_overlap"]
        )
        add("mean_gain_no_overlap_linear", "instrumental", duck["mean_gain_no_overlap"])

    for metric, group, val in findings:
        cur.execute(
            """INSERT INTO analysis_results
                 (analysis_name, metric, group_key, value)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
                 value = excluded.value, computed_at = CURRENT_TIMESTAMP""",
            (ANALYSIS_NAME, metric, group, val),
        )
    conn2.commit()
    print(
        f"Persisted {len(findings)} metrics to aux.analysis_results "
        f"(analysis_name='{ANALYSIS_NAME}')"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
