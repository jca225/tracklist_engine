"""Empirical cue-detr eval against OUR ground truth (not the paper's DJs).

cue-detr (canonical_track_cue_points) proposes REF-time cue markers. Two
empirical questions, both answerable locally:

  A. GRID COHERENCE — DJ-usable cue points sit on downbeats. What fraction
     of our stored cues are within 0.25 s of a beat_this downbeat, vs the
     rate a uniform-random marker would achieve?

  B. GT ENTRY UTILITY — when the DJ actually entered a reference (GT
     ref_start_s, plus every loop segment's ref_start_s), how close is the
     nearest stored cue? Compared against a seeded Monte-Carlo null (random
     positions in the same track with the same cue set), this is the lift
     cue-detr gives as a ref-offset prior — the "soft channel not gate"
     question, measured.

Coverage caveat: local dev DB is BB11-era (~120 analyzed tracks), so B is
dominated by BB11 spans; ref_source is restricted to {reference, demucs} so
GT times share the analyzed file's timeline.

Run from repo root:
    PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/cue_eval.py
"""

from __future__ import annotations

import bisect
import json
import random
import sqlite3
import statistics
from pathlib import Path

from labeling.schema import load

from eda.alignment.placement_structure import repo_root

REPO = repo_root()
FIX = REPO / "labeling" / "fixtures"
DB = REPO / "data" / "db" / "music_database.db"
OUT = Path(__file__).resolve().parent / "out"
SETS = {
    "bb11": FIX / "bb11_ground_truth.yaml",
    "bb12": FIX / "bb12_ground_truth.yaml",
}
TIMELINE_SAFE_REF_SOURCES = {"reference", "demucs"}
DOWNBEAT_TOL_S = 0.25
ENTRY_THRESHOLDS_S = (2.0, 4.0, 8.0)
MC_DRAWS = 500


def nearest_dist(t: float, sorted_times: list[float]) -> float:
    i = bisect.bisect_left(sorted_times, t)
    cands = []
    if i < len(sorted_times):
        cands.append(abs(sorted_times[i] - t))
    if i > 0:
        cands.append(abs(sorted_times[i - 1] - t))
    return min(cands)


def load_db() -> tuple[
    dict[str, list[float]], dict[str, list[float]], dict[str, float]
]:
    """track_id -> (cues, downbeat grid, duration)."""
    con = sqlite3.connect(DB)
    cues: dict[str, list[float]] = {}
    for tid, blob in con.execute(
        "select track_id, cue_points_json from canonical_track_cue_points"
    ):
        cues[str(tid)] = sorted(json.loads(blob))
    grids: dict[str, list[float]] = {}
    durs: dict[str, float] = {}
    for tid, blob, dur in con.execute(
        """
        select ta.track_id, an.downbeat_times_json, ta.duration_s
        from track_audio ta join track_analysis an using(track_audio_id)
        where an.downbeat_times_json is not null
        order by ta.is_reference desc
        """
    ):
        if str(tid) not in grids:
            grids[str(tid)] = json.loads(blob)
            durs[str(tid)] = float(dur or 0.0)
    con.close()
    return cues, grids, durs


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cues, grids, durs = load_db()

    # A. grid coherence
    cue_db_dists: list[float] = []
    bar_lens: list[float] = []
    for tid, cue_list in cues.items():
        g = grids.get(tid)
        if not g or len(g) < 2:
            continue
        bar_lens.append(statistics.median(b - a for a, b in zip(g, g[1:])))
        for c in cue_list:
            if g[0] <= c <= g[-1]:
                cue_db_dists.append(nearest_dist(c, g))
    med_bar = statistics.median(bar_lens)
    frac_on_downbeat = sum(1 for d in cue_db_dists if d <= DOWNBEAT_TOL_S) / len(
        cue_db_dists
    )
    null_on_downbeat = min(1.0, 2 * DOWNBEAT_TOL_S / med_bar)

    coherence = {
        "n_cues": len(cue_db_dists),
        "median_dist_to_downbeat_s": round(statistics.median(cue_db_dists), 3),
        "frac_within_0.25s": round(frac_on_downbeat, 3),
        "uniform_null_frac": round(null_on_downbeat, 3),
        "median_bar_len_s": round(med_bar, 3),
    }

    # B. GT entry utility
    rng = random.Random(20260710)
    per_stem: dict[str, dict[str, list[float]]] = {}
    entries: list[tuple[str, str, float]] = []  # (stem, track_id, ref_time)
    for name, path in SETS.items():
        gt = load(path).value
        for t in gt.tracks:
            if t.unalignable or t.ref_source not in TIMELINE_SAFE_REF_SOURCES:
                continue
            tid = t.track_id or ""
            if tid not in cues or not cues[tid]:
                continue
            entries.append((t.claimed_stem, tid, t.ref_start_s))
            for seg in t.ref_segments[1:]:
                entries.append((t.claimed_stem + "_loopseg", tid, seg.ref_start_s))

    utility: dict[str, dict] = {}
    for stem in sorted({s for s, _, _ in entries}):
        evs = [(tid, rt) for s, tid, rt in entries if s == stem]
        dists = [nearest_dist(rt, cues[tid]) for tid, rt in evs]
        null_dists: list[float] = []
        for tid, _ in evs:
            dur = durs.get(tid) or (max(cues[tid]) + 30.0)
            for _ in range(MC_DRAWS):
                null_dists.append(nearest_dist(rng.uniform(0, dur), cues[tid]))
        n = len(dists)
        row: dict = {
            "n": n,
            "median_dist_s": round(statistics.median(dists), 2),
            "null_median_dist_s": round(statistics.median(null_dists), 2),
        }
        for thr in ENTRY_THRESHOLDS_S:
            hit = sum(1 for d in dists if d <= thr) / n
            null_hit = sum(1 for d in null_dists if d <= thr) / len(null_dists)
            row[f"hit@{thr:g}s"] = round(hit, 3)
            row[f"null@{thr:g}s"] = round(null_hit, 3)
            row[f"lift@{thr:g}s"] = round(hit / null_hit, 2) if null_hit else None
        utility[stem] = row

    # descriptives
    per_track_counts = sorted(len(v) for v in cues.values())
    cues_per_min = sorted(
        len(v) / (durs[tid] / 60.0) for tid, v in cues.items() if durs.get(tid)
    )
    desc = {
        "n_tracks_with_cues": len(cues),
        "cues_per_track": {
            "median": statistics.median(per_track_counts),
            "min": per_track_counts[0],
            "max": per_track_counts[-1],
        },
        "cues_per_minute_median": round(statistics.median(cues_per_min), 2),
    }

    result = {
        "descriptives": desc,
        "grid_coherence": coherence,
        "gt_entry_utility": utility,
    }
    (OUT / "cue_eval.json").write_text(json.dumps(result, indent=2))

    print("== cue descriptives ==")
    print(
        f"  tracks={desc['n_tracks_with_cues']}  cues/track median="
        f"{desc['cues_per_track']['median']} [{desc['cues_per_track']['min']},"
        f"{desc['cues_per_track']['max']}]  cues/min median={desc['cues_per_minute_median']}"
    )
    print("\n== A. downbeat coherence ==")
    print(
        f"  n={coherence['n_cues']} cues  median dist to downbeat="
        f"{coherence['median_dist_to_downbeat_s']}s  within 0.25s: "
        f"{coherence['frac_within_0.25s']:.0%} (uniform null {coherence['uniform_null_frac']:.0%})"
    )
    print("\n== B. GT entry utility (nearest cue to actual DJ entry, ref time) ==")
    for stem, r in utility.items():
        print(
            f"  {stem:22s} n={r['n']:3d}  median={r['median_dist_s']:6.2f}s "
            f"(null {r['null_median_dist_s']:6.2f}s)  "
            + "  ".join(
                f"hit@{thr:g}s={r[f'hit@{thr:g}s']:.0%} (lift {r[f'lift@{thr:g}s']}x)"
                for thr in ENTRY_THRESHOLDS_S
            )
        )
    print(f"\nwrote {OUT / 'cue_eval.json'}")


if __name__ == "__main__":
    main()
