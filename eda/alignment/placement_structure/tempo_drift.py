"""How badly does the constant-tempo assumption fail — and how much of the
apparent failure is the beat tracker, not the music?

BPM in this repo is a global scalar (beat_this median inter-beat interval,
Essentia scalar). Two distinct problems hide under "tempo drift":

  1. TRACKER INSTABILITY — half/double-time flips and missed/inserted beats
     in the raw beat grid. Fold-correcting intervals (x2 / /2 toward the
     track median) removes these; the fraction of intervals that needed
     folding measures how unreliable the raw grid is.
  2. TRUE TEMPO DRIFT — what survives fold-correction: real BPM movement
     (live drummers, ritardando outros, DJ tempo rides inside a mix).

Part 1 — reference tracks (local dev DB, track_analysis.beat_times_json):
  per-track fold-corrected instantaneous BPM curve; spread; and the
  operational metric: anchor a constant-BPM grid on the first 32 beats and
  measure cumulative timing error by the last beat. >0.5 beat = audibly
  desynced if you trusted a single global BPM.

Part 2 — mixes (data/analysis/<set_id>_measure_times.json, bar starts):
  fold-corrected bar-level BPM curve per set: range, plateaus, spread —
  the within-mix tempo variation the aligner must absorb.

Run from repo root:
    PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/tempo_drift.py
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

from eda.alignment.placement_structure import repo_root
from eda.alignment.placement_structure.gridmath import fold_intervals

REPO = repo_root()
DB = REPO / "data" / "db" / "music_database.db"
ANALYSIS_DIR = REPO / "data" / "analysis"
OUT = Path(__file__).resolve().parent / "out"

ANCHOR_BEATS = 32  # beats used to estimate the "drop-in" BPM
PLATEAU_SHIFT = 0.01  # 1% sustained tempo change = new plateau
PLATEAU_MIN_RUN = 16  # beats/bars the shift must persist


def smooth_median(vals: list[float], window: int = 9) -> list[float]:
    half = window // 2
    return [
        statistics.median(vals[max(0, k - half) : k + half + 1])
        for k in range(len(vals))
    ]


def count_plateaus(bpm: list[float], med: float) -> int:
    plateaus = 1
    run = 0
    away = False
    for v in bpm:
        off = abs(v - med) / med > PLATEAU_SHIFT
        if off == away:
            run += 1
        else:
            away, run = off, 1
        if run == PLATEAU_MIN_RUN and off:
            plateaus += 1
    return plateaus


def track_drift(beat_times: list[float]) -> dict | None:
    """Fold-corrected drift metrics for one track's raw beat grid."""
    ibis = [b - a for a, b in zip(beat_times, beat_times[1:])]
    ibis = [i for i in ibis if 0.05 <= i <= 4.0]  # drop degenerate stamps
    if len(ibis) < ANCHOR_BEATS + 16:
        return None
    med_ibi = statistics.median(ibis)
    folded, fold_frac = fold_intervals(ibis, med_ibi)
    bpm = smooth_median([60.0 / i for i in folded])
    med = statistics.median(bpm)
    qs = statistics.quantiles(bpm, n=20)
    spread_pct = (qs[18] - qs[0]) / med * 100  # p95 - p5

    # anchored constant-tempo grid over the fold-corrected timeline
    anchor_ibi = statistics.median(folded[:ANCHOR_BEATS])
    end_err_s = abs(sum(folded) - len(folded) * anchor_ibi)
    end_err_beats = end_err_s / anchor_ibi

    return {
        "n_beats": len(beat_times),
        "median_bpm": round(med, 2),
        "fold_frac": round(fold_frac, 3),
        "spread_p5_p95_pct": round(spread_pct, 2),
        "end_err_s": round(end_err_s, 2),
        "end_err_beats": round(end_err_beats, 2),
        "plateaus": count_plateaus(bpm, med),
    }


def mix_drift(grid: list[float]) -> dict:
    """Fold-corrected bar-level tempo curve stats for one mix (assumes 4/4)."""
    ivals = [b - a for a, b in zip(grid, grid[1:]) if 0.2 <= b - a <= 8.0]
    med_ival = statistics.median(ivals)
    folded, fold_frac = fold_intervals(ivals, med_ival)
    bpm = smooth_median([240.0 / i for i in folded], window=5)
    med = statistics.median(bpm)
    qs = statistics.quantiles(bpm, n=20)
    return {
        "n_bars": len(grid),
        "median_bpm": round(med, 2),
        "fold_frac": round(fold_frac, 3),
        "p5_bpm": round(qs[0], 2),
        "p95_bpm": round(qs[18], 2),
        "spread_p5_p95_pct": round((qs[18] - qs[0]) / med * 100, 2),
        "plateaus": count_plateaus(bpm, med),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        select ta.track_audio_id, ta.track_id, an.beat_times_json
        from track_audio ta join track_analysis an using(track_audio_id)
        where an.beat_times_json is not null
        """
    ).fetchall()
    con.close()

    tracks: dict[str, dict] = {}
    for taid, tid, blob in rows:
        m = track_drift(json.loads(blob))
        if m is not None:
            tracks[f"{tid}/{taid}"] = m

    n = len(tracks)
    spreads = sorted(t["spread_p5_p95_pct"] for t in tracks.values())
    end_beats = sorted(t["end_err_beats"] for t in tracks.values())
    end_secs = sorted(t["end_err_s"] for t in tracks.values())
    fold_fracs = sorted(t["fold_frac"] for t in tracks.values())

    def frac_gt(vals: list[float], thr: float) -> float:
        return sum(1 for v in vals if v > thr) / len(vals)

    summary = {
        "n_tracks": n,
        "fold_frac": {
            "median": round(statistics.median(fold_fracs), 3),
            "p90": round(fold_fracs[int(0.9 * (n - 1))], 3),
            "frac_tracks_gt_5pct_folded": round(frac_gt(fold_fracs, 0.05), 3),
        },
        "spread_pct": {
            "median": round(statistics.median(spreads), 2),
            "p90": round(spreads[int(0.9 * (n - 1))], 2),
        },
        "frac_spread_gt_2pct": round(frac_gt(spreads, 2), 3),
        "frac_spread_gt_5pct": round(frac_gt(spreads, 5), 3),
        "end_err_beats": {
            "median": round(statistics.median(end_beats), 2),
            "p90": round(end_beats[int(0.9 * (n - 1))], 2),
        },
        "end_err_s": {
            "median": round(statistics.median(end_secs), 2),
            "p90": round(end_secs[int(0.9 * (n - 1))], 2),
        },
        "frac_end_err_gt_half_beat": round(frac_gt(end_beats, 0.5), 3),
        "frac_end_err_gt_one_beat": round(frac_gt(end_beats, 1.0), 3),
        "frac_multi_plateau": round(
            sum(1 for t in tracks.values() if t["plateaus"] > 1) / n, 3
        ),
    }

    mixes = {}
    for p in sorted(ANALYSIS_DIR.glob("*_measure_times.json")):
        grid = json.loads(p.read_text())
        if len(grid) > 64:
            mixes[p.stem.replace("_measure_times", "")] = mix_drift(grid)

    result = {"track_summary": summary, "tracks": tracks, "mixes": mixes}
    (OUT / "tempo_drift.json").write_text(json.dumps(result, indent=2))

    print(f"== reference tracks (n={n}, local dev DB — BB11-era sample) ==")
    print(
        f"  tracker instability (fold_frac): median {summary['fold_frac']['median']:.1%}  "
        f"p90 {summary['fold_frac']['p90']:.1%}  tracks >5% folded: "
        f"{summary['fold_frac']['frac_tracks_gt_5pct_folded']:.0%}"
    )
    print(
        f"  TRUE BPM spread (p95-p5)/median after folding: median "
        f"{summary['spread_pct']['median']}%  p90 {summary['spread_pct']['p90']}%"
    )
    print(
        f"  tracks with true spread >2%: {summary['frac_spread_gt_2pct']:.0%}   "
        f">5%: {summary['frac_spread_gt_5pct']:.0%}"
    )
    print(
        f"  anchored end-of-track error: median {summary['end_err_beats']['median']} beats "
        f"({summary['end_err_s']['median']}s)  p90 {summary['end_err_beats']['p90']} beats "
        f"({summary['end_err_s']['p90']}s)"
    )
    print(
        f"  tracks desynced >0.5 beat by end: {summary['frac_end_err_gt_half_beat']:.0%}   "
        f">1 beat: {summary['frac_end_err_gt_one_beat']:.0%}"
    )
    print(
        f"  tracks with >1 sustained tempo plateau: {summary['frac_multi_plateau']:.0%}"
    )

    worst = sorted(tracks.items(), key=lambda kv: -kv[1]["end_err_beats"])[:8]
    print("\n  worst anchored-drift tracks (track_id/taid):")
    for k, t in worst:
        print(
            f"    {k:16s} bpm={t['median_bpm']:6.1f} spread={t['spread_p5_p95_pct']:5.1f}% "
            f"end_err={t['end_err_beats']:6.1f} beats ({t['end_err_s']:6.1f}s) "
            f"folded={t['fold_frac']:.0%}"
        )

    print(f"\n== mixes (n={len(mixes)}, fold-corrected bar grids) ==")
    for sid, m in sorted(mixes.items(), key=lambda kv: -kv[1]["spread_p5_p95_pct"])[
        :12
    ]:
        print(
            f"  {sid:10s} median={m['median_bpm']:6.1f} bpm  p5-p95=[{m['p5_bpm']:6.1f},"
            f"{m['p95_bpm']:6.1f}] ({m['spread_p5_p95_pct']:4.1f}%)  "
            f"plateaus={m['plateaus']}  folded={m['fold_frac']:.0%}"
        )
    print(f"\nwrote {OUT / 'tempo_drift.json'}")


if __name__ == "__main__":
    main()
