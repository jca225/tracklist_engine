"""Is there an objective optimal placement? Test GT placements against grids.

Three tests on the hand-labeled GT (BB11 + BB12):

  A. MIX-GRID SNAP — do GT entries / exits / loop-jump points land on the mix
     bar grid (data/analysis/<set_id>_measure_times.json)? If placement is
     objective, event phases concentrate at the bar line (Rayleigh R -> 1);
     if placement were free, phases would be uniform (R -> 0).

  B. PHRASE OFFSET VS BED — for each acappella, the bar-index offset of its
     entry relative to the concurrent bed track's entry, mod 4/8/16 bars.
     Concentration at 0 mod k means acappellas enter on the bed's phrase
     grid, i.e. the placement search space is the phrase lattice, not R.

  C. REF-SIDE SNAP — does the DJ cut INTO the reference at the reference's
     own bar lines? Uses local downbeat grids (track_analysis), restricted
     to ref_source in {reference, demucs} so the GT timeline matches the
     analyzed file. Coverage is BB11-heavy (local dev DB).

  D. PHASE TRANSFER — the entry time itself can sit off-grid (vocal pickup)
     while the placement is still objective: what must match is the PHASE,
     i.e. where the entry falls inside the ref track's own bar equals where
     it falls inside the mix bar. Delta-phase = (mix beat phase - ref beat
     phase) mod 1 concentrating near 0 means the ref's beat grid is laid
     onto the mix grid exactly — placement is a lattice choice, not a free
     continuous variable — even when neither phase alone concentrates.

Run from repo root:
    PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/placement_grid.py
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

from labeling.ground_truth.schema import GroundTruthTrack, load

from eda.alignment.placement_structure import repo_root
from eda.alignment.placement_structure.gridmath import (
    PhaseSample,
    circular_stats,
    phase_in_grid,
    phase_report,
)

REPO = repo_root()
FIX = REPO / "labeling" / "fixtures"
DB = REPO / "data" / "db" / "music_database.db"
OUT = Path(__file__).resolve().parent / "out"
SETS = {
    "bb11": ("2nvzlh2k", FIX / "bb11_ground_truth.yaml"),
    "bb12": ("1fsnxchk", FIX / "bb12_ground_truth.yaml"),
}
TIMELINE_SAFE_REF_SOURCES = {"reference", "demucs"}


def mix_grid(set_id: str) -> list[float]:
    return json.loads((REPO / f"data/analysis/{set_id}_measure_times.json").read_text())


def load_ref_downbeat_grids() -> dict[str, list[float]]:
    """track_id (legacy recording id) -> downbeat grid of its analyzed audio."""
    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        select ta.track_id, an.downbeat_times_json
        from track_audio ta
        join track_analysis an using(track_audio_id)
        where an.downbeat_times_json is not null
        order by ta.is_reference desc
        """
    ).fetchall()
    con.close()
    grids: dict[str, list[float]] = {}
    for tid, blob in rows:
        if str(tid) not in grids:  # is_reference row wins
            grids[str(tid)] = json.loads(blob)
    return grids


def collect_events(
    tracks: tuple[GroundTruthTrack, ...],
) -> dict[str, list[tuple[GroundTruthTrack, float]]]:
    """Event classes -> (track, mix-time) pairs."""
    usable = [t for t in tracks if not t.unalignable]
    events: dict[str, list[tuple[GroundTruthTrack, float]]] = {
        "entries": [(t, t.set_start_s) for t in usable],
        "exits": [(t, t.set_end_s) for t in usable],
        "loop_jumps": [
            (t, seg.mix_start_s)
            for t in usable
            for seg in t.ref_segments[1:]  # first segment == the entry itself
        ],
    }
    return events


def sample_phases(
    events: list[tuple[GroundTruthTrack, float]], grid: list[float]
) -> list[PhaseSample]:
    out = []
    for _, t in events:
        s = phase_in_grid(t, grid)
        if s is not None:
            out.append(s)
    return out


def bed_phrase_offsets(
    tracks: tuple[GroundTruthTrack, ...], grid: list[float]
) -> list[int]:
    """Bar-index offset of each acappella entry vs its concurrent bed's entry."""
    usable = [t for t in tracks if not t.unalignable]
    beds = [t for t in usable if t.claimed_stem in ("instrumental", "regular")]
    offsets: list[int] = []
    for t in usable:
        if t.claimed_stem != "acappella":
            continue
        cov = [b for b in beds if b.set_start_s <= t.set_start_s <= b.set_end_s]
        if not cov:
            continue
        bed = min(cov, key=lambda b: abs(b.set_start_s - t.set_start_s))
        sa = phase_in_grid(t.set_start_s, grid)
        sb = phase_in_grid(bed.set_start_s, grid)
        if sa is None or sb is None:
            continue
        offsets.append(sa.bar_idx - sb.bar_idx)
    return offsets


def mod_report(offsets: list[int], k: int) -> dict:
    n = len(offsets)
    counts = Counter(o % k for o in offsets)
    top_residue, top_count = counts.most_common(1)[0]
    return {
        "k": k,
        "top_residue": top_residue,
        "top_frac": round(top_count / n, 3),
        "uniform_expectation": round(1 / k, 3),
        "counts": {str(r): counts.get(r, 0) for r in range(k)},
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ref_grids = load_ref_downbeat_grids()
    result: dict = {}

    all_entries_by_stem: dict[str, list[PhaseSample]] = {}
    all_ref_side: dict[str, list[PhaseSample]] = {}
    all_phase_transfer: dict[str, list[float]] = {}
    all_offsets: list[int] = []

    for name, (set_id, path) in SETS.items():
        gt = load(path).value
        grid = mix_grid(set_id)
        events = collect_events(gt.tracks)

        set_res: dict = {"set_id": set_id, "n_tracks": len(gt.tracks)}

        # A. mix-grid snap per event class
        for cls, evs in events.items():
            set_res[cls] = phase_report(sample_phases(evs, grid))

        # entries split by stem (pooled across sets below)
        for t, tt in events["entries"]:
            s = phase_in_grid(tt, grid)
            if s is not None:
                all_entries_by_stem.setdefault(t.claimed_stem, []).append(s)

        # B. phrase offsets acappella vs bed
        offsets = bed_phrase_offsets(gt.tracks, grid)
        all_offsets.extend(offsets)
        set_res["acap_vs_bed_bar_offsets"] = {
            "n": len(offsets),
            "mods": [mod_report(offsets, k) for k in (4, 8, 16)] if offsets else [],
        }

        # C. ref-side snap (timeline-safe sources only)
        # D. phase transfer: mix beat phase minus ref beat phase per entry
        covered = 0
        for t in gt.tracks:
            if t.unalignable or t.ref_source not in TIMELINE_SAFE_REF_SOURCES:
                continue
            g = ref_grids.get(t.track_id or "")
            if not g:
                continue
            s = phase_in_grid(t.ref_start_s, g)
            if s is None:
                continue
            covered += 1
            all_ref_side.setdefault(t.claimed_stem, []).append(s)
            sm = phase_in_grid(t.set_start_s, grid)
            if sm is not None:
                d_beat = ((sm.phase - s.phase) * 4) % 1.0
                all_phase_transfer.setdefault(t.claimed_stem, []).append(d_beat)
        set_res["ref_side_coverage"] = covered
        result[name] = set_res

    result["pooled_entries_by_stem"] = {
        stem: phase_report(samples) for stem, samples in all_entries_by_stem.items()
    }
    result["pooled_ref_side_by_stem"] = {
        stem: phase_report(samples) for stem, samples in all_ref_side.items()
    }
    result["pooled_acap_vs_bed_mods"] = (
        [mod_report(all_offsets, k) for k in (4, 8, 16)] if all_offsets else []
    )
    transfer: dict[str, dict] = {}
    for stem, deltas in all_phase_transfer.items():
        r, mu, p = circular_stats(deltas)
        # wrapped distance of each delta-phase from the circular mean, in beats
        dev = sorted(min(abs(d - mu), 1 - abs(d - mu)) for d in deltas)
        transfer[stem] = {
            "n": len(deltas),
            "R": round(r, 3),
            "mean_delta_beats": round(mu, 3),
            "p": p,
            "median_dev_beats": round(statistics.median(dev), 3),
            "frac_within_tenth_beat": round(
                sum(1 for d in dev if d <= 0.1) / len(dev), 3
            ),
        }
    result["pooled_phase_transfer_by_stem"] = transfer

    (OUT / "placement_grid.json").write_text(json.dumps(result, indent=2))

    # ---- console report ----
    for name in SETS:
        r = result[name]
        print(f"\n== {name} ({r['set_id']}) — {r['n_tracks']} GT spans ==")
        for cls in ("entries", "exits", "loop_jumps"):
            s = r[cls]
            if s["n"] == 0:
                print(f"  {cls:11s} n=0")
                continue
            print(
                f"  {cls:11s} n={s['n']:3d}  median|d bar|={s['median_dist_to_barline_s']:.3f}s  "
                f"median|d beat|={s['median_dist_to_beat_s']:.3f}s  "
                f"beat-R={s['beat']['R']:.2f} (p={s['beat']['p']:.1e})  "
                f"bar-R={s['bar']['R']:.2f} (p={s['bar']['p']:.1e})"
            )
        ao = r["acap_vs_bed_bar_offsets"]
        print(f"  acap-vs-bed offsets n={ao['n']}")
        for m in ao["mods"]:
            print(
                f"    mod {m['k']:2d}: top residue {m['top_residue']} at "
                f"{m['top_frac']:.0%} (uniform {m['uniform_expectation']:.0%})"
            )
        print(f"  ref-side coverage: {r['ref_side_coverage']} spans")

    print("\n== pooled entries by stem (mix grid) ==")
    for stem, s in result["pooled_entries_by_stem"].items():
        print(
            f"  {stem:12s} n={s['n']:3d}  median|d beat|={s['median_dist_to_beat_s']:.3f}s  "
            f"beat-R={s['beat']['R']:.2f} (p={s['beat']['p']:.1e})"
        )
    print("\n== pooled ref-side snap by stem (ref downbeat grid) ==")
    for stem, s in result["pooled_ref_side_by_stem"].items():
        print(
            f"  {stem:12s} n={s['n']:3d}  median|d bar|={s['median_dist_to_barline_s']:.3f}s  "
            f"median|d beat|={s['median_dist_to_beat_s']:.3f}s  "
            f"beat-R={s['beat']['R']:.2f} (p={s['beat']['p']:.1e})"
        )
    print("\n== pooled phase transfer (mix beat phase - ref beat phase) ==")
    for stem, s in result["pooled_phase_transfer_by_stem"].items():
        print(
            f"  {stem:12s} n={s['n']:3d}  R={s['R']:.2f} (p={s['p']:.1e})  "
            f"median dev from mean = {s['median_dev_beats']:.3f} beats  "
            f"within 0.1 beat: {s['frac_within_tenth_beat']:.0%}"
        )
    print("\n== pooled acap-vs-bed phrase lattice ==")
    for m in result["pooled_acap_vs_bed_mods"]:
        print(
            f"  mod {m['k']:2d}: top residue {m['top_residue']} at {m['top_frac']:.0%} "
            f"(uniform {m['uniform_expectation']:.0%})  counts={m['counts']}"
        )
    print(f"\nwrote {OUT / 'placement_grid.json'}")


if __name__ == "__main__":
    main()
