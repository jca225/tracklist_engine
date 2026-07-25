"""The anatomy of a Big Bootie mashup, in bar units.

The generation stage needs an arrangement grammar: how long a bed plays, how
many acappellas ride it, where they enter/exit relative to the bed, and how
long a vocal stays on. This measures all of that from BB11+BB12 GT, expressed
in mix-grid bars (the unit generation would compose in).

Run from repo root:
    PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/arrangement_anatomy.py
"""

from __future__ import annotations

import bisect
import json
import statistics
from collections import Counter
from pathlib import Path

from labeling.schema import GroundTruthTrack, load

from eda.alignment.placement_structure import repo_root

REPO = repo_root()
FIX = REPO / "labeling" / "fixtures"
OUT = Path(__file__).resolve().parent / "out"
SETS = {
    "bb11": ("2nvzlh2k", FIX / "bb11_ground_truth.yaml"),
    "bb12": ("1fsnxchk", FIX / "bb12_ground_truth.yaml"),
}


def bar_index(t: float, grid: list[float]) -> float:
    """Fractional bar index of mix-time t."""
    i = min(max(bisect.bisect_right(grid, t) - 1, 0), len(grid) - 2)
    lo, hi = grid[i], grid[i + 1]
    return i + (t - lo) / (hi - lo)


def qhist(vals: list[float], quanta: tuple[int, ...] = (4, 8, 16, 32)) -> dict:
    """How close values sit to small multiples of common phrase lengths."""
    n = len(vals)
    out = {}
    for q in quanta:
        near = sum(
            1 for v in vals if abs(v / q - round(v / q)) * q <= 1.0 and v >= q / 2
        )
        out[f"within_1bar_of_mult_of_{q}"] = round(near / n, 2)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bed_lens: list[float] = []
    acaps_per_bed: Counter = Counter()
    acap_lens: list[float] = []
    entry_offsets: list[float] = []
    exit_offsets: list[float] = []  # bed_end - acap_end, bars
    coverage: list[float] = []
    seq_gaps: list[float] = []
    rounded_lens: Counter = Counter()

    for name, (set_id, path) in SETS.items():
        gt = load(path).value
        grid = json.loads(
            (REPO / f"data/analysis/{set_id}_measure_times.json").read_text()
        )
        usable = [t for t in gt.tracks if not t.unalignable]
        beds = [t for t in usable if t.claimed_stem in ("instrumental", "regular")]
        acaps = [t for t in usable if t.claimed_stem == "acappella"]

        # attach each acappella to ONE primary bed — the bed covering its
        # entry (nearest-entry overlapping bed as tiebreak) — so a vocal that
        # rides across a bed swap isn't counted once per bed it touches
        primary: dict[int, list[GroundTruthTrack]] = {id(b): [] for b in beds}
        for a in acaps:
            over = [
                b
                for b in beds
                if a.set_start_s < b.set_end_s and a.set_end_s > b.set_start_s
            ]
            if not over:
                continue
            best = min(
                over,
                key=lambda b: (
                    not (b.set_start_s <= a.set_start_s <= b.set_end_s),
                    abs(b.set_start_s - a.set_start_s),
                ),
            )
            primary[id(best)].append(a)

        for b in beds:
            b0, b1 = bar_index(b.set_start_s, grid), bar_index(b.set_end_s, grid)
            bed_lens.append(b1 - b0)
            riders = sorted(primary[id(b)], key=lambda a: a.set_start_s)
            acaps_per_bed[len(riders)] += 1
            covered = 0.0
            prev_end: float | None = None
            for a in riders:
                a0, a1 = bar_index(a.set_start_s, grid), bar_index(a.set_end_s, grid)
                acap_lens.append(a1 - a0)
                rounded_lens[round(a1 - a0)] += 1
                entry_offsets.append(a0 - b0)
                exit_offsets.append(b1 - a1)
                covered += max(0.0, min(a1, b1) - max(a0, b0))
                if prev_end is not None and a0 > prev_end:
                    seq_gaps.append(a0 - prev_end)
                prev_end = max(prev_end or a1, a1)
            if b1 > b0:
                coverage.append(min(1.0, covered / (b1 - b0)))

    def stats(vals: list[float]) -> str:
        s = sorted(vals)
        return (
            f"median={statistics.median(s):.1f}  p25={s[len(s) // 4]:.1f}  "
            f"p75={s[3 * len(s) // 4]:.1f}  n={len(s)}"
        )

    result = {
        "bed_len_bars": stats(bed_lens),
        "bed_len_quantization": qhist(bed_lens),
        "acaps_per_bed": dict(sorted(acaps_per_bed.items())),
        "acap_len_bars": stats(acap_lens),
        "acap_len_quantization": qhist(acap_lens),
        "acap_len_rounded_top": rounded_lens.most_common(10),
        "entry_offset_bars": stats(entry_offsets),
        "entry_offset_quantization": qhist([e for e in entry_offsets if e > 0]),
        "exit_offset_bars": stats(exit_offsets),
        "vocal_coverage_of_bed": stats(coverage),
        "gap_between_sequential_acaps_bars": stats(seq_gaps) if seq_gaps else "n=0",
    }
    (OUT / "arrangement_anatomy.json").write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
