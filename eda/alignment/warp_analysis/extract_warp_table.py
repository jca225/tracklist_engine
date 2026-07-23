"""Phase 0 warp-prior extractor.

Builds one flat table (one row per GT span, plus per-RefSegment sub-rows for
multiseg/loop warp) from the BB11 + BB12 ground-truth fixtures, and prints the
warp-magnitude distributions that de-risk test 2(a) (is warp low-rank / a
function of the BPM gap?).

Semantics recap (see labeling/als/semantics.py:tempo_ratio):
    tempo_ratio = ref_span / set_span = playback speed = mix_BPM / ref_BPM
so |tempo_ratio - 1| is the warp aggressiveness. A ratio > 1 means the DJ sped
the track up (mix faster than the track's native tempo); < 1 means slowed down.

No pi-storage needed — everything here is in the committed fixtures. The BPM
decomposition (does tempo_ratio == mix_BPM/ref_BPM independently measured?) is
the follow-up that pulls ref/mix BPM from track_measures/set_measures.

Run from repo root:
    venvs/audio/bin/python -m eda.alignment.warp_analysis.extract_warp_table
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from labeling.schema import GroundTruthTrack, load

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "labeling" / "fixtures"
OUT_DIR = Path(__file__).resolve().parent / "out"

SETS = {
    "bb11": FIXTURES / "bb11_ground_truth.yaml",
    "bb12": FIXTURES / "bb12_ground_truth.yaml",
}


def span_type(t: GroundTruthTrack) -> str:
    """straight | loop | multiseg — the same taxonomy the generator branches on."""
    if t.is_loop:
        return "loop"
    if len(t.ref_segments) > 1:
        return "multiseg"
    return "straight"


def ref_span_s(t: GroundTruthTrack) -> float | None:
    """Reference audio actually played (sum of segments for multiseg/loop)."""
    if t.ref_segments:
        return sum(s.ref_end_s - s.ref_start_s for s in t.ref_segments)
    if t.ref_end_s is not None:
        return t.ref_end_s - t.ref_start_s
    return None


def span_rows(set_name: str, gt) -> list[dict]:
    rows: list[dict] = []
    for t in gt.tracks:
        set_span = t.set_end_s - t.set_start_s
        rspan = ref_span_s(t)
        # tempo_ratio as recorded, plus a recomputed check from spans
        recomputed = (rspan / set_span) if (rspan and set_span > 0) else None
        rows.append(
            {
                "set": set_name,
                "slot": t.slot_label,
                "label": t.label,
                "stem": t.claimed_stem,
                "span_type": span_type(t),
                "ref_source": t.ref_source,
                "tempo_ratio": t.tempo_ratio,
                "tempo_recomputed": recomputed,
                "pitch_shift_semi": t.pitch_shift_semi,
                "n_segments": len(t.ref_segments),
                "set_span_s": round(set_span, 3),
                "ref_span_s": round(rspan, 3) if rspan is not None else None,
                "audible_frac": t.audible_frac,
                "unalignable": t.unalignable,
                "skip_training": t.skip_training,
            }
        )
    return rows


def seg_rows(set_name: str, gt) -> list[dict]:
    """Per-segment warp for multiseg/loop spans — each segment's own stretch."""
    rows: list[dict] = []
    for t in gt.tracks:
        segs = t.ref_segments
        if len(segs) < 2:
            continue
        ordered = sorted(segs, key=lambda s: s.mix_start_s)
        for i, s in enumerate(ordered):
            mix_end = (
                ordered[i + 1].mix_start_s if i + 1 < len(ordered) else t.set_end_s
            )
            mix_dur = mix_end - s.mix_start_s
            ref_dur = s.ref_end_s - s.ref_start_s
            rows.append(
                {
                    "set": set_name,
                    "slot": t.slot_label,
                    "stem": t.claimed_stem,
                    "span_type": span_type(t),
                    "seg_idx": i,
                    "ref_dur_s": round(ref_dur, 3),
                    "mix_dur_s": round(mix_dur, 3),
                    "seg_tempo": round(ref_dur / mix_dur, 5) if mix_dur > 0 else None,
                    "ref_jump_s": round(s.ref_start_s - ordered[i - 1].ref_end_s, 3)
                    if i > 0
                    else None,
                }
            )
    return rows


def _summ(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  (no data)"
    vals_sorted = sorted(vals)
    n = len(vals)
    med = statistics.median(vals)
    p10 = vals_sorted[int(0.10 * (n - 1))]
    p90 = vals_sorted[int(0.90 * (n - 1))]
    return (
        f"  n={n:3d}  min={min(vals):.4f}  p10={p10:.4f}  "
        f"median={med:.4f}  p90={p90:.4f}  max={max(vals):.4f}"
    )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    all_spans: list[dict] = []
    all_segs: list[dict] = []
    for name, path in SETS.items():
        r = load(path)
        if not r.is_ok():
            raise SystemExit(f"failed to load {path}: {r.error}")
        all_spans.extend(span_rows(name, r.value))
        all_segs.extend(seg_rows(name, r.value))

    # write CSVs
    if all_spans:
        with (OUT_DIR / "warp_spans.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_spans[0].keys()))
            w.writeheader()
            w.writerows(all_spans)
    if all_segs:
        with (OUT_DIR / "warp_segments.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_segs[0].keys()))
            w.writeheader()
            w.writerows(all_segs)

    # ---- console report ----
    n = len(all_spans)
    with_tempo = [r for r in all_spans if r["tempo_ratio"] is not None]
    exact_one = [r for r in with_tempo if abs(r["tempo_ratio"] - 1.0) < 1e-9]
    print(f"\n=== Phase 0 warp table — {n} spans, {len(all_segs)} segments ===")
    print(f"spans with tempo_ratio: {len(with_tempo)}/{n}")
    print(
        f"tempo_ratio EXACTLY 1.0: {len(exact_one)}  "
        f"(suspect default vs genuine no-warp — needs BPM check)"
    )

    print("\n-- tempo_ratio, all spans --")
    print(_summ([r["tempo_ratio"] for r in with_tempo]))

    print("\n-- |tempo_ratio - 1| (warp aggressiveness), non-exact-1 spans --")
    warped = [r for r in with_tempo if abs(r["tempo_ratio"] - 1.0) >= 1e-9]
    print(_summ([abs(r["tempo_ratio"] - 1.0) for r in warped]))
    print(f"  ({len(warped)} spans carry a non-trivial warp)")

    for axis in ("stem", "span_type"):
        print(f"\n-- tempo_ratio by {axis} --")
        keys = sorted({r[axis] for r in with_tempo})
        for k in keys:
            grp = [r["tempo_ratio"] for r in with_tempo if r[axis] == k]
            ex1 = sum(1 for v in grp if abs(v - 1.0) < 1e-9)
            print(f"  {k:12s} (exact-1: {ex1:2d})")
            print("   " + _summ(grp))

    print("\n-- pitch_shift_semi distribution --")
    pitches = [r["pitch_shift_semi"] for r in all_spans]
    from collections import Counter

    for semi, c in sorted(Counter(pitches).items()):
        print(f"  {semi:+d} semi: {c}")

    print("\n-- per-segment warp (multiseg/loop) --")
    print(_summ([r["seg_tempo"] for r in all_segs]))

    print(f"\nwrote: {OUT_DIR / 'warp_spans.csv'}")
    print(f"wrote: {OUT_DIR / 'warp_segments.csv'}")


if __name__ == "__main__":
    main()
