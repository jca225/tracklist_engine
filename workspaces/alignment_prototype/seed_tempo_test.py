#!/usr/bin/env python3
"""Stage 0 crash-test: write a real tempo envelope onto the template MasterTrack.

The seeder uses a fixed 60 BPM ("1 beat = 1 s"); the goal is faithful tempo
automation reflecting the set's actual BPM. Before rebuilding the clip-placement
math (seconds->beats integration), this proves the riskiest assumption in
ISOLATION: does Ableton open a .als with a populated MasterTrack tempo envelope
without crashing? (The round-trip self-test cannot catch a Live crash — only
opening it can. See the als-seed strip_automation history.)

It reuses the template's existing tempo AutomationTarget/envelope, so no new
PointeeId is allocated. Clips are left untouched — expect them misplaced; this
is purely a crash + "tempo visibly varies" check.

Tempo source: the mix beat grid (measure downbeats) -> per-measure BPM.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.seed_tempo_test \
        --measures data/analysis/1fsnxchk_measure_times.json \
        [--beats-per-measure 4] [--out ~/Desktop/bb12_tempo_test.als]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

from lxml import etree

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als_io import load_als_xml, write_locators, write_tempo_envelope  # noqa: E402

DEFAULT_TEMPLATE = (
    Path.home()
    / "Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als"
)
ALIGNING_ROOT = Path.home() / "aligning"
BPM_MIN, BPM_MAX = 80.0, 180.0


def measure_times(path: Path) -> list[float]:
    d = json.loads(path.read_text())
    xs = (
        d
        if isinstance(d, list)
        else (d.get("measure_times") or d.get("beats") or next(iter(d.values())))
    )
    return [float(x) for x in xs]


def _fold_octave(bpm: float, ref: float) -> float:
    """Collapse half/double-time beat-tracker octave errors toward `ref`."""
    while bpm > ref * 1.45:
        bpm /= 2.0
    while bpm < ref * 0.69:
        bpm *= 2.0
    return bpm


def grid_bpm(
    times: list[float], beats_per_measure: int
) -> tuple[list[float], list[float]]:
    """Per-measure (time_s, octave-folded bpm). Beat axis = seconds (file is 60
    BPM baseline, so 1 beat == 1 s — keeps the tempo curve aligned to clips)."""
    import numpy as np

    ts, raw = [], []
    for i in range(len(times) - 1):
        dur = times[i + 1] - times[i]
        if dur > 0:
            ts.append(times[i])
            raw.append(beats_per_measure * 60.0 / dur)
    ref = float(np.median(np.clip(raw, BPM_MIN, BPM_MAX)))
    folded = [min(BPM_MAX, max(BPM_MIN, _fold_octave(b, ref))) for b in raw]
    return ts, folded


def gt_boundaries(gt_path: Path) -> list[float]:
    """Track-transition times (set_start_s of every GT span) — where tempo may
    actually change. These replace change-point detection on the noisy grid."""
    import yaml

    rows = yaml.safe_load(gt_path.read_text())["tracks"]
    bs = {
        round(float(r["set_start_s"]), 3)
        for r in rows
        if r.get("track_id") and r.get("set_start_s") is not None
    }
    return sorted(bs)


_SLOT_RE = re.compile(r"^(\d{3}(?:w\d+)?)__")
_BPM_RE = re.compile(r"\[(\d+)\s*bpm", re.IGNORECASE)


def slot_bpm_map(tracks_dir: Path) -> dict[str, float]:
    """slot_label -> native BPM from the annotator's `[NNNbpm KK]` filename tags.
    Prefers a non-acappella file's tag for a slot (the bed carries tempo)."""
    by_slot: dict[str, tuple[float, bool]] = {}
    for f in tracks_dir.iterdir():
        ms, mb = _SLOT_RE.match(f.name), _BPM_RE.search(f.name)
        if not (ms and mb):
            continue
        slot, bpm = ms.group(1), float(mb.group(1))
        is_acap = "acappella" in f.name.lower()
        prev = by_slot.get(slot)
        if prev is None or (prev[1] and not is_acap):  # upgrade acap-tag -> bed-tag
            by_slot[slot] = (bpm, is_acap)
    return {k: v[0] for k, v in by_slot.items()}


def track_segments(
    gt_path: Path,
    slot_bpm: dict[str, float],
    times: list[float],
    *,
    beats_per_measure: int,
    round_bpm: float,
) -> list[tuple[float, float, float]]:
    """Piecewise-constant tempo from the bed spans: BPM = tagged_native × tempo_ratio
    (exact, no grid octave errors), held from each bed span's start to the next.
    Acappellas are layered on the bed and carry no tempo, so they're excluded.
    Untagged slots fall back to the octave-folded grid median over the span."""
    import numpy as np
    import yaml

    rows = [
        r
        for r in yaml.safe_load(gt_path.read_text())["tracks"]
        if r.get("track_id") and r.get("set_start_s") is not None
    ]
    bed = sorted(
        (r for r in rows if (r.get("claimed_stem") or "regular") != "acappella"),
        key=lambda r: float(r["set_start_s"]),
    )
    ts, bp = grid_bpm(times, beats_per_measure)
    ts_a, bp_a = np.array(ts), np.array(bp)
    natives = [
        slot_bpm[str(r.get("slot_label"))]
        for r in bed
        if str(r.get("slot_label")) in slot_bpm
    ]
    ref = float(np.median(natives)) if natives else 128.0

    raw: list[tuple[float, float]] = []
    for r in bed:
        ratio = float(r.get("tempo_ratio") or 1.0)
        native = slot_bpm.get(str(r.get("slot_label")))
        if native is not None:
            val = native * ratio
        else:  # untagged: robust grid median inside the span
            lo, hi = float(r["set_start_s"]), float(r["set_end_s"])
            m = bp_a[(ts_a >= lo) & (ts_a < hi)]
            val = float(np.median(m)) if m.size else float(np.median(bp_a))
        # one-directional octave guard: halve double-time tag errors; never
        # double a low value (genuine half-time sections must survive).
        while val > ref * 1.35:
            val /= 2.0
        raw.append((float(r["set_start_s"]), round(val / round_bpm) * round_bpm))

    segs: list[list] = []
    for i, (st, val) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else times[-1]
        if end <= st:
            continue
        if segs and segs[-1][2] == val:
            segs[-1][1] = end
        else:
            segs.append([st, end, val])
    return [(float(s), float(e), float(v)) for s, e, v in segs]


def tempo_segments(
    times: list[float],
    *,
    beats_per_measure: int,
    round_bpm: float,
    boundaries: list[float] | None,
) -> list[tuple[float, float, float]]:
    """Piecewise-constant (start_s, end_s, bpm). Boundaries come from track
    transitions; each segment's BPM is the median folded-grid BPM inside it
    (robust to per-measure jitter). Adjacent equal segments are coalesced so a
    step appears only where the tempo genuinely changes."""
    import numpy as np

    ts, bp = grid_bpm(times, beats_per_measure)
    ts_a, bp_a = np.array(ts), np.array(bp)
    if boundaries is None:
        boundaries = list(ts_a)  # degenerate: every measure (auto fallback)
    edges = sorted({ts_a[0], *[b for b in boundaries if ts_a[0] < b < ts_a[-1]]})
    edges = edges + [ts_a[-1] + 1.0]
    segs: list[list] = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = bp_a[(ts_a >= lo) & (ts_a < hi)]
        if m.size == 0:
            continue
        val = round(float(np.median(m)) / round_bpm) * round_bpm
        if segs and segs[-1][2] == val:
            segs[-1][1] = hi  # coalesce equal neighbour
        else:
            segs.append([lo, hi, val])
    return [(float(s), float(e), float(v)) for s, e, v in segs]


def stepped_points(segs: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Each segment emits its value at the start and a hold point just before its
    end, so Ableton's linear interpolation renders a flat shelf then a sharp step.
    Beat axis = seconds (60 BPM baseline)."""
    eps = 0.01
    pts: list[tuple[float, float]] = []
    for s, e, bpm in segs:
        pts.append((s, bpm))
        pts.append((e - eps, bpm))
    return pts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--measures",
        type=Path,
        default=_REPO / "data/analysis/1fsnxchk_measure_times.json",
    )
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--beats-per-measure", type=int, default=4)
    p.add_argument(
        "--round-bpm",
        type=float,
        default=1.0,
        help="quantize each segment's BPM to this grid",
    )
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument(
        "--no-gt",
        action="store_true",
        help="ignore GT+track tags; derive boundaries from the grid (noisier)",
    )
    p.add_argument(
        "--out", type=Path, default=Path.home() / "Desktop/bb12_tempo_test.als"
    )
    p.add_argument(
        "--no-markers",
        action="store_true",
        help="skip writing arrangement locators at each tempo change",
    )
    args = p.parse_args(argv)

    if not args.template.is_file():
        sys.exit(f"template not found: {args.template}")
    times = measure_times(args.measures)

    if args.no_gt:
        segs = tempo_segments(
            times,
            beats_per_measure=args.beats_per_measure,
            round_bpm=args.round_bpm,
            boundaries=None,
        )
        src = "auto-grid"
    else:
        aligning = sorted(ALIGNING_ROOT.glob(f"{args.set_id}__*"))
        slot_bpm = slot_bpm_map(aligning[0] / "tracks") if aligning else {}
        segs = track_segments(
            args.gt,
            slot_bpm,
            times,
            beats_per_measure=args.beats_per_measure,
            round_bpm=args.round_bpm,
        )
        src = f"{len(slot_bpm)} tagged tracks x tempo_ratio, bed-span boundaries"
    pts = stepped_points(segs)
    if not pts:
        sys.exit("no tempo breakpoints derived")

    root = load_als_xml(args.template)
    n = write_tempo_envelope(root, pts)
    n_loc = 0
    if not args.no_markers:
        # one marker at each tempo-segment start, labelled with its BPM (same
        # beat axis as the envelope). Markers are arrangement-global, schema-safe.
        n_loc = write_locators(root, [(s, f"{bpm:.0f} BPM") for (s, e, bpm) in segs])
    args.out.write_bytes(
        gzip.compress(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    )
    bpms = [v for _, _, v in segs]
    print(
        f"wrote {len(segs)} tempo segments ({n} points) + {n_loc} markers -> {args.out}"
    )
    print(
        f"  source: {src}; bpm range {min(bpms):.0f}-{max(bpms):.0f}, "
        f"median {sorted(bpms)[len(bpms) // 2]:.0f}"
    )
    print("  ladder (start_s -> bpm):")
    for s, e, bpm in segs:
        print(f"    {s:7.1f}s  {bpm:5.0f} bpm  ({e - s:5.0f}s)")
    print("OPEN IN ABLETON: flat shelves, sharp steps only at song changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
