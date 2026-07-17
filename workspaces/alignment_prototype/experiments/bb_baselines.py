#!/usr/bin/env python3
"""bb_baselines — placement head-to-head on the REAL BB mixes.

Compares full-mix matched-filter *baselines* (identity given) against our
drivers (classical, agentic) on the one metric they can share: **set_start
placement error**, scored through the SAME `score_timeline_vs_gt.score_spans`
convention (audible-onset, nearest same-recording GT row).

Design + provenance:
  docs/superpowers/specs/2026-07-17-bb-baseline-placement-comparison-design.md

Why no NMF / DTW: André's NMF and subsequence-DTW build matrices sized to the
mix length; they were built for UnmixDB *excerpts* and blow up on full-length
BB mixes (~3581 s, ~150 spans: DTW ≈ 12 GB/span, NMF ≈ 2.2 GB/track and loses
its joint-superposition purpose per-track). They optimize a different regime;
we state that rather than force a windowed number that changes the method.

What runs here (all `fftconvolve`, O(N log N) — fine on a full mix):
  - no_warp : chroma matched filter, locked stretch=1.0 (naive floor)
  - grid_mf : chroma matched filter with a tempo-stretch grid
Both are handed the CORRECT reference per span (identity given) — a setup that
FAVORS the baselines (our drivers must find identity themselves). Placement for
classical/agentic is read from their existing timelines over their
identity-correct spans.

    venvs/audio/bin/python -m workspaces.alignment_prototype.experiments.bb_baselines
    venvs/audio/bin/python -m workspaces.alignment_prototype.experiments.bb_baselines \
        --sets 2nvzlh2k --methods no_warp,grid_mf
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent.parent.parent  # experiments->…->repo
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.contracts import MANIFEST_FILENAME, load_manifest  # noqa: E402
from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    SR,
    STRETCHES,
    chroma,
    detect_offset,
    find_aligning_dir,
)
from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans  # noqa: E402

OUT_DIR = _REPO / "workspaces" / "alignment_prototype" / "out"
CACHE = Path(__file__).resolve().parent / "cache" / "bb_baselines"

# matched-filter baselines: name -> stretch grid handed to detect_offset.
# (no_warp == eval_bench.method_grid_locked; grid_mf == eval_bench.method_grid_mf.)
BASELINES: dict[str, tuple[float, ...]] = {
    "no_warp": (1.0,),
    "grid_mf": STRETCHES,
}
# our drivers, read from existing timelines: name -> out/ filename template
DRIVERS: dict[str, str] = {
    "classical": "{sid}_classical_timeline.json",
    "agentic": "{sid}_agentic_timeline.json",
}


# --------------------------------------------------------------------------- io
def _load_mono(path: Path) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def mix_chroma(set_id: str, set_dir: Path) -> np.ndarray:
    """Chroma of the FULL mix (mix.m4a), cached to .npy — computed once per set."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{set_id}_mix_chroma.npy"
    if cache.is_file():
        return np.load(cache)
    feat = chroma(_load_mono(set_dir / "mix.m4a"))
    np.save(cache, feat)
    return feat


def gt_rows(gt_path: Path) -> list[dict]:
    """GT track rows carrying a track_id (drop the 'mix' row and untracked rows)."""
    doc = yaml.safe_load(gt_path.read_text())
    return [
        r
        for r in doc["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")
    ]


def gt_fixture_for(set_id: str) -> Path:
    fixtures = sorted((_REPO / "labeling" / "fixtures").glob("*_ground_truth.yaml"))
    matches = [
        f for f in fixtures if yaml.safe_load(f.read_text()).get("set_id") == set_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} GT fixtures match set_id={set_id}")
    return matches[0]


# --------------------------------------------------------- matched-filter runner
def run_baseline(
    set_id: str,
    stretches: tuple[float, ...],
    mixc: np.ndarray,
    rows: list[dict],
    ref_audio: dict[str, Path],
    _chroma_cache: dict[str, np.ndarray],
) -> dict[str, float]:
    """Predicted set_start_s per GT slot: slide the CORRECT reference (identity
    given) over the full mix; detect_offset returns the mix position of the
    reference's frame 0. On real mid-song entries this lands ref_start earlier
    than the GT audible onset — that heavy tail IS the finding, not a bug."""
    preds: dict[str, float] = {}
    for r in rows:
        tid = str(r["track_id"])
        ap = ref_audio.get(tid)
        if ap is None:
            continue
        if tid not in _chroma_cache:
            disk = CACHE / f"ref_{tid}.npy"
            if disk.is_file():
                _chroma_cache[tid] = np.load(disk)
            else:
                CACHE.mkdir(parents=True, exist_ok=True)
                _chroma_cache[tid] = chroma(_load_mono(ap))
                np.save(disk, _chroma_cache[tid])
        rc = _chroma_cache[tid]
        if rc.shape[1] < 8 or mixc.shape[1] <= rc.shape[1]:
            continue
        pos, _peak, _st = detect_offset(rc, mixc, stretches)
        preds[str(r["slot_label"])] = float(pos)
    return preds


def baseline_timeline(set_id: str, rows: list[dict], preds: dict[str, float]) -> dict:
    """A load_timeline-valid timeline whose spans ARE the GT rows (identity given:
    recording_id = GT track_id) with set_start_s replaced by the baseline pred.
    Scored through score_spans exactly as a real driver timeline is."""
    spans = []
    for r in rows:
        slot = str(r["slot_label"])
        if slot not in preds:
            continue
        spans.append(
            {
                "slot_label": slot,
                "recording_id": str(r["track_id"]),
                "set_start_s": preds[slot],
                "set_end_s": float(r["set_end_s"]),
                "claimed_stem": r.get("claimed_stem") or "regular",
                # placement-only harness: a nominal straight ref span keeps
                # score_spans' trajectory helper from KeyError'ing; we read
                # only place_err_s (set_start vs GT audible onset).
                "ref_start_s": 0.0,
            }
        )
    return {"set_id": set_id, "spans": spans}


# ------------------------------------------------------------------- scoring/agg
def place_errs(scores) -> np.ndarray:
    """Finite per-span placement errors from a score_spans result."""
    return np.array(
        [s.place_err_s for s in scores if s.place_err_s is not None], dtype=float
    )


def summarize(errs: np.ndarray) -> dict:
    if errs.size == 0:
        return dict(n=0, mae=None, median=None, lt15_pct=None, p90=None)
    return dict(
        n=int(errs.size),
        mae=round(float(errs.mean()), 2),
        median=round(float(np.median(errs)), 2),
        lt15_pct=round(100.0 * float((errs < 15).mean()), 0),
        p90=round(float(np.percentile(errs, 90)), 1),
    )


def score_driver(set_id: str, template: str, gt_path: Path) -> dict:
    tl = OUT_DIR / template.format(sid=set_id)
    if not tl.is_file():
        return dict(
            n=0, mae=None, median=None, lt15_pct=None, p90=None, missing=str(tl)
        )
    return summarize(place_errs(score_spans(set_id, tl, gt_path=gt_path)))


def score_baseline(
    set_id: str, name: str, rows: list[dict], preds: dict[str, float], gt_path: Path
) -> dict:
    tl = baseline_timeline(set_id, rows, preds)
    with tempfile.NamedTemporaryFile(
        "w", suffix=f"_{set_id}_{name}.json", delete=False
    ) as fh:
        json.dump(tl, fh)
        tmp = Path(fh.name)
    try:
        return summarize(place_errs(score_spans(set_id, tmp, gt_path=gt_path)))
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- run
def run_set(set_id: str, methods: list[str]) -> dict[str, dict]:
    gt_path = gt_fixture_for(set_id)
    rows = gt_rows(gt_path)
    set_dir = find_aligning_dir(set_id)
    manifest = load_manifest(set_dir / MANIFEST_FILENAME)
    ref_audio = {
        k: Path(r.local_path)
        for k, r in manifest.by_track_id().items()
        if r.local_path and Path(r.local_path).is_file()
    }
    out: dict[str, dict] = {}

    baseline_methods = [m for m in methods if m in BASELINES]
    if baseline_methods:
        mixc = mix_chroma(set_id, set_dir)
        chroma_cache: dict[str, np.ndarray] = {}
        for name in baseline_methods:
            preds = run_baseline(
                set_id, BASELINES[name], mixc, rows, ref_audio, chroma_cache
            )
            out[name] = score_baseline(set_id, name, rows, preds, gt_path)

    for name in methods:
        if name in DRIVERS:
            out[name] = score_driver(set_id, DRIVERS[name], gt_path)
    return out


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", default="2nvzlh2k,1fsnxchk")
    p.add_argument(
        "--methods",
        default="no_warp,grid_mf,classical,agentic",
        help="ordered subset of no_warp,grid_mf,classical,agentic",
    )
    args = p.parse_args(argv)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    results: dict[str, dict] = {}
    for sid in sets:
        print(f"[{sid}] scoring {methods} …", file=sys.stderr)
        results[sid] = run_set(sid, methods)

    # table: method rows x (set: median / <15s / p90 / n)
    print("\nplacement set_start error — baselines (identity GIVEN) vs our drivers")
    print("NMF/DTW omitted: different regime (short synthetic excerpts), see spec.\n")
    hdr = f"{'method':<10}"
    for sid in sets:
        hdr += f" | {sid}: med / <15s / p90 / n"
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        line = f"{m:<10}"
        for sid in sets:
            r = results[sid].get(m, {})
            line += (
                f" | {_fmt(r.get('median'))} / {_fmt(r.get('lt15_pct'))}%"
                f" / {_fmt(r.get('p90'))} / {_fmt(r.get('n'))}"
            )
        print(line)
    print(
        "\nCaveats: n=2 real sets (no cross-set CI); baselines get correct identity"
        " for free (favors them); baselines assume ref_start=0 single-instance —"
        " the heavy tail is the real-mix difficulty, not a bug."
    )

    res_dir = Path(__file__).resolve().parent / "results"
    res_dir.mkdir(exist_ok=True)
    out_path = res_dir / "bb_baselines_placement.json"
    out_path.write_text(
        json.dumps({"sets": sets, "methods": methods, "results": results}, indent=2)
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
