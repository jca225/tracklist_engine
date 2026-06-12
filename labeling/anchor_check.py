"""Compare saved ground-truth YAML against a fresh `.als` re-export (P1 verify).

Use this offline on the Mac — no pi-storage required. Prints a short checklist
for manual Ableton playhead checks plus programmatic drift vs re-parsed `.als`.

Usage:
    venvs/audio/bin/python -m labeling.anchor_check \\
        --yaml labeling/fixtures/bb12_ground_truth.yaml

    venvs/audio/bin/python -m labeling.anchor_check --anchors 002,003,120,154
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.export_als_to_gt import DEFAULT_ALS, DEFAULT_SET_DIR, export_gt
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load
from core.result import Err, Ok

DEFAULT_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
DEFAULT_ANCHORS = ("002", "003:instrumental", "120", "154")
BEAT_TOL = 1.0          # plan: ±~1 beat
DEFAULT_BPM = 128.0     # display-only beat conversion for tolerance line


@dataclass(frozen=True)
class AnchorResult:
    slot: str
    stem: str
    label: str
    yaml_start: float
    yaml_end: float
    fresh_start: float | None
    fresh_end: float | None
    delta_start_s: float | None
    delta_end_s: float | None

    @property
    def ok(self) -> bool:
        if self.delta_start_s is None:
            return False
        beat_s = 60.0 / DEFAULT_BPM
        return abs(self.delta_start_s) <= BEAT_TOL * beat_s and abs(self.delta_end_s or 0) <= BEAT_TOL * beat_s


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:06.3f}"


def _norm_slot(slot: str) -> str:
    s = slot.strip()
    if s.isdigit():
        return str(int(s)).zfill(3) if len(s) <= 3 else s
    if "w" in s:
        base, _, suffix = s.partition("w")
        if base.isdigit():
            return f"{int(base):03d}w{suffix}"
    return s


def _track_key(t: GroundTruthTrack) -> tuple[str, str, float, float]:
    stem = t.claimed_stem or "regular"
    slot = _norm_slot(t.slot_label or t.label)
    return slot, stem, round(t.set_start_s, 3), round(t.set_end_s, 3)


def _index_gt(gt: GroundTruthSet) -> dict[tuple[str, str, float, float], GroundTruthTrack]:
    out: dict[tuple[str, str, float, float], GroundTruthTrack] = {}
    for t in gt.tracks:
        out[_track_key(t)] = t
    return out


def _find_fresh(
    t: GroundTruthTrack,
    fresh_index: dict[tuple[str, str, float, float], GroundTruthTrack],
) -> GroundTruthTrack | None:
    key = _track_key(t)
    if key in fresh_index:
        return fresh_index[key]
    slot = _norm_slot(t.slot_label or t.label)
    stem = t.claimed_stem or "regular"
    candidates = [
        f for k, f in fresh_index.items()
        if k[0] == slot and k[1] == stem
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs(f.set_start_s - t.set_start_s))


def compare_anchors(
    yaml_gt: GroundTruthSet,
    fresh_gt: GroundTruthSet,
    anchor_slots: tuple[str, ...],
) -> list[AnchorResult]:
    fresh_index = _index_gt(fresh_gt)
    by_slot: dict[str, list[GroundTruthTrack]] = {}
    for t in yaml_gt.tracks:
        if t.slot_label:
            by_slot.setdefault(_norm_slot(t.slot_label), []).append(t)
    results: list[AnchorResult] = []
    for spec in anchor_slots:
        stem_filter = ""
        if ":" in spec:
            slot, stem_filter = spec.split(":", 1)
        else:
            slot = spec
        ns = _norm_slot(slot)
        candidates = by_slot.get(ns, [])
        if stem_filter:
            candidates = [t for t in candidates if (t.claimed_stem or "regular") == stem_filter]
        t = min(candidates, key=lambda x: x.set_start_s) if candidates else None
        if t is None:
            results.append(AnchorResult(
                slot=slot, stem="?", label="(missing from yaml)",
                yaml_start=0, yaml_end=0,
                fresh_start=None, fresh_end=None,
                delta_start_s=None, delta_end_s=None,
            ))
            continue
        fresh = _find_fresh(t, fresh_index)
        if fresh is None:
            results.append(AnchorResult(
                slot=slot, stem=t.claimed_stem, label=t.label,
                yaml_start=t.set_start_s, yaml_end=t.set_end_s,
                fresh_start=None, fresh_end=None,
                delta_start_s=None, delta_end_s=None,
            ))
            continue
        results.append(AnchorResult(
            slot=slot, stem=t.claimed_stem, label=t.label,
            yaml_start=t.set_start_s, yaml_end=t.set_end_s,
            fresh_start=fresh.set_start_s, fresh_end=fresh.set_end_s,
            delta_start_s=fresh.set_start_s - t.set_start_s,
            delta_end_s=fresh.set_end_s - t.set_end_s,
        ))
    return results


def summarize_drift(yaml_gt: GroundTruthSet, fresh_gt: GroundTruthSet) -> tuple[int, int, float, float]:
    fresh_index = _index_gt(fresh_gt)
    matched = 0
    max_start_delta = 0.0
    max_ref_delta = 0.0
    for t in yaml_gt.tracks:
        fresh = _find_fresh(t, fresh_index)
        if fresh is None:
            continue
        matched += 1
        max_start_delta = max(max_start_delta, abs(fresh.set_start_s - t.set_start_s))
        max_ref_delta = max(max_ref_delta, abs(fresh.ref_start_s - t.ref_start_s))
    return matched, len(yaml_gt.tracks), max_start_delta, max_ref_delta


def summarize_audible(gt: GroundTruthSet) -> tuple[int, int]:
    """Return (muted, partial) counts."""
    muted = partial = 0
    for t in gt.tracks:
        if t.skip_training:
            muted += 1
        elif t.audible_frac is not None and t.audible_frac < 1.0:
            partial += 1
    return muted, partial


def print_report(
    results: list[AnchorResult],
    *,
    matched: int,
    total: int,
    max_start_delta: float,
    max_ref_delta: float,
    muted: int,
    partial: int,
) -> None:
    beat_s = 60.0 / DEFAULT_BPM
    tol_s = BEAT_TOL * beat_s
    print(f"corpus drift: matched {matched}/{total} yaml rows; "
          f"max |Δstart|={max_start_delta:.3f}s; max |Δref_start|={max_ref_delta:.3f}s")
    print(f"tolerance line: ±{BEAT_TOL} beat @ {DEFAULT_BPM:.0f}bpm ≈ ±{tol_s:.3f}s")
    print(f"audible metadata: muted={muted} partial={partial}")
    print()
    print("Manual Ableton check — park playhead on 1-mix at mix time, compare to YAML:")
    print(f"{'slot':6} {'stem':12} {'yaml mix span':24} {'Δstart':8} {'ok':3}  track")
    for r in results:
        span = f"{_fmt_time(r.yaml_start)} – {_fmt_time(r.yaml_end)}"
        delta = f"{r.delta_start_s:+.3f}s" if r.delta_start_s is not None else "MISSING"
        ok = "yes" if r.ok else "NO"
        print(f"{r.slot:6} {r.stem:12} {span:24} {delta:8} {ok:3}  {r.label[:45]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    p.add_argument("--als", type=Path, default=DEFAULT_ALS)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    p.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS),
                   help="Comma-separated slot labels; optional slot:stem (e.g. 003:instrumental)")
    p.add_argument("--strict-ref", action="store_true",
                   help="exit non-zero when ref_start drift exceeds tolerance")
    args = p.parse_args(argv)

    match load(args.yaml):
        case Err(e):
            print(f"yaml: {e.detail}", file=sys.stderr)
            return 1
        case Ok(yaml_gt):
            pass

    try:
        fresh_gt, _ = export_gt(args.als, args.set_dir)
    except (OSError, ValueError) as e:
        print(f"als re-export failed: {e}", file=sys.stderr)
        return 1

    slots = tuple(s.strip() for s in args.anchors.split(",") if s.strip())
    results = compare_anchors(yaml_gt, fresh_gt, slots)
    matched, total, max_start_delta, max_ref_delta = summarize_drift(yaml_gt, fresh_gt)
    muted, partial = summarize_audible(yaml_gt)
    print_report(
        results,
        matched=matched,
        total=total,
        max_start_delta=max_start_delta,
        max_ref_delta=max_ref_delta,
        muted=muted,
        partial=partial,
    )

    if not all(r.ok for r in results):
        return 1
    tol_s = BEAT_TOL * (60.0 / DEFAULT_BPM)
    if max_start_delta > tol_s:
        print("\nnote: full-corpus drift is higher (loops/merges); anchor spots are OK.", file=sys.stderr)
    if max_ref_delta > tol_s:
        print(f"\nwarning: ref_start drift up to {max_ref_delta:.3f}s — re-export GT yaml recommended.",
              file=sys.stderr)
        if args.strict_ref:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
