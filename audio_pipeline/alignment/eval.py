"""Alignment evaluation harness.

Loads hand-annotated ground-truth YAMLs from `tests/fixtures/` and scores
the current `set_section_alignment` rows against them. Every algorithmic
change to the alignment pipeline should be gated on this score moving
up — without it, tuning is blind.

Three metrics:

* **mix_iou**         — intersection-over-union on the mix-side span per
                        matched ground-truth track. 1.0 = the reported
                        span exactly matches what a human annotated.
* **row_recall**      — fraction of ground-truth tracks matched to some
                        DB row (by set_start proximity + ref_track_id
                        or by excerpt substring). Misses indicate either
                        failed alignment or wrong GT↔row mapping.
* **span_inflation**  — mean of `reported_duration / gt_duration`. The
                        00:26 CCC runaway showed up as 3–20× here,
                        which is how we caught it in the first place.

Run as `python -m audio_pipeline.alignment.eval --db ... [--set-id ...]`
or import `evaluate(...)` programmatically.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from ..errors import DbError
from ..result import Err, Ok, Result


FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@dataclass(frozen=True)
class GroundTruthTrack:
    """One row of the hand-annotated yaml."""
    label: str                   # human-readable track name
    version_tag: str             # 'instrumental' | 'acappella' | 'full'
    set_start_s: float
    set_end_s: float
    ref_segments: tuple[tuple[float, float, float], ...]  # (ref_start, ref_end, mix_start)


@dataclass(frozen=True)
class GroundTruthSet:
    set_id: str
    tracks: tuple[GroundTruthTrack, ...]


@dataclass(frozen=True)
class DbAlignment:
    """One `set_section_alignment` row pruned to what eval needs."""
    section_idx: int
    set_start_s: float
    set_end_s: float
    ref_track_id: str | None
    row_text: str                # text_excerpt from dj_set_rows


@dataclass(frozen=True)
class MatchedPair:
    gt: GroundTruthTrack
    db: DbAlignment | None


@dataclass(frozen=True)
class EvalReport:
    set_id: str
    n_gt: int
    n_matched: int
    mix_iou_mean: float
    mix_iou_per_row: tuple[tuple[str, float], ...]
    span_inflation_mean: float
    per_row_detail: tuple[dict, ...]     # list of per-row diagnostic dicts

    def as_json(self) -> str:
        return json.dumps({
            "set_id": self.set_id,
            "n_gt": self.n_gt,
            "n_matched": self.n_matched,
            "mix_iou_mean": self.mix_iou_mean,
            "mix_iou_per_row": list(self.mix_iou_per_row),
            "span_inflation_mean": self.span_inflation_mean,
            "per_row_detail": list(self.per_row_detail),
        }, indent=2)


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def load_ground_truth(yaml_path: Path) -> Result[GroundTruthSet, str]:
    """Parse one ground-truth yaml.

    Tolerant of missing `ref_segments` (optional per the fixture spec).
    Returns Err on malformed yaml so the CLI can print a clean error.
    """
    try:
        payload = yaml.safe_load(yaml_path.read_text())
    except (yaml.YAMLError, OSError) as e:
        return Err(f"yaml load {yaml_path}: {e}")

    tracks: list[GroundTruthTrack] = []
    for t in payload.get("tracks", []):
        segs = t.get("ref_segments") or ()
        ref_segments = tuple(
            (float(s["ref_start_s"]), float(s["ref_end_s"]), float(s["mix_start_s"]))
            for s in segs
        )
        tracks.append(GroundTruthTrack(
            label=str(t.get("track", "")),
            version_tag=str(t.get("version_tag", "full")),
            set_start_s=float(t["set_start_s"]),
            set_end_s=float(t["set_end_s"]),
            ref_segments=ref_segments,
        ))
    return Ok(GroundTruthSet(
        set_id=str(payload["set_id"]),
        tracks=tuple(tracks),
    ))


def load_db_alignments(
    db_path: Path, set_id: str,
) -> Result[tuple[DbAlignment, ...], DbError]:
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT sa.section_idx, sa.set_start_s, sa.set_end_s,
                       sa.ref_track_id,
                       COALESCE(dsr.text_excerpt, '') AS text_excerpt
                FROM set_section_alignment sa
                LEFT JOIN dj_set_rows dsr
                  ON dsr.set_id = sa.set_id AND dsr.row_index = sa.section_idx
                WHERE sa.set_id = ?
                ORDER BY sa.section_idx
                """,
                (set_id,),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    return Ok(tuple(
        DbAlignment(
            section_idx=int(r["section_idx"]),
            set_start_s=float(r["set_start_s"]),
            set_end_s=float(r["set_end_s"]),
            ref_track_id=r["ref_track_id"],
            row_text=str(r["text_excerpt"]),
        )
        for r in rows
    ))


def _name_tokens(label: str) -> set[str]:
    """Extract lower-case word tokens ≥ 4 chars from a track label.

    Used as the substring-matching key between yaml `track` strings and
    DB `text_excerpt` strings. Shorter tokens ('of', 'the') cause too
    many false positives so we drop them.
    """
    import re
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z]{4,}", label)
        if t.lower() not in {"remix", "mix", "edit", "version"}
    }


def match_gt_to_db(
    gt: GroundTruthTrack, db_rows: tuple[DbAlignment, ...],
) -> DbAlignment | None:
    """Find the DB row that best corresponds to a GT track.

    Two-stage rank:
      1. Excerpt overlap on name tokens — highest score wins.
      2. Tie-break by proximity of `set_start_s` to GT `set_start_s`.

    Ignores rows whose `set_start_s` is more than 30s from GT — those
    are a different track with name-word coincidence.
    """
    tokens = _name_tokens(gt.label)
    best: DbAlignment | None = None
    best_score = (-1, float("inf"))   # (-overlap, |delta_start|)
    for row in db_rows:
        if abs(row.set_start_s - gt.set_start_s) > 30.0:
            continue
        row_tokens = _name_tokens(row.row_text)
        overlap = len(tokens & row_tokens)
        delta = abs(row.set_start_s - gt.set_start_s)
        key = (-overlap, delta)
        if key < best_score:
            best_score = key
            best = row
    return best


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 1e-6 else 0.0


def evaluate(
    db_path: Path, yaml_path: Path,
) -> Result[EvalReport, str]:
    gt_r = load_ground_truth(yaml_path)
    if not gt_r.is_ok():
        return gt_r
    gt = gt_r.value

    db_r = load_db_alignments(db_path, gt.set_id)
    if not db_r.is_ok():
        return Err(f"db: {db_r.error}")
    db_rows = db_r.value

    matched: list[MatchedPair] = []
    iou_entries: list[tuple[str, float]] = []
    inflations: list[float] = []
    details: list[dict] = []

    for gt_track in gt.tracks:
        db_row = match_gt_to_db(gt_track, db_rows)
        matched.append(MatchedPair(gt_track, db_row))
        if db_row is None:
            iou = 0.0
            inflation = float("nan")
            details.append({
                "label": gt_track.label,
                "gt_start": gt_track.set_start_s,
                "gt_end": gt_track.set_end_s,
                "db_start": None, "db_end": None,
                "iou": 0.0, "inflation": None, "matched": False,
            })
        else:
            iou = _iou(
                gt_track.set_start_s, gt_track.set_end_s,
                db_row.set_start_s, db_row.set_end_s,
            )
            gt_dur = gt_track.set_end_s - gt_track.set_start_s
            db_dur = db_row.set_end_s - db_row.set_start_s
            inflation = db_dur / gt_dur if gt_dur > 1e-6 else float("inf")
            inflations.append(inflation)
            details.append({
                "label": gt_track.label,
                "gt_start": gt_track.set_start_s,
                "gt_end": gt_track.set_end_s,
                "db_start": db_row.set_start_s,
                "db_end": db_row.set_end_s,
                "iou": iou,
                "inflation": inflation,
                "matched": True,
                "section_idx": db_row.section_idx,
            })
        iou_entries.append((gt_track.label, iou))

    n_matched = sum(1 for m in matched if m.db is not None)
    mean_iou = sum(v for _, v in iou_entries) / max(1, len(iou_entries))
    mean_inflation = (
        sum(inflations) / max(1, len(inflations)) if inflations else float("nan")
    )

    return Ok(EvalReport(
        set_id=gt.set_id,
        n_gt=len(gt.tracks),
        n_matched=n_matched,
        mix_iou_mean=mean_iou,
        mix_iou_per_row=tuple(iou_entries),
        span_inflation_mean=mean_inflation,
        per_row_detail=tuple(details),
    ))


def _cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/db/music_database.db")
    p.add_argument("--yaml", default=None,
                   help="Specific GT yaml to score. Defaults to every "
                        "*_ground_truth.yaml in tests/fixtures/.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    yamls: list[Path]
    if args.yaml:
        yamls = [Path(args.yaml)]
    else:
        yamls = sorted(FIXTURES_DIR.glob("*_ground_truth.yaml"))
    if not yamls:
        print(f"no ground-truth yamls found under {FIXTURES_DIR}", file=sys.stderr)
        return 2

    all_reports: list[EvalReport] = []
    for y in yamls:
        r = evaluate(db_path, y)
        if not r.is_ok():
            print(f"[eval] FAIL {y.name}: {r.error}", file=sys.stderr)
            continue
        all_reports.append(r.value)

    if args.json:
        print(json.dumps([json.loads(r.as_json()) for r in all_reports], indent=2))
        return 0

    # Human-readable summary.
    for rep in all_reports:
        print(f"\n=== {rep.set_id} ({Path(y).name if len(yamls)==1 else ''}) ===")
        print(f"  GT rows:          {rep.n_gt}")
        print(f"  matched to DB:    {rep.n_matched}/{rep.n_gt}")
        print(f"  mean mix IoU:     {rep.mix_iou_mean:.3f}")
        print(f"  mean inflation:   {rep.span_inflation_mean:.2f}x  (1.0 = perfect, >>1 = over-reported)")
        print(f"  per-row:")
        for d in rep.per_row_detail:
            if d["matched"]:
                print(f"    [{d['section_idx']:>3}] iou={d['iou']:.2f}  "
                      f"infl={d['inflation']:.1f}x  "
                      f"gt={d['gt_start']:.0f}-{d['gt_end']:.0f}s  "
                      f"db={d['db_start']:.0f}-{d['db_end']:.0f}s  "
                      f"{d['label'][:50]}")
            else:
                print(f"    [ -- ] iou=0.00  (no DB row matched)  "
                      f"gt={d['gt_start']:.0f}-{d['gt_end']:.0f}s  {d['label'][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
