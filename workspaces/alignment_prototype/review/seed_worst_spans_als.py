#!/usr/bin/env python3
"""Worst-spans A/B Ableton session — PRED vs GT, the fiber-overlay pattern.

For the top-N spans by GT-seconds lost, render one Live set per DJ set:

  * "1-mix (ruler)"        — the mix, unwarped at 60 BPM (beat == second)
  * "PRED r<k> <slot> …"   — what the aligner claims played (red): one warped
                             clip per predicted ref_segment, the aligner's
                             recording (which may be an identity miss)
  * "GT   r<k> <slot> …"   — what actually played (green), from hand GT

Solo mix+PRED to hear the machine's claim, mix+GT for truth. Verdict
protocol (optional, parsed later): prepend BUG / HARD / GTW to a pair's
PRED track name. Ranking source: failure_analysis span_table.csv
(sec_lost = gt_duration x (1 - traj_strict)).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.review.seed_worst_spans_als \
        --set-id 1fsnxchk [--top 15] [--out <als>]
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import itertools
import json
import sys
from datetime import date
from pathlib import Path

from lxml import etree

from labeling.als import load_als_xml, parse_layer_clips
from workspaces.alignment_prototype.review.seed_als_from_timeline import (
    _SET_DISPLAY,
    _TEMPO_BPM,
    DEFAULT_TEMPLATE,
    OUT_DIR,
    _clip_specs,
    build_track,
    doc_max_id,
    ffprobe_audio,
    find_template_track,
    pick_audio,
    renumber_pointee_ids,
)

# repo root, derived from the prototype-level out/ (OUT_DIR = <repo>/
# workspaces/alignment_prototype/out — see seed_als_from_timeline)
_REPO = OUT_DIR.parent.parent.parent
SPAN_TABLE = _REPO / "eda/alignment/failure_analysis/out/span_table.csv"
ALIGNING_ROOT = Path.home() / "aligning"
OUT_ROOT = ALIGNING_ROOT / "_review"
_COLOR_RED, _COLOR_GREEN, _COLOR_BLUE = 14, 5, 9


def worst_slots(set_id: str, top: int) -> list[dict]:
    rows = [
        r
        for r in csv.DictReader(SPAN_TABLE.open())
        if r["set_id"] == set_id and r.get("gt_duration_s")
    ]
    for r in rows:
        r["sec_lost"] = float(r["gt_duration_s"]) * (
            1.0 - float(r.get("traj_strict") or 0.0)
        )
    rows.sort(key=lambda r: -r["sec_lost"])
    return rows[:top]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    ranked = worst_slots(args.set_id, args.top)
    if not ranked:
        sys.exit(f"no rows for {args.set_id} in {SPAN_TABLE} — run make scorecard")

    timeline = json.loads(
        (OUT_DIR / f"{args.set_id}_predicted_timeline.json").read_text()
    )
    pred_by_slot = {str(s.get("slot_label")): s for s in timeline["spans"]}

    set_dirs = sorted(ALIGNING_ROOT.glob(f"{args.set_id}__*"))
    if not set_dirs:
        sys.exit(f"no ~/aligning folder for {args.set_id}")
    set_dir = set_dirs[0]
    from core.contracts.manifest import MANIFEST_FILENAME, load_manifest

    manifest = load_manifest(set_dir / MANIFEST_FILENAME)
    # pick_audio (seed_als_from_timeline) consumes dict-shaped rows
    by_tid = {
        r.track_id: {"local_path": r.local_path, "stems": dict(r.stems)}
        for r in manifest.tracks
    }
    mix_path = Path(manifest.mix_local_path)
    mix_dur, mix_sr = ffprobe_audio(mix_path)

    gt_fix = _REPO / "labeling/fixtures"
    gt_path = next(
        (
            q
            for q in (
                gt_fix / f"bb12_ground_truth.yaml",
                gt_fix / f"bb11_ground_truth.yaml",
            )
            if q.is_file() and args.set_id in q.read_text()[:4000]
        ),
        None,
    )
    import yaml

    if gt_path is None:
        for q in gt_fix.glob("*_ground_truth.yaml"):
            if yaml.safe_load(q.read_text()).get("set_id") == args.set_id:
                gt_path = q
                break
    if gt_path is None:
        sys.exit(f"no GT fixture for {args.set_id}")
    gt_doc = yaml.safe_load(gt_path.read_text())
    gt_by_slot = {
        str(r.get("slot_label")): r
        for r in gt_doc["tracks"]
        if str(r.get("slot_label")) != "mix"
    }

    root = load_als_xml(args.template)
    template_track = copy.deepcopy(find_template_track(root))
    tracks_node = root.find(".//LiveSet/Tracks")
    for t in list(tracks_node):
        if t.tag in ("AudioTrack", "GroupTrack", "MidiTrack"):
            tracks_node.remove(t)
    try:
        from labeling.als import write_tempo_envelope

        write_tempo_envelope(root, [(0.0, _TEMPO_BPM)])
    except Exception:
        for tm in root.xpath(".//MasterTrack//Tempo/Manual"):
            tm.set("Value", f"{_TEMPO_BPM:.1f}")

    alloc = itertools.count(doc_max_id(root) + args.top * 8 + 4000)
    next_id = 1000
    ruler = build_track(
        template_track,
        track_id=next_id,
        track_name="1-mix (ruler)",
        name="mix",
        color=_COLOR_BLUE,
        arr_start=0.0,
        arr_end=mix_dur,
        file_path=mix_path,
        file_dur_s=mix_dur,
        sample_rate=mix_sr,
        ref_start_s=0.0,
        ref_end_s=mix_dur,
        is_warped=False,
    )
    renumber_pointee_ids(ruler, alloc)
    tracks_node.insert(0, ruler)

    placed, skipped, insert_at = 0, [], 1

    def _add(span: dict, track: dict | None, tname: str, color: int) -> int:
        nonlocal next_id, insert_at, placed
        fpath = pick_audio(span, track) if track is not None else None
        if fpath is None:
            return 0
        fdur, fsr = ffprobe_audio(fpath)
        n = 0
        for j, (a0, a1, rs, re_) in enumerate(_clip_specs(span)):
            ref_start = max(0.0, min(rs, fdur - 1.0))
            ref_end = max(ref_start + 1.0, min(re_, fdur))
            seg = f" seg{j + 1}" if len(_clip_specs(span)) > 1 else ""
            next_id += 1
            tr = build_track(
                template_track,
                track_id=next_id,
                track_name=tname,
                name=f"{tname}{seg}",
                color=color,
                arr_start=a0,
                arr_end=max(a0 + 1.0, a1),
                file_path=fpath,
                file_dur_s=fdur,
                sample_rate=fsr,
                ref_start_s=ref_start,
                ref_end_s=ref_end,
            )
            renumber_pointee_ids(tr, alloc)
            tracks_node.insert(insert_at, tr)
            insert_at += 1
            placed += 1
            n += 1
        return n

    for k, row in enumerate(ranked, 1):
        slot = str(row["slot"])
        gt = gt_by_slot.get(slot)
        pred = pred_by_slot.get(slot)
        nm = (row.get("name") or "")[:34]
        lost = f"{row['sec_lost']:.0f}s"
        if pred is not None:
            n = _add(
                pred,
                by_tid.get(pred.get("recording_id")),
                f"PRED r{k} {slot} [{lost}] {nm}",
                _COLOR_RED,
            )
            if not n:
                skipped.append(f"PRED {slot}")
        if gt is not None:
            g = dict(gt)
            g.setdefault("claimed_stem", row.get("gt_stem"))
            n = _add(
                g,
                by_tid.get(str(gt.get("track_id"))),
                f"GT   r{k} {slot} {nm}",
                _COLOR_GREEN,
            )
            if not n:
                skipped.append(f"GT {slot}")

    try:
        from labeling.als import write_locators

        write_locators(
            root,
            [
                (
                    0.0,
                    f"WORST-SPANS SEED {date.today().isoformat()} — "
                    "PRED red vs GT green, NOT a labeling session",
                )
            ]
            + [
                (
                    float(r["gt_set_start_s"]),
                    f"r{k} {r['slot']} lost {r['sec_lost']:.0f}s",
                )
                for k, r in enumerate(ranked, 1)
                if r.get("gt_set_start_s")
            ],
        )
    except Exception as exc:
        print(f"  locators skipped: {exc}", file=sys.stderr)

    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))

    display = _SET_DISPLAY.get(args.set_id, args.set_id)
    out_path = args.out or (OUT_ROOT / f"{display} WORST SPANS SEEDED.als")
    if out_path.name.endswith(" align.als"):
        sys.exit("refusing hand-GT name")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        bak = out_path.with_suffix(".als.seedbak")
        out_path.rename(bak)
        print(f"backed up existing -> {bak.name}")
    out_path.write_bytes(
        gzip.compress(
            etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
        )
    )

    n_parsed = len(parse_layer_clips(load_als_xml(out_path)))
    if n_parsed != placed:
        sys.exit(f"VALIDATION FAIL: {n_parsed} clips parsed vs {placed} placed")
    print(
        f"wrote {out_path}\n  {len(ranked)} worst spans, {placed} clips "
        f"(PRED red / GT green), round-trip OK"
        + (f"; skipped: {', '.join(skipped)}" if skipped else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
