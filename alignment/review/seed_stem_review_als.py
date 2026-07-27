#!/usr/bin/env python3
"""Stem-match review as an Ableton session — the worst-spans A/B pattern.

Renders a Discord stem-library review queue (`apply_stem_matches --review-out`)
as one Live set, reusing the seed machinery from seed_als_from_timeline:

  * "CAND nn <file>"           — the downloaded acappella/instrumental (red)
  * "CAT  nn <artist - title>" — the catalog recording the matcher proposed
                                 it belongs to (green)

Pairs are laid out sequentially (candidate then catalog) with one locator per
pair, so you can play straight through or jump by locator. Verdict protocol
(same idea as seed_worst_spans_als BUG/HARD/GTW): rename a pair's CAND track
prefix to **ACC** (accept — file IS that song's acappella/instrumental) or
**REJ** (reject), keep the number, save. Then:

    --harvest <als>   parses the renames back into the queue CSV
                      (decision -> accept / reject) and prints the
                      apply_stem_matches command for the accepts.

Seed usage:
    venvs/audio/bin/python -m alignment.review.seed_stem_review_als \\
        --pack ~/Desktop/stem_review_20260709 --queue <review_queue.csv>

Harvest usage (after listening + renaming in Live):
    ... seed_stem_review_als --harvest "<pack>/<pack name> REVIEW.als"
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import itertools
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lxml import etree

from labeling.als import load_als_xml
from alignment.review.seed_als_from_timeline import (
    DEFAULT_TEMPLATE,
    _TEMPO_BPM,
    build_track,
    doc_max_id,
    ffprobe_audio,
    find_template_track,
    renumber_pointee_ids,
)

_COLOR_RED, _COLOR_GREEN = 14, 5
_GAP_WITHIN_PAIR_S = 2.0
_GAP_BETWEEN_PAIRS_S = 6.0
_NAME_MAX = 56
# verdict rename on either track of a pair: "ACC 03 ...", "rej04 ...", ...
_VERDICT_RE = re.compile(r"^\s*(acc|rej)\W*(\d{1,2})\b", re.IGNORECASE)
_CAND_RE = re.compile(r"^\s*cand\W*(\d{1,2})\b", re.IGNORECASE)


def _pairs_from_pack(pack: Path) -> list[dict]:
    """Numbered pack subfolders -> [{nn, cand (m4a), cat (m4a)}]."""
    out = []
    for d in sorted(p for p in pack.iterdir() if p.is_dir()):
        m = re.match(r"^(\d{1,2})\s", d.name)
        if not m:
            continue
        cand = sorted(d.glob("CANDIDATE - *.m4a"))
        cat = sorted(d.glob("CATALOG - *.m4a"))
        if len(cand) != 1 or len(cat) != 1:
            sys.exit(f"pack folder {d.name!r}: expected 1 CANDIDATE + 1 CATALOG m4a")
        out.append({"nn": int(m.group(1)), "cand": cand[0], "cat": cat[0]})
    if not out:
        sys.exit(f"no numbered pair folders under {pack}")
    return out


def _assign_rows(pairs: list[dict], rows: list[dict]) -> dict[int, int]:
    """Pair nn -> queue-CSV row index, matched on the candidate file stem.

    The pack numbers folders alphabetically while the CSV keeps matcher order,
    so positional zip is wrong. Duplicate stems (the same candidate file staged
    twice) are assigned greedily in order — interchangeable for verdict
    purposes, and the sha256 skip-guard dedups at apply time anyway."""
    by_stem: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_stem.setdefault(Path(row["file"]).stem, []).append(i)
    out: dict[int, int] = {}
    for pair in pairs:
        stem = pair["cand"].stem.removeprefix("CANDIDATE - ")
        free = by_stem.get(stem) or []
        if not free:
            sys.exit(
                f"pair {pair['nn']:02d}: candidate {stem!r} has no unclaimed "
                "queue row — pack and CSV disagree; re-render the pack"
            )
        out[pair["nn"]] = free.pop(0)
    return out


def _trunc(s: str, n: int = _NAME_MAX) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def seed(args: argparse.Namespace) -> int:
    pack = args.pack.expanduser().resolve()
    pairs = _pairs_from_pack(pack)
    rows = list(csv.DictReader(args.queue.expanduser().open()))
    if len(rows) != len(pairs):
        sys.exit(f"{len(pairs)} pack pairs but {len(rows)} queue rows")
    row_of = _assign_rows(pairs, rows)

    # keep the whole review kit in one folder: CSV copy travels with the .als
    queue_copy = pack / "review_queue.csv"
    if args.queue.expanduser().resolve() != queue_copy:
        shutil.copy2(args.queue.expanduser(), queue_copy)

    root = load_als_xml(args.template)
    template_track = copy.deepcopy(find_template_track(root))
    tracks_node = root.find(".//LiveSet/Tracks")
    for t in list(tracks_node):
        if t.tag in ("AudioTrack", "GroupTrack", "MidiTrack"):
            tracks_node.remove(t)

    # flat 60 BPM: beat == second, everything plays unwarped at natural rate
    try:
        from labeling.als import write_tempo_envelope

        write_tempo_envelope(root, [(0.0, _TEMPO_BPM)])
    except Exception:
        for tempo_manual in root.xpath(".//MasterTrack//Tempo/Manual"):
            tempo_manual.set("Value", f"{_TEMPO_BPM:.1f}")

    alloc = itertools.count(doc_max_id(root) + len(pairs) * 4 + 2000)
    next_id = 1000
    t_cursor = 0.0
    markers: list[tuple[float, str]] = [
        (
            0.0,
            f"STEM REVIEW {date.today().isoformat()} — rename CAND nn -> "
            "ACC nn / REJ nn, save, then --harvest",
        )
    ]
    insert_at = 0
    for pair in pairs:
        row = rows[row_of[pair["nn"]]]
        nn = pair["nn"]
        proposed = f"{row['cand_artist']} - {row['cand_title']}"
        markers.append(
            (
                t_cursor,
                f"{nn:02d} {_trunc(Path(row['file']).stem, 40)}  vs  "
                f"{_trunc(proposed, 40)} [audio {row.get('audio_score') or '?'}]",
            )
        )
        for prefix, color, fpath, label in (
            (
                "CAND",
                _COLOR_RED,
                pair["cand"],
                _trunc(pair["cand"].stem.removeprefix("CANDIDATE - ")),
            ),
            ("CAT ", _COLOR_GREEN, pair["cat"], _trunc(proposed)),
        ):
            dur, sr = ffprobe_audio(fpath)
            next_id += 1
            track = build_track(
                template_track,
                track_id=next_id,
                track_name=f"{prefix}{nn:02d} {label}",
                name=f"{prefix}{nn:02d} {label}",
                color=color,
                arr_start=t_cursor,
                arr_end=t_cursor + dur,
                file_path=fpath,
                file_dur_s=dur,
                sample_rate=sr,
                ref_start_s=0.0,
                ref_end_s=dur,
                is_warped=False,  # natural-rate listening, no stretch
            )
            renumber_pointee_ids(track, alloc)
            tracks_node.insert(insert_at, track)
            insert_at += 1
            t_cursor += dur + _GAP_WITHIN_PAIR_S
        t_cursor += _GAP_BETWEEN_PAIRS_S - _GAP_WITHIN_PAIR_S

    try:
        from labeling.als import write_locators

        write_locators(root, markers)
    except Exception as exc:
        print(f"  locators skipped: {exc}", file=sys.stderr)

    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))

    out_path = args.out or pack / f"{pack.name} REVIEW.als"
    out_path.write_bytes(
        gzip.compress(
            etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
        )
    )
    sidecar = out_path.with_suffix(".map.json")
    sidecar.write_text(
        json.dumps(
            {
                "queue_csv": str(queue_copy),
                "pairs": {
                    str(p["nn"]): {
                        "row": row_of[p["nn"]],
                        "path": rows[row_of[p["nn"]]]["path"],
                    }
                    for p in pairs
                },
            },
            indent=2,
        )
    )

    # self-validate: every clip parses back and points at an existing file
    reparsed = load_als_xml(out_path)
    clips = reparsed.findall(".//LiveSet/Tracks/AudioTrack//AudioClip")
    missing = [
        fr.get("Value")
        for fr in reparsed.findall(".//AudioClip/SampleRef/FileRef/Path")
        if not Path(fr.get("Value", "")).is_file()
    ]
    if len(clips) != 2 * len(pairs) or missing:
        sys.exit(
            f"VALIDATION FAIL: {len(clips)} clips (want {2 * len(pairs)}), "
            f"missing files: {missing[:3]}"
        )
    print(
        f"wrote {out_path}\n{len(pairs)} pairs, {len(clips)} clips, "
        f"validation OK\nsidecar: {sidecar}"
    )
    return 0


def harvest(args: argparse.Namespace) -> int:
    als = args.harvest.expanduser().resolve()
    sidecar = als.with_suffix(".map.json")
    if not sidecar.is_file():
        sys.exit(f"no sidecar {sidecar} — was this .als written by seed mode?")
    side = json.loads(sidecar.read_text())
    queue_csv = Path(side["queue_csv"])
    pair_map: dict[str, dict] = side["pairs"]

    verdicts: dict[int, str] = {}
    pending: list[int] = []
    root = load_als_xml(als)
    for t in root.findall(".//LiveSet/Tracks/AudioTrack"):
        name_el = t.find(".//Name/EffectiveName")
        nm = name_el.get("Value", "") if name_el is not None else ""
        m = _VERDICT_RE.match(nm)
        if m:
            nn = int(m.group(2))
            verdict = "accept" if m.group(1).lower() == "acc" else "reject"
            if verdicts.get(nn, verdict) != verdict:
                sys.exit(f"pair {nn:02d}: conflicting ACC and REJ track renames")
            verdicts[nn] = verdict
        elif _CAND_RE.match(nm):
            pending.append(int(_CAND_RE.match(nm).group(1)))

    pending = sorted(set(pending) - set(verdicts))
    rows = list(csv.DictReader(queue_csv.open()))
    changed = 0
    for nn_str, entry in pair_map.items():
        v = verdicts.get(int(nn_str))
        if v is None:
            continue
        row = rows[entry["row"]]
        if row["path"] != entry["path"]:
            sys.exit(f"pair {nn_str}: CSV row order changed since seeding")
        if row["decision"] != v:
            row["decision"] = v
            changed += 1

    bak = queue_csv.with_suffix(".csv.bak")
    if not bak.exists():
        shutil.copy2(queue_csv, bak)
    with queue_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_acc = sum(1 for v in verdicts.values() if v == "accept")
    n_rej = len(verdicts) - n_acc
    print(
        f"harvested {len(verdicts)} verdicts ({n_acc} accept / {n_rej} reject), "
        f"{changed} CSV rows updated -> {queue_csv}"
    )
    if pending:
        print(f"still unmarked (left as-is): {sorted(pending)}")
    if n_acc:
        print(
            "\napply the accepts with:\n"
            f"  venvs/audio/bin/python scripts/apply_stem_matches.py "
            f"--csv '{queue_csv}' --pi-files "
            "--reason 'source:discord|identity:human-reviewed|batch:unzipped_20260709'"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, help="review-pack folder (numbered pair dirs)")
    p.add_argument("--queue", type=Path, help="review-queue CSV (apply --review-out)")
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--harvest", type=Path, help="parse verdict renames from this .als")
    args = p.parse_args(argv)
    if args.harvest:
        return harvest(args)
    if not (args.pack and args.queue):
        p.error("seed mode needs --pack and --queue (or use --harvest)")
    return seed(args)


if __name__ == "__main__":
    sys.exit(main())
