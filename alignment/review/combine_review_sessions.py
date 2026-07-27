#!/usr/bin/env python3
"""Concatenate seeded review .als sessions into ONE listening-queue Live set.

The human-review loop produces several independent Ableton sessions (worst-spans
PRED-vs-GT per set, stem-match CAND/CAT pairs, fibers) that all share the same
construction: flat 60 BPM (beat == second), one clip per AudioTrack, verdicts
recorded by renaming tracks. That makes them concatenable: this script lays the
sessions end-to-end on one timeline — each source's tracks shifted by a running
offset, locators carried along, a "==== <label> ====" section locator at each
seam — so one Live sitting covers the whole listening backlog.

Verdict protocols are track-name renames and therefore survive the merge
unchanged (ACC/REJ for stem pairs, BUG/HARD/GTW for worst spans).
seed_stem_review_als --harvest works on the combined file: any .map.json
sidecar found next to an input is copied next to the output.

Usage:
    venvs/audio/bin/python -m alignment.review.combine_review_sessions \\
        --out "~/Desktop/LISTENING QUEUE.als" \\
        "STEM REVIEW:~/Desktop/stem_review_20260709/stem_review_20260709 REVIEW.als" \\
        "BB12 WORST SPANS:~/aligning/_review/BB12 WORST SPANS SEEDED.als" \\
        ...

Inputs are "<label>:<path>" (label optional — defaults to the filename stem).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import itertools
import shutil
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lxml import etree

from labeling.als import load_als_xml, write_locators
from alignment.review.seed_als_from_timeline import (
    _TEMPO_BPM,
    doc_max_id,
    renumber_pointee_ids,
)

_SECTION_GAP_S = 30.0  # silence between concatenated sessions


def _parse_input(spec: str) -> tuple[str, Path]:
    """ "<label>:<path>" or bare path (label = filename stem)."""
    if ":" in spec and not spec.startswith(("/", "~", ".")):
        label, _, path = spec.partition(":")
        return label, Path(path).expanduser().resolve()
    p = Path(spec).expanduser().resolve()
    return p.stem, p


def _assert_flat_60(root: etree._Element, name: str) -> None:
    tempo = root.find(".//MasterTrack//Tempo/Manual")
    if tempo is None or abs(float(tempo.get("Value", 0)) - _TEMPO_BPM) > 1e-6:
        sys.exit(
            f"{name}: master tempo is not flat {_TEMPO_BPM:.0f} BPM — only "
            "beat==second seeded sessions can be concatenated"
        )


def _audio_tracks(root: etree._Element) -> list[etree._Element]:
    return root.findall(".//LiveSet/Tracks/AudioTrack")


def _doc_span_beats(root: etree._Element) -> float:
    ends = [float(c.get("Value", 0)) for c in root.findall(".//AudioClip/CurrentEnd")]
    return max(ends) if ends else 0.0


def _shift_track_clips(track: etree._Element, offset: float) -> None:
    """Move every clip by `offset` arrangement beats. Warp markers and Loop
    values are clip-relative, so only the arrangement anchors move."""
    for clip in track.iter("AudioClip"):
        t = float(clip.get("Time", 0))
        clip.set("Time", f"{t + offset:.6f}")
        for tag in ("CurrentStart", "CurrentEnd"):
            el = clip.find(tag)
            if el is not None:
                el.set("Value", f"{float(el.get('Value', 0)) + offset:.6f}")


def _locators(root: etree._Element) -> list[tuple[float, str]]:
    out = []
    for loc in root.findall(".//Locators/Locators/Locator"):
        t_el, n_el = loc.find("Time"), loc.find("Name")
        if t_el is not None and n_el is not None:
            out.append((float(t_el.get("Value", 0)), n_el.get("Value", "")))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help='"<label>:<als path>" or bare path')
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    sections = [_parse_input(s) for s in args.inputs]
    for label, path in sections:
        if not path.is_file():
            sys.exit(f"{label}: no such file {path}")
    docs = [(label, path, load_als_xml(path)) for label, path in sections]
    for label, _path, root in docs:
        _assert_flat_60(root, label)

    # base document = first input; the rest contribute AudioTracks + locators
    base_label, base_path, base = docs[0]
    tracks_node = base.find(".//LiveSet/Tracks")
    # keep return/master wiring intact: new tracks go before the first non-audio
    insert_at = list(tracks_node).index(_audio_tracks(base)[-1]) + 1

    global_max = max(doc_max_id(root) for _l, _p, root in docs)
    alloc = itertools.count(global_max + 10_000)

    n_clips_expected = sum(
        len(root.findall(".//LiveSet/Tracks/AudioTrack//AudioClip"))
        for _l, _p, root in docs
    )
    markers: list[tuple[float, str]] = [
        (
            0.0,
            f"LISTENING QUEUE {date.today().isoformat()} — verdicts by track "
            "rename per section protocol",
        ),
        (0.0, f"==== {base_label} ===="),
    ]
    markers += _locators(base)

    offset = _doc_span_beats(base) + _SECTION_GAP_S
    for label, _path, root in docs[1:]:
        markers.append((offset, f"==== {label} ===="))
        markers += [(t + offset, name) for t, name in _locators(root)]
        for track in _audio_tracks(root):
            t = copy.deepcopy(track)
            _shift_track_clips(t, offset)
            t.set("Id", str(next(alloc)))
            renumber_pointee_ids(t, alloc)
            tracks_node.insert(insert_at, t)
            insert_at += 1
        offset += _doc_span_beats(root) + _SECTION_GAP_S

    write_locators(base, markers)
    for npi in base.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))

    out_path = args.out.expanduser().resolve()
    if out_path.name.endswith(" align.als"):
        sys.exit("refusing the hand-GT '<SET> align.als' name for a merged session")
    out_path.write_bytes(
        gzip.compress(
            etree.tostring(
                base, xml_declaration=True, encoding="UTF-8", standalone=True
            )
        )
    )

    # carry the stem-review sidecar so --harvest works on the combined file
    sidecars = [
        p.with_suffix(".map.json")
        for _l, p, _r in docs
        if p.with_suffix(".map.json").is_file()
    ]
    if len(sidecars) > 1:
        sys.exit(f"multiple .map.json sidecars among inputs: {sidecars}")
    if sidecars:
        shutil.copy2(sidecars[0], out_path.with_suffix(".map.json"))

    # self-validate: reparse, count clips, check every referenced file exists
    reparsed = load_als_xml(out_path)
    clips = reparsed.findall(".//LiveSet/Tracks/AudioTrack//AudioClip")
    missing = [
        fr.get("Value")
        for fr in reparsed.findall(".//AudioClip/SampleRef/FileRef/Path")
        if not Path(fr.get("Value", "")).is_file()
    ]
    if len(clips) != n_clips_expected or missing:
        sys.exit(
            f"VALIDATION FAIL: {len(clips)} clips (want {n_clips_expected}), "
            f"missing files: {missing[:3]}"
        )
    print(
        f"wrote {out_path}\n{len(docs)} sections, {len(clips)} clips, "
        f"{len(markers)} locators, span {offset - _SECTION_GAP_S:.0f}s — validation OK"
    )
    if sidecars:
        print(f"sidecar carried: {out_path.with_suffix('.map.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
