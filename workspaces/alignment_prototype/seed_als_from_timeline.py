#!/usr/bin/env python3
"""Generate a pre-seeded Ableton .als from a predicted timeline.

Inverts labeling/export_als_to_gt.py: instead of a human placing clips that we
parse into GT, we place every predicted span as a clip the human then verifies
and corrects. The corrected project round-trips back through export_als_to_gt
to become the target set's ground truth.

Construction (chosen so labeling/als_io.py parses the result exactly):

  * template      <- a real labeling .als (BB12); we strip its tracks and keep
                     the LiveSet skeleton, deep-copying one AudioTrack as the
                     clip-bearing template
  * global tempo  <- 60 BPM, so 1 arrangement beat == 1 second
  * mix track     <- "1-mix", one warped clip, markers (0,0)..(dur,dur)
  * span tracks   <- one AudioTrack per predicted span, clip at
                     [set_start, set_end] beats, two warp markers mapping the
                     arrangement span onto [ref_start, ref_end] file seconds
                     (LoopStart=0, so als_io ref_start_s() == ref_start)
  * clip color    <- suspicion: red = unanchored or |pred-cue| > 45 s,
                     yellow = 25-45 s, green = < 25 s

The script self-validates: it re-parses its own output with als_io and asserts
every span survives the round trip before the human ever opens it.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.seed_als_from_timeline \\
        --set-id 2nvzlh2k [--out ~/Desktop/...] [--template <als>]
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lxml import etree

from labeling.als_io import (
    ArrangementMapper,
    build_manifest_index,
    load_als_xml,
    parse_layer_clips,
    resolve_identity,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
ALIGNING_ROOT = Path.home() / "aligning"
DEFAULT_TEMPLATE = (
    Path.home() / "Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als"
)

_TEMPO_BPM = 60.0  # 1 beat == 1 second: drag math stays trivial for the human
# Live 11 clip palette indices (approximate hues)
_COLOR_RED, _COLOR_YELLOW, _COLOR_GREEN, _COLOR_BLUE = 14, 2, 5, 9
_SUS_RED_S, _SUS_YELLOW_S = 45.0, 25.0


def ffprobe_audio(path: Path) -> tuple[float, int]:
    """(duration_s, sample_rate)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-show_entries", "stream=sample_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    j = json.loads(r.stdout)
    dur = float(j["format"]["duration"])
    sr = next((int(s["sample_rate"]) for s in j.get("streams", []) if s.get("sample_rate")), 44100)
    return dur, sr


def _set_value(parent: etree._Element, tag: str, value: str) -> None:
    el = parent.find(tag)
    if el is not None:
        el.set("Value", value)


def _suspicion(span: dict) -> float:
    cue = span.get("cue_anchor_s")
    if cue is None or cue <= 0.0:
        return 9999.0
    return abs(span["set_start_s"] - cue)


def _color_for(span: dict) -> int:
    s = _suspicion(span)
    if s >= _SUS_RED_S:
        return _COLOR_RED
    if s >= _SUS_YELLOW_S:
        return _COLOR_YELLOW
    return _COLOR_GREEN


def find_template_track(root: etree._Element) -> etree._Element:
    """A non-mix AudioTrack holding exactly one warped AudioClip."""
    for t in root.findall(".//LiveSet/Tracks/AudioTrack"):
        name = t.find(".//Name/EffectiveName")
        nm = name.get("Value") if name is not None else ""
        if nm.startswith(("1-mix", "2-mix")):
            continue
        clips = t.findall(".//ArrangerAutomation/Events/AudioClip")
        if len(clips) != 1:
            continue
        warped = clips[0].find("IsWarped")
        if warped is not None and warped.get("Value") == "true":
            return t
    sys.exit("no single-clip warped AudioTrack found in template .als")


def rewrite_clip(
    clip: etree._Element,
    *,
    name: str,
    color: int,
    arr_start: float,
    arr_end: float,
    file_path: Path,
    file_dur_s: float,
    sample_rate: int,
    ref_start_s: float,
    ref_end_s: float,
) -> None:
    arr_len = arr_end - arr_start
    _set_value(clip, "CurrentStart", f"{arr_start:.6f}")
    _set_value(clip, "CurrentEnd", f"{arr_end:.6f}")
    _set_value(clip, "Name", name)
    _set_value(clip, "Color", str(color))
    _set_value(clip, "IsWarped", "true")
    _set_value(clip, "PitchCoarse", "0")
    _set_value(clip, "PitchFine", "0")
    _set_value(clip, "Disabled", "false")
    loop = clip.find("Loop")
    _set_value(loop, "LoopStart", "0")
    _set_value(loop, "LoopEnd", f"{arr_len:.6f}")
    _set_value(loop, "StartRelative", "0")
    _set_value(loop, "LoopOn", "false")
    _set_value(loop, "OutMarker", f"{arr_len:.6f}")
    _set_value(loop, "HiddenLoopStart", "0")
    _set_value(loop, "HiddenLoopEnd", f"{arr_len:.6f}")

    # exactly two warp markers: arrangement-relative beat 0 -> ref_start_s,
    # beat arr_len -> ref_end_s (linear stretch; als_io interpolates the same)
    wm = clip.find(".//WarpMarkers")
    for w in list(wm):
        wm.remove(w)
    for i, (beat, sec) in enumerate(((0.0, ref_start_s), (arr_len, ref_end_s))):
        etree.SubElement(wm, "WarpMarker", Id=str(i), SecTime=f"{sec:.6f}", BeatTime=f"{beat:.6f}")

    # repoint the sample; drop every OriginalFileRef (one hides nested in
    # SampleRef/SourceContext) so clip_original_path falls through to
    # SampleRef/FileRef/Path instead of a stale template path
    for ofr in clip.findall(".//OriginalFileRef"):
        ofr.getparent().remove(ofr)
    sref = clip.find("SampleRef")
    fref = sref.find("FileRef")
    _set_value(fref, "Path", str(file_path))
    _set_value(fref, "RelativePath", str(file_path))
    _set_value(fref, "RelativePathType", "0")
    _set_value(sref, "DefaultDuration", str(int(round(file_dur_s * sample_rate))))
    _set_value(sref, "DefaultSampleRate", str(sample_rate))


def build_track(
    template: etree._Element,
    *,
    track_id: int,
    track_name: str,
    **clip_kwargs,
) -> etree._Element:
    t = copy.deepcopy(template)
    t.set("Id", str(track_id))
    _set_value(t.find(".//Name"), "EffectiveName", track_name)
    _set_value(t.find(".//Name"), "UserName", track_name)
    _set_value(t, "TrackGroupId", "-1")
    clip = t.find(".//ArrangerAutomation/Events/AudioClip")
    rewrite_clip(clip, **clip_kwargs)
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    timeline = json.loads((OUT_DIR / f"{args.set_id}_predicted_timeline.json").read_text())
    spans = timeline["spans"]

    set_dirs = sorted(ALIGNING_ROOT.glob(f"{args.set_id}__*"))
    if not set_dirs:
        sys.exit(f"no ~/aligning folder for {args.set_id}")
    set_dir = set_dirs[0]
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    mix_path = Path(manifest["mix_local_path"])
    mix_dur, mix_sr = ffprobe_audio(mix_path)

    root = load_als_xml(args.template)
    template_track = copy.deepcopy(find_template_track(root))

    # strip all audio + group tracks; keep returns and the master chain
    tracks_node = root.find(".//LiveSet/Tracks")
    for t in list(tracks_node):
        if t.tag in ("AudioTrack", "GroupTrack", "MidiTrack"):
            tracks_node.remove(t)

    # global tempo: 60 BPM (1 beat = 1 s)
    for tempo_manual in root.xpath(".//MasterTrack//Tempo/Manual"):
        tempo_manual.set("Value", f"{_TEMPO_BPM:.1f}")

    next_id = 1000  # clear of any surviving template ids
    mix_track = build_track(
        template_track,
        track_id=next_id,
        track_name="1-mix",
        name="mix",
        color=_COLOR_BLUE,
        arr_start=0.0,
        arr_end=mix_dur,
        file_path=mix_path,
        file_dur_s=mix_dur,
        sample_rate=mix_sr,
        ref_start_s=0.0,
        ref_end_s=mix_dur,
    )
    tracks_node.insert(0, mix_track)

    placed: list[dict] = []
    skipped: list[str] = []
    insert_at = 1
    for i, s in enumerate(spans):
        t = by_tid.get(s["recording_id"])
        if t is None or not Path(t["local_path"]).is_file():
            skipped.append(f"{s['slot_label']} {s['name'][:50]}")
            continue
        fpath = Path(t["local_path"])
        fdur, fsr = ffprobe_audio(fpath)
        ref_start = max(0.0, min(s["ref_start_s"], fdur - 1.0))
        ref_end = max(ref_start + 1.0, min(s["ref_end_s"], fdur))
        next_id += 1
        sus = _suspicion(s)
        sus_tag = "?" if sus >= 9999 else f"{sus:.0f}s"
        track = build_track(
            template_track,
            track_id=next_id,
            track_name=f"{s['slot_label']}__{s['name']}",
            name=f"{s['slot_label']}__{s['name']} [{sus_tag}]",
            color=_color_for(s),
            arr_start=s["set_start_s"],
            arr_end=s["set_start_s"] + (s["set_end_s"] - s["set_start_s"]),
            file_path=fpath,
            file_dur_s=fdur,
            sample_rate=fsr,
            ref_start_s=ref_start,
            ref_end_s=ref_end,
        )
        tracks_node.insert(insert_at, track)
        insert_at += 1
        placed.append({**s, "ref_start_s": ref_start, "ref_end_s": ref_end, "path": str(fpath)})
        if (i + 1) % 30 == 0:
            print(f"  placed {i + 1}/{len(spans)}")

    out_path = args.out or (
        Path.home() / "Desktop" / f"{args.set_id} predicted review Project"
        / f"{args.set_id} predicted review.als"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(gzip.compress(etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True,
    )))
    print(f"\nwrote {out_path} ({len(placed)} clips, {len(skipped)} skipped)")
    for m in skipped:
        print(f"  SKIP {m}")

    # ---- round-trip validation through the real GT export parser -----------
    reparsed = load_als_xml(out_path)
    mix_tracks = [
        t for t in reparsed.findall(".//LiveSet/Tracks/AudioTrack")
        if (t.find(".//Name/EffectiveName") is not None
            and t.find(".//Name/EffectiveName").get("Value", "").startswith("1-mix"))
    ]
    mapper = ArrangementMapper.from_mix_track(mix_tracks[0], mix_duration_s=mix_dur)
    clips = parse_layer_clips(reparsed)
    mindex = build_manifest_index(set_dir / "manifest.json")
    if len(clips) != len(placed):
        sys.exit(f"VALIDATION FAIL: {len(clips)} clips parsed, {len(placed)} placed")
    errs = 0
    for clip, s in zip(sorted(clips, key=lambda c: c.arr_start),
                       sorted(placed, key=lambda x: x["set_start_s"])):
        set_start = mapper.arr_to_set_sec(clip.arr_start)
        rid, _slot, _label, _stem = resolve_identity(clip, mindex)
        bad = []
        if set_start is None or abs(set_start - s["set_start_s"]) > 0.05:
            bad.append(f"set_start {set_start} != {s['set_start_s']:.2f}")
        if abs(clip.ref_start_s() - s["ref_start_s"]) > 0.05:
            bad.append(f"ref_start {clip.ref_start_s():.2f} != {s['ref_start_s']:.2f}")
        if rid != s["recording_id"]:
            bad.append(f"identity {rid} != {s['recording_id']}")
        if bad:
            errs += 1
            print(f"  MISMATCH {s['slot_label']}: {'; '.join(bad)}")
    if errs:
        sys.exit(f"VALIDATION FAIL: {errs}/{len(placed)} clips mismatched")
    print(f"round-trip validation OK: {len(placed)} clips parse back exactly "
          f"(set_start, ref_start, identity)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
