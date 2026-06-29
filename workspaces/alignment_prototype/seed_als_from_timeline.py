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
# Friendly display names for output filenames (fallback: the set_id itself).
_SET_DISPLAY = {
    "1fsnxchk": "BB12",
    "2nvzlh2k": "BB11",
    "w1mgcjt": "BB10",
    "pwgrrb1": "Murph",
}
# A dedicated *clean* seed template (one warped audio track + master), NOT the
# live labeling session — deep-copying the evolving labeling .als crashes Live
# (accumulated device/automation state). Pin a stable copy here. Recreate from
# any early Ableton backup if lost. See als-seed crash debugging, 2026-06-16.
DEFAULT_TEMPLATE = Path.home() / "aligning/_seed_template.als"

_TEMPO_BPM = 60.0  # 1 beat == 1 second: drag math stays trivial for the human
# Live 11 clip palette indices (approximate hues)
_COLOR_RED, _COLOR_YELLOW, _COLOR_GREEN, _COLOR_BLUE = 14, 2, 5, 9
_SUS_RED_S, _SUS_YELLOW_S = 45.0, 25.0


def ffprobe_audio(path: Path) -> tuple[float, int]:
    """(duration_s, sample_rate)."""
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    j = json.loads(r.stdout)
    dur = float(j["format"]["duration"])
    sr = next(
        (int(s["sample_rate"]) for s in j.get("streams", []) if s.get("sample_rate")),
        44100,
    )
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
    is_warped: bool = True,
) -> None:
    arr_len = arr_end - arr_start
    # Time attr = the clip's ARRANGEMENT position (beats); Ableton positions by
    # this, not CurrentStart. The template clip's Time is inherited by every
    # deep-copy, so without this every clip stacks at the template's beat.
    clip.set("Time", f"{arr_start:.6f}")
    _set_value(clip, "CurrentStart", f"{arr_start:.6f}")
    _set_value(clip, "CurrentEnd", f"{arr_end:.6f}")
    _set_value(clip, "Name", name)
    _set_value(clip, "Color", str(color))
    # The mix is placed UNWARPED so it plays at natural real time — a fixed,
    # tempo-agnostic "ruler" (1 beat = 1 s at the 60-BPM grid). Layer clips stay
    # warped so they stretch onto their ref sections.
    _set_value(clip, "IsWarped", "true" if is_warped else "false")
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
        etree.SubElement(
            wm, "WarpMarker", Id=str(i), SecTime=f"{sec:.6f}", BeatTime=f"{beat:.6f}"
        )

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


def strip_automation(track: etree._Element) -> None:
    """Empty every <Envelopes> in a deep-copied track.

    A copied track duplicates its AutomationEnvelope/PointeeId cross-references
    document-wide. als_io ignores them (so round-trip validation passes), but
    Ableton can't reconcile the duplicates — it offers to "fix" the file and
    then CRASHES during the migration. With no envelopes, nothing references the
    automation targets, so the deep-copy is safe and the .als opens cleanly.
    A track with empty automation is fully valid (just no automation drawn)."""
    for env in track.iter("Envelopes"):
        for child in list(env):
            env.remove(child)


def build_track(
    template: etree._Element,
    *,
    track_id: int,
    track_name: str,
    **clip_kwargs,
) -> etree._Element:
    t = copy.deepcopy(template)
    strip_automation(t)  # remove cross-references that crash Ableton on copy
    t.set("Id", str(track_id))
    _set_value(t.find(".//Name"), "EffectiveName", track_name)
    _set_value(t.find(".//Name"), "UserName", track_name)
    _set_value(t, "TrackGroupId", "-1")
    clip = t.find(".//ArrangerAutomation/Events/AudioClip")
    rewrite_clip(clip, **clip_kwargs)
    return t


def doc_max_id(root: etree._Element) -> int:
    mx = 0
    for el in root.iter():
        for attr in ("Id",):
            v = el.get(attr)
            if v is not None and v.lstrip("-").isdigit():
                mx = max(mx, int(v))
        if el.tag == "NextPointeeId":
            v = el.get("Value")
            if v and v.isdigit():
                mx = max(mx, int(v))
    return mx


def renumber_pointee_ids(track: etree._Element, alloc) -> None:
    """Deep-copied tracks share ids in Live's global *pointee namespace*
    (governed by NextPointeeId) — every `*Target` AND `<Pointee>` element must be
    unique document-wide, or Live offers to "fix" and then CRASHES. Re-id every
    such element in the copy and rewrite same-track PointeeId references to match.
    (The earlier version missed <Pointee>: 450 duplicated Pointee ids = the crash.)"""
    idmap: dict[str, str] = {}
    for el in track.iter():
        if el.get("Id") is not None and (
            el.tag.endswith("Target") or el.tag == "Pointee"
        ):
            new = str(next(alloc))
            idmap[el.get("Id")] = new
            el.set("Id", new)
    for el in track.iter("PointeeId"):
        v = el.get("Value")
        if v in idmap:
            el.set("Value", idmap[v])


_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def pick_audio(span: dict, track: dict) -> Path | None:
    """Clip audio: the Demucs stem for acappella/instrumental claims, else
    the full track. Stems share the full track's timeline, so predicted ref
    offsets hold — and export_als_to_gt's classify_path() reads the stem
    kind back from the stems/ path, so claimed_stem round-trips into GT."""
    stem_key = _STEM_FILE.get(span.get("claimed_stem") or "regular")
    if stem_key:
        stem_path = (track.get("stems") or {}).get(stem_key)
        if stem_path and Path(stem_path).is_file():
            return Path(stem_path)
    p = Path(track["local_path"])
    return p if p.is_file() else None


def _track_bpm(track: dict | None) -> float | None:
    """Per-track BPM from the M4A iTunes 'tmpo' tag (written by
    tag_aligning_folder, Essentia-accurate) — set-agnostic, unlike the BB12-only
    [NNNbpm] filename convention. Falls through the 'essentia bpm=' comment."""
    if not track:
        return None
    p = Path(track.get("local_path") or "")
    if not p.is_file():
        return None
    try:
        from mutagen.mp4 import MP4

        tags = MP4(str(p)).tags or {}
        if tags.get("tmpo"):
            return float(tags["tmpo"][0])
        cmt = str((tags.get("\xa9cmt") or [""])[0])
        import re as _re

        m = _re.search(r"bpm=(\d+(?:\.\d+)?)", cmt)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def add_tempo_and_markers(
    root, spans: list[dict], set_id: str, set_dir: Path, by_tid: dict
) -> tuple[int, int]:
    """Mark every song on the fixed real-time ruler (the unwarped mix).

    One arrangement locator per bed span at its mix-second (60-BPM grid =>
    beat=second), labelled with that song's BPM read from the M4A iTunes tag
    (accurate, every song — not the BB12-only filename convention, which left the
    back half un-marked). No tempo automation: an unwarped-mix ruler and live
    tempo automation are mutually exclusive, so BPM lives in the readout, not by
    warping time. Best-effort: returns 0 if no bed spans. Returns marker count.
    """
    from labeling.als_io import write_locators

    bed = sorted(
        (
            s
            for s in spans
            if (s.get("claimed_stem") or "regular") != "acappella"
            and s.get("set_start_s") is not None
        ),
        key=lambda s: float(s["set_start_s"]),
    )
    if not bed:
        return 0, 0

    markers: list[tuple[float, str]] = []
    for s in bed:
        bpm = _track_bpm(by_tid.get(s.get("recording_id")))
        name = str(s.get("name") or s.get("slot_label") or "")
        label = f"{bpm:.0f} BPM — {name}" if bpm else name
        markers.append((float(s["set_start_s"]), label))
    try:
        n_m = write_locators(root, markers)
    except Exception as exc:  # template lacks a locator block — skip, keep the seed
        print(f"  markers skipped: {exc}", file=sys.stderr)
        return 0, 0
    return 0, n_m


def _clip_specs(s: dict) -> list[tuple[float, float, float, float]]:
    """(arr_start, arr_end, ref_start, ref_end) per clip for a span. One clip per
    ref_segment (loops/cuts/warps -> one warped Ableton clip each, arrangement-
    ordered); falls back to a single span-length clip from the scalar ref_start/end.
    A loop is just consecutive clips whose ref_start steps back — Live re-triggers."""
    segs = s.get("ref_segments")
    if segs:
        out = []
        for j, seg in enumerate(segs):
            a = float(seg["mix_start_s"])
            b = (
                float(segs[j + 1]["mix_start_s"])
                if j + 1 < len(segs)
                else float(s["set_end_s"])
            )
            out.append(
                (a, max(a + 1.0, b), float(seg["ref_start_s"]), float(seg["ref_end_s"]))
            )
        return out
    return [
        (
            float(s["set_start_s"]),
            float(s["set_end_s"]),
            float(s["ref_start_s"]),
            float(s["ref_end_s"]),
        )
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing seed .als without backing it up to .seedbak",
    )
    args = p.parse_args(argv)

    timeline = json.loads(
        (OUT_DIR / f"{args.set_id}_predicted_timeline.json").read_text()
    )
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

    # Global tempo: a FLAT 60 BPM (1 beat = 1 s). The unwarped-mix ruler only
    # works at exactly 60 — at any other tempo the native-rate mix and the
    # beat=second clip grid diverge (a leftover template tempo automation made
    # the mix span ~2x the beats, so clips covered only its first ~half).
    # write_tempo_envelope sets Manual AND replaces any drawn tempo automation
    # with a flat line, so nothing overrides 60.
    try:
        from labeling.als_io import write_tempo_envelope

        write_tempo_envelope(root, [(0.0, _TEMPO_BPM)])
    except Exception:
        for tempo_manual in root.xpath(".//MasterTrack//Tempo/Manual"):
            tempo_manual.set("Value", f"{_TEMPO_BPM:.1f}")

    import itertools

    # one Ableton clip per ref_segment -> total clips can far exceed len(spans)
    n_clip_est = sum(len(s.get("ref_segments") or [1]) for s in spans)
    alloc = itertools.count(doc_max_id(root) + n_clip_est * 2 + 2000)

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
        is_warped=False,  # the mix is the fixed real-time ruler
    )
    renumber_pointee_ids(mix_track, alloc)
    tracks_node.insert(0, mix_track)

    placed: list[dict] = []
    skipped: list[str] = []
    insert_at = 1
    for i, s in enumerate(spans):
        t = by_tid.get(s["recording_id"])
        fpath = pick_audio(s, t) if t is not None else None
        if fpath is None:
            skipped.append(f"{s['slot_label']} {s['name'][:50]}")
            continue
        fdur, fsr = ffprobe_audio(fpath)
        sus = _suspicion(s)
        sus_tag = "?" if sus >= 9999 else f"{sus:.0f}s"
        # one clip per ref_segment (loops/cuts/warps); each its own track to keep
        # the proven per-track deep-copy + renumber_pointee_ids path (cloning clips
        # within a track collides ids in renumber's idmap -> Live crash).
        specs = _clip_specs(s)
        for j, (a_start, a_end, rs, re) in enumerate(specs):
            ref_start = max(0.0, min(rs, fdur - 1.0))
            ref_end = max(ref_start + 1.0, min(re, fdur))
            seg_tag = f" seg{j + 1}/{len(specs)}" if len(specs) > 1 else ""
            next_id += 1
            track = build_track(
                template_track,
                track_id=next_id,
                track_name=f"{s['slot_label']}__{s['name']}{seg_tag}",
                name=f"{s['slot_label']}__{s['name']}{seg_tag} [{sus_tag}]",
                color=_color_for(s),
                arr_start=a_start,
                arr_end=a_end,
                file_path=fpath,
                file_dur_s=fdur,
                sample_rate=fsr,
                ref_start_s=ref_start,
                ref_end_s=ref_end,
            )
            renumber_pointee_ids(track, alloc)
            tracks_node.insert(insert_at, track)
            insert_at += 1
            placed.append(
                {
                    **s,
                    "set_start_s": a_start,
                    "ref_start_s": ref_start,
                    "ref_end_s": ref_end,
                    "path": str(fpath),
                }
            )
        if (i + 1) % 30 == 0:
            print(f"  placed {i + 1}/{len(spans)}")

    # one BPM marker per song on the unwarped-mix real-time ruler
    _, n_m = add_tempo_and_markers(root, spans, args.set_id, set_dir, by_tid)
    print(f"placed {n_m} per-song BPM markers (unwarped mix ruler)")

    # Live requires NextPointeeId above every pointee id in the document.
    for npi in root.findall(".//NextPointeeId"):
        npi.set("Value", str(next(alloc)))

    # Default: write the seed straight into the set's aligning folder (alongside
    # mix/tracks/stems/manifest), the same dir export_als_to_gt --set-dir reads.
    display = _SET_DISPLAY.get(args.set_id, args.set_id)
    out_path = args.out or (set_dir / f"{display} align.als")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Safety: never silently clobber an existing .als at the default seed location —
    # it may hold un-exported hand-corrections. Back it up unless --force.
    if out_path.exists() and not args.force:
        bak = out_path.with_suffix(".als.seedbak")
        n = 1
        while bak.exists():
            bak = out_path.with_suffix(f".als.seedbak{n}")
            n += 1
        out_path.rename(bak)
        print(f"backed up existing {out_path.name} -> {bak.name}")
    out_path.write_bytes(
        gzip.compress(
            etree.tostring(
                root,
                xml_declaration=True,
                encoding="UTF-8",
                standalone=True,
            )
        )
    )
    print(f"\nwrote {out_path} ({len(placed)} clips, {len(skipped)} skipped)")
    for m in skipped:
        print(f"  SKIP {m}")

    # ---- round-trip validation through the real GT export parser -----------
    reparsed = load_als_xml(out_path)
    mix_tracks = [
        t
        for t in reparsed.findall(".//LiveSet/Tracks/AudioTrack")
        if (
            t.find(".//Name/EffectiveName") is not None
            and t.find(".//Name/EffectiveName").get("Value", "").startswith("1-mix")
        )
    ]
    mapper = ArrangementMapper.from_mix_track(mix_tracks[0], mix_duration_s=mix_dur)
    clips = parse_layer_clips(reparsed)
    mindex = build_manifest_index(set_dir / "manifest.json")
    if len(clips) != len(placed):
        sys.exit(f"VALIDATION FAIL: {len(clips)} clips parsed, {len(placed)} placed")
    # Content-based matching (NOT positional zip): segments + concurrent finale
    # layers put many clips at the same (set_start, ref_start), so each parsed clip
    # must find SOME placed entry matching (recording_id, set_start, ref_start)
    # within tolerance, consumed greedily.
    remaining = list(placed)
    errs = 0
    for clip in clips:
        set_start = mapper.arr_to_set_sec(clip.arr_start)
        rid, _slot, _label, _stem = resolve_identity(clip, mindex)
        rs = clip.ref_start_s()
        hit = None
        for k, s in enumerate(remaining):
            if (
                rid == s["recording_id"]
                and set_start is not None
                and abs(set_start - s["set_start_s"]) <= 0.05
                and abs(rs - s["ref_start_s"]) <= 0.05
            ):
                hit = k
                break
        if hit is None:
            errs += 1
            if errs <= 12:
                print(
                    f"  MISMATCH clip rid={rid} set_start={set_start} ref_start={rs:.2f}"
                    " — no matching placed segment"
                )
        else:
            remaining.pop(hit)
    if errs:
        sys.exit(f"VALIDATION FAIL: {errs}/{len(placed)} clips unmatched")
    print(
        f"round-trip validation OK: {len(placed)} clips parse back exactly "
        f"(recording_id, set_start, ref_start)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
