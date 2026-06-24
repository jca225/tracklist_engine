#!/usr/bin/env python3
"""Audio-verify a labeling .als against the mix, fixing every misalignment class.

The labeling session asserts, per clip: a source song (identity), where it plays
in the mix (placement), where in the source it comes from (ref offset), a pitch
shift and a time-stretch. NONE of that has ever been checked against the actual
mix audio — the GT is asserted from Ableton geometry alone. This tool closes that
gap by, for each clip:

  identity   <- als_io reads SourceContext/OriginalFileRef (the real stems/<song>
                path, not the flattened Samples/Imported/vocals-N.flac). For the
                tail with no OriginalFileRef, we content-match the flattened file
                back to a canonical stem by OriginalFileSize (+CRC tiebreak).
  placement  <- ArrangementMapper maps arrangement beats -> mix seconds.
  ref/stretch<- WarpMarkers give ref_start/ref_end and the warp ratio.
  pitch      <- PitchCoarse.

then VERIFIES with a pitch-rotated chroma matched filter (the proven
refine_ref_offsets primitive): warp the source segment as asserted, slide it over
the mix stem, and ask three questions:
  * does it actually appear at the asserted position?        -> placement OK?
  * does a strong peak appear somewhere ELSE?                 -> position mismatch
  * does it appear at all, at any pitch?                      -> wrong audio / id?
  * is the best pitch the asserted PitchCoarse?               -> pitch mismatch

Output: out/<set_id>_als_audit.json + a per-clip status table.

    venvs/audio/bin/python -m workspaces.source_detection.als_audit \\
        --set-id 1fsnxchk [--als <path>] [--limit N] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling import als_io  # noqa: E402
from workspaces.source_detection import config, features  # noqa: E402
from workspaces.source_detection.matcher import _match_curve, _resample_cols  # noqa: E402

FRAME_S = config.FRAME_S
VERIFY_WINDOW_S = 24.0      # max source seconds used as the matched-filter template
POS_TOL_S = 3.0            # placed vs detected within this == placement confirmed
SCORE_OK = 0.50           # matched-filter score floor to call a peak "real"
SCORE_WEAK = 0.40         # below this everywhere == source doesn't appear


# --------------------------------------------------------------------------- records
@dataclass
class ClipFacts:
    """Everything the .als asserts about one clip + how identity was resolved."""

    idx: int
    track_name: str
    group: str
    als_path: str            # OriginalFileRef path (or flattened fallback)
    src_path: str | None     # resolved audio file on disk
    id_method: str           # orig-path | size-match | crc-match | ambiguous | unresolved
    song: str
    claimed_stem: str        # acappella | instrumental | regular
    slot: str
    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float
    pitch_coarse: int
    asserted_stretch: float  # mix seconds per source second (warp ratio)
    audible_frac: float = 1.0  # fraction of the span where track volume > mute floor
    unwarped: bool = False     # IsWarped=false: 1:1 playback, warp markers don't apply
    n_warp_markers: int = 0    # >2 == variable internal stretch (asserted_stretch is an avg)
    warp_points: tuple = ()    # (beat, sec) pairs — the real warp curve, for B1 warp eval


@dataclass
class ClipVerdict:
    idx: int
    song: str
    claimed_stem: str
    slot: str
    id_method: str
    mix_start_s: float
    mix_end_s: float
    status: str              # OK | POSITION_MISMATCH | WRONG_AUDIO | UNRESOLVED | NO_AUDIO
    flags: list[str] = field(default_factory=list)
    placed_score: float = 0.0
    best_score: float = 0.0
    best_pos_s: float = 0.0
    pos_error_s: float = 0.0
    detected_pitch: int = 0
    pitch_coarse: int = 0
    note: str = ""


# --------------------------------------------------------------------------- identity
_SLOT_PREFIX = re.compile(r"^\d{3}(?:w\d+)?__")


def song_label(path: str) -> str:
    """Canonical per-slot song identity from a clip path: the stems/<HERE> or
    tracks/<HERE> folder (real acapella candidates live under
    stems/<song>/candidates/vocals/, so the immediate parent is useless), with
    the slot prefix and annotator [bpm key] tags stripped."""
    folder = als_io._stem_folder_name(path) or Path(path).stem
    return als_io.strip_user_tags(_SLOT_PREFIX.sub("", folder)).strip()


def _stem_kind(path: str) -> str:
    b = os.path.basename(path).lower()
    if b.startswith("vocals"):
        return "vocals"
    if b.startswith("instrumental"):
        return "instrumental"
    return "full"


def build_stem_size_index(set_dir: Path) -> dict[tuple[int, str], list[str]]:
    """(file_size, kind) -> [source folder names] for canonical stems + tracks."""
    idx: dict[tuple[int, str], list[str]] = {}
    for f in list((set_dir / "stems").glob("*/vocals.flac")) + \
             list((set_dir / "stems").glob("*/instrumental.flac")):
        try:
            key = (f.stat().st_size, _stem_kind(f.name))
        except OSError:
            continue
        idx.setdefault(key, []).append(str(f))
    return idx


def _orig_size_crc(clip_el) -> tuple[int | None, str | None]:
    fr = clip_el.find(".//FileRef")
    if fr is None:
        return None, None
    sz = fr.find("OriginalFileSize")
    crc = fr.find("OriginalCrc")
    return (int(sz.get("Value")) if sz is not None else None,
            crc.get("Value") if crc is not None else None)


def resolve_source(als_path: str, clip_el, size_index) -> tuple[str | None, str]:
    """(src_path_on_disk, id_method). Prefer the OriginalFileRef path; fall back
    to content-matching the flattened Samples/Imported file by size."""
    p = Path(os.path.expanduser(als_path)) if als_path else None
    flattened = bool(p) and "/Samples/Imported/" in str(p)
    if p and p.is_file() and not flattened:
        return str(p), "orig-path"
    # flattened or missing -> content match by size
    size, _crc = _orig_size_crc(clip_el)
    if size is not None:
        kind = _stem_kind(als_path)
        cands = size_index.get((size, kind), [])
        if len(cands) == 1:
            return cands[0], "size-match"
        if len(cands) > 1:
            return cands[0], "ambiguous"
    if p and p.is_file():           # flattened file exists but identity unknown
        return str(p), "unresolved"
    return None, "unresolved"


# --------------------------------------------------------------------------- parsing
def _mix_duration_s(set_dir: Path) -> float:
    for name in ("mix_instrumental.flac", "mix_vocals.flac", "mix.m4a"):
        f = set_dir / name
        if f.is_file():
            import soundfile as sf
            try:
                return float(sf.info(str(f)).duration)
            except Exception:
                pass
    return 3600.0


def parse_als(als_path: Path, set_dir: Path, size_index) -> tuple[list[ClipFacts], list[dict]]:
    root = als_io.load_als_xml(als_path)
    # mix lane -> arrangement mapper
    mix_track = None
    for tr in root.xpath(".//LiveSet/Tracks/AudioTrack"):
        nm = als_io.track_display_name(tr)
        if nm.startswith("1-mix") or nm.startswith("2-mix") or nm.lower() == "mix":
            mix_track = tr
            break
    if mix_track is None:  # fall back: any lane referencing mix.*
        for tr in root.xpath(".//LiveSet/Tracks/AudioTrack"):
            if tr.xpath(".//AudioClip") and "mix" in (
                    als_io.clip_original_path(tr.xpath(".//AudioClip")[0]) or "").lower():
                mix_track = tr
                break
    mapper = als_io.ArrangementMapper.from_mix_track(
        mix_track, mix_duration_s=_mix_duration_s(set_dir))
    vol_envs = als_io.build_vol_envelopes(root)

    facts: list[ClipFacts] = []
    dropped: list[dict] = []           # clips we couldn't place — logged, never silent
    i = 0
    current_group = ""
    for tr in root.xpath(".//LiveSet/Tracks/*"):
        if tr.tag == "GroupTrack":
            current_group = als_io.track_display_name(tr) or ""
            continue
        if tr.tag != "AudioTrack":
            continue
        nm = als_io.track_display_name(tr)
        if nm.startswith(("1-mix", "2-mix")) or nm.lower() == "mix":
            continue
        vpts = vol_envs.get(als_io.volume_automation_id(tr), [])
        for clip_el in tr.xpath(".//AudioClip"):
            path = als_io.clip_original_path(clip_el)
            cs = clip_el.find("CurrentStart"); ce = clip_el.find("CurrentEnd")
            if cs is None or ce is None:
                dropped.append({"track": nm, "reason": "no CurrentStart/End"})
                continue
            warp = als_io.WarpMarkers.from_clip(clip_el)
            arr_s = float(cs.get("Value")); arr_e = float(ce.get("Value"))
            ms = mapper.arr_to_set_sec(arr_s); me = mapper.arr_to_set_sec(arr_e)
            if ms is None or me is None or me <= ms:
                dropped.append({"track": nm, "arr_start": arr_s, "song": song_label(path),
                                "reason": "arrangement beat is outside every mix-reference "
                                          "warp (parked past the mix) — staged/unaligned clip "
                                          "with no derivable mix position"})
                continue
            anchor = warp.points[0][0] if warp.points else 0.0
            ref_s = warp.beat_to_sec(anchor)
            ref_e = warp.beat_to_sec(anchor + (arr_e - arr_s))
            # unwarped clips play 1:1 — warp markers collapse ref_e onto ref_s, so
            # derive the ref extent from the mix span instead (stretch == 1.0).
            iw = clip_el.find("IsWarped")
            unwarped = iw is not None and iw.get("Value") == "false"
            if unwarped:
                ref_e = ref_s + (me - ms)
            pc = clip_el.find("PitchCoarse")
            pitch = int(pc.get("Value") or 0) if pc is not None else 0
            claimed_stem, _ = als_io.classify_path(path)
            src, method = resolve_source(path, clip_el, size_index)
            song = song_label(src or path) or als_io.track_display_name(tr)
            ref_span = max(ref_e - ref_s, 1e-3); mix_span = max(me - ms, 1e-3)
            facts.append(ClipFacts(
                idx=i, track_name=nm, group=current_group, als_path=path,
                src_path=src, id_method=method, song=song,
                claimed_stem=claimed_stem, slot=als_io.slot_from_path(path) or "",
                mix_start_s=ms, mix_end_s=me, ref_start_s=ref_s, ref_end_s=ref_e,
                pitch_coarse=pitch, asserted_stretch=mix_span / ref_span,
                audible_frac=round(
                    als_io.audible_span(vpts, arr_s, arr_e).fraction, 3),
                unwarped=unwarped, n_warp_markers=len(warp.points),
                warp_points=tuple((round(b, 3), round(s, 3)) for b, s in warp.points),
            ))
            i += 1
    return facts, dropped


# --------------------------------------------------------------------------- verify
def _mix_chroma_for(stem: str, set_dir: Path):
    name = {"acappella": "mix_vocals.flac", "instrumental": "mix_instrumental.flac"}.get(
        stem, "mix_instrumental.flac")
    f = set_dir / name
    if not f.is_file():
        f = set_dir / "mix_instrumental.flac"
    return features.chroma_of(f) if f.is_file() else None


def verify_clip(fact: ClipFacts, set_dir: Path) -> ClipVerdict:
    v = ClipVerdict(idx=fact.idx, song=fact.song, claimed_stem=fact.claimed_stem,
                    slot=fact.slot, id_method=fact.id_method,
                    mix_start_s=fact.mix_start_s, mix_end_s=fact.mix_end_s,
                    pitch_coarse=fact.pitch_coarse, status="OK")
    if fact.audible_frac < 0.1:
        v.status = "MUTED"
        v.note = f"track volume automated to silence (audible {fact.audible_frac:.0%}) — not in mix"
        return v
    if not fact.src_path or not Path(fact.src_path).is_file():
        v.status = "UNRESOLVED"; v.note = "no source audio on disk"; return v

    mixc = _mix_chroma_for(fact.claimed_stem, set_dir)
    srcc = features.chroma_of(Path(fact.src_path))
    if mixc is None or srcc is None or srcc.shape[1] < 4:
        v.status = "NO_AUDIO"; v.note = "missing chroma"; return v

    # source segment as asserted, warped to mix tempo
    r0 = int(round(fact.ref_start_s / FRAME_S))
    seg = srcc[:, r0:r0 + int(round(min(VERIFY_WINDOW_S, fact.ref_end_s - fact.ref_start_s) / FRAME_S))]
    if seg.shape[1] < 8:
        v.status = "NO_AUDIO"; v.note = "ref window too short"; return v
    tmpl = _resample_cols(seg, fact.asserted_stretch)
    if tmpl.shape[1] >= mixc.shape[1]:
        tmpl = tmpl[:, : mixc.shape[1] - 1]

    placed_frame = int(round(fact.mix_start_s / FRAME_S))
    best_score = -2.0; best_frame = 0; best_rot = 0
    placed_score = -2.0; placed_rot = 0
    placed_by_rot = [-2.0] * 12
    for r in range(12):
        curve = _match_curve(np.roll(tmpl, r, axis=0), mixc)
        if curve.size == 0:
            continue
        k = int(curve.argmax())
        if curve[k] > best_score:
            best_score, best_frame, best_rot = float(curve[k]), k, r
        if placed_frame < curve.size:
            ps = float(curve[placed_frame])
            placed_by_rot[r] = ps
            if ps > placed_score:
                placed_score, placed_rot = ps, r
    placed_at_asserted = placed_by_rot[fact.pitch_coarse % 12]

    v.placed_score = round(placed_score, 3)
    v.best_score = round(best_score, 3)
    v.best_pos_s = round(best_frame * FRAME_S, 2)
    v.pos_error_s = round(abs(best_frame * FRAME_S - fact.mix_start_s), 2)
    # pitch is read at the ASSERTED position (the global argmax is often a
    # spurious self-similar peak elsewhere in the mix).
    v.detected_pitch = placed_rot if placed_rot <= 6 else placed_rot - 12

    # classify
    if best_score < SCORE_WEAK:
        v.status = "WRONG_AUDIO"
        v.note = f"source not found in mix (best {best_score:.2f})"
    elif placed_score >= SCORE_OK or v.pos_error_s <= POS_TOL_S:
        v.status = "OK"
    else:
        v.status = "POSITION_MISMATCH"
        v.note = f"strong peak {v.best_score:.2f} at {v.best_pos_s:.1f}s, not placed {fact.mix_start_s:.1f}s"
    # flag pitch only when the asserted transpose matches clearly WORSE than the
    # best rotation at the placed position (chroma pitch is ambiguous when tied).
    if (v.status in ("OK", "POSITION_MISMATCH") and placed_rot != fact.pitch_coarse % 12
            and placed_score >= 0.55 and placed_score - placed_at_asserted > 0.10):
        v.flags.append(f"pitch: als={fact.pitch_coarse} audio≈{v.detected_pitch:+d} "
                       f"(Δ{placed_score - placed_at_asserted:.2f})")
    if fact.id_method in ("ambiguous", "unresolved"):
        v.flags.append(f"identity:{fact.id_method}")
    return v


# --------------------------------------------------------------------------- main
def find_als(set_id: str) -> list[Path]:
    # CANONICAL = the hand-edited "_fast" session; prefer it, then _slow, then rest.
    home = Path.home()
    hits = set((home / "Desktop").glob("*labeling*/*.als")) | \
           set((home / "Desktop").glob(f"*{set_id}*/*.als"))
    return sorted(hits, key=lambda p: (0 if "_fast" in p.name else 1 if "_slow" in p.name else 2,
                                       p.name))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--als", help="explicit .als path (else auto-discover)")
    ap.add_argument("--limit", type=int, default=0, help="verify only first N clips")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    set_dir = next(iter(sorted((Path.home() / "aligning").glob(f"{args.set_id}__*"))), None)
    if set_dir is None:
        sys.exit(f"no ~/aligning folder for {args.set_id}")

    als_path = Path(args.als) if args.als else (find_als(args.set_id)[:1] or [None])[0]
    if als_path is None or not als_path.is_file():
        sys.exit("no .als found; pass --als")
    print(f"set_dir = {set_dir.name}\nals     = {als_path.name}\n")

    size_index = build_stem_size_index(set_dir)
    facts, dropped = parse_als(als_path, set_dir, size_index)
    if dropped:
        print(f"⚠️  {len(dropped)} clips dropped (not placed) — NOT silently ignored:")
        for d in dropped[:12]:
            print(f"     {d.get('song') or '?':36} {d['reason']}")
        print()
    n_unwarped = sum(1 for f in facts if f.unwarped)
    n_varwarp = sum(1 for f in facts if f.n_warp_markers > 2)
    if args.limit:
        facts = facts[: args.limit]
    print(f"parsed {len(facts)} layer clips; verifying against mix audio…\n")

    verdicts: list[ClipVerdict] = []
    for f in facts:
        verdicts.append(verify_clip(f, set_dir))
        if (f.idx + 1) % 25 == 0:
            print(f"  verified {f.idx + 1}/{len(facts)}")

    # summary
    from collections import Counter
    by_status = Counter(v.status for v in verdicts)
    by_id = Counter(f.id_method for f in facts)
    print("\n=== identity resolution ===")
    for k, n in by_id.most_common():
        print(f"  {n:4}  {k}")
    print("\n=== audio-verification status ===")
    for k in ("OK", "MUTED", "POSITION_MISMATCH", "WRONG_AUDIO", "UNRESOLVED", "NO_AUDIO"):
        if by_status.get(k):
            print(f"  {by_status[k]:4}  {k}")
    flagged = [v for v in verdicts if v.flags or v.status != "OK"]
    print(f"\n=== {len(flagged)} clips needing attention (worst first) ===")
    for v in sorted(flagged, key=lambda x: (x.status == "OK", -x.pos_error_s))[:40]:
        fl = (" | " + "; ".join(v.flags)) if v.flags else ""
        print(f"  [{v.status:17}] {v.mix_start_s:7.1f}s {v.song[:42]:42} "
              f"placed={v.placed_score:+.2f} best={v.best_score:+.2f}@{v.best_pos_s:.0f}s{fl}")

    out = Path(args.out) if args.out else config.OUT_ROOT / f"{args.set_id}_als_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nparsing notes: {len(dropped)} dropped, {n_unwarped} unwarped (stretch forced 1.0), "
          f"{n_varwarp} with variable internal warp (full curve stored)")
    out.write_text(json.dumps({
        "set_id": args.set_id, "als": str(als_path),
        "facts": [asdict(f) for f in facts],
        "verdicts": [asdict(v) for v in verdicts],
        "dropped": dropped,
        "summary": {"status": dict(by_status), "identity": dict(by_id),
                    "dropped": len(dropped), "unwarped": n_unwarped, "variable_warp": n_varwarp},
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
