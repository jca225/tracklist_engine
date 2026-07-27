"""Measure the per-clip pitch offset of every GT clip in a labeling .als.

For each reference clip (acappella / instrumental / full track) placed against a
recorded DJ mix, measures how far the track is pitched in the mix relative to its
native rip (``pitch_detune.estimate_cents``), and records the context needed to
tell *why*: the dialed Ableton pitch, native vs local BPM (varispeed test),
Camelot key (harmonic-mixing test), and whether the clip plays solo or layered.

Output is a per-clip CSV consumed by ``lab/corpus_empirics/bb_pitch_detune.py``,
which runs the hypothesis tests and persists headline metrics. The audio work
lives here (Mac-only, reads ~/aligning) so the eda script stays a pure reader.

Run:  venvs/audio/bin/python -m alignment.measure_detune
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np

from labeling.als import cst, read
from labeling.als.models import ParsedClip
from labeling.als.semantics import (
    audible_from_curve,
    clip_gain_breakpoints,
    envelope_value,
    parse_master_tempo,
    select_arrangement_mapper,
)
from alignment.pitch_detune import (
    SR,
    VOCAL_FMAX,
    VOCAL_FMIN,
    estimate_cents,
    load_span,
)

# run from the repo root (python -m alignment.measure_detune)
OUT_CSV = Path("alignment/out/pitch_detune_clips.csv")

SETS = {
    "BB11": (
        "2nvzlh2k",
        "~/aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11/"
        "BB11 align Project/BB11 align.als",
    ),
    "BB12": (
        "1fsnxchk",
        "~/aligning/_backups/20260616_150150/big bootie 12 labeling Project/"
        "big bootie 12 labeling_fast.als",
    ),
}

TAG = re.compile(r"\[(\d+)\s*bpm\s+([0-9]{1,2}[AB])\]", re.I)
SLOT = re.compile(r"(\d{3}w\d+|\d{3})__")
MIN_MIX_SPAN_S = 3.0  # need enough audio for a stable averaged spectrum
MIN_CORR = 0.06  # cross-corr peak below this ⇒ ref not really present → drop


def tag_of(path: str, name: str) -> tuple[float | None, str | None]:
    """(bpm_native, camelot) from a [NNNbpm KK] tag anywhere in path/name."""
    for s in [name, *Path(path).parts]:
        m = TAG.search(s)
        if m:
            return float(m.group(1)), m.group(2).upper()
    return None, None


def slot_of(path: str, name: str) -> str:
    for s in [*Path(path).parts, name]:
        m = SLOT.search(s)
        if m:
            return m.group(1)
    return name  # fall back to the arrangement track name


def stem_of(path: str) -> str:
    b = os.path.basename(path).lower()
    par = os.path.basename(os.path.dirname(path)).lower()
    if "vocal" in b or "vocal" in par or "acap" in path.lower():
        return "acappella"
    if "instrumental" in b or "instrumental" in par:
        return "instrumental"
    return "regular"


def raw_fine_map(root) -> dict[tuple, float]:
    """(path, round(arr_start,3), round(loop_start,3)) -> raw fractional PitchFine.

    The codec rounds PitchFine to int; the ear-calibration angle needs the exact
    dialed value (e.g. 25.5), so re-read it straight off the tree.
    """
    out: dict[tuple, float] = {}
    for clip in root.xpath(".//LiveSet/Tracks/*[self::AudioTrack]//AudioClip"):
        ps = clip.xpath(".//SourceContext//OriginalFileRef//Path") or clip.xpath(
            ".//Path"
        )
        cs = clip.find("CurrentStart")
        ls = clip.find(".//Loop/LoopStart")
        pf = clip.find("PitchFine")
        if not ps or cs is None or ls is None:
            continue
        import html

        path = html.unescape(ps[0].get("Value") or "")
        try:
            key = (
                path,
                round(float(cs.get("Value")), 3),
                round(float(ls.get("Value")), 3),
            )
            out[key] = (
                float(pf.get("Value")) if pf is not None and pf.get("Value") else 0.0
            )
        except (TypeError, ValueError):
            continue
    return out


def camelot_distance(a: str | None, b: str | None) -> int | None:
    """Steps on the Camelot wheel between two keys (0 = identical/perfectly
    compatible). None if either key missing/malformed. Distance combines the
    hour ring (mod 12) and the A/B letter; harmonically-compatible neighbours
    (±1 hour same letter, or same hour other letter) are distance 1."""

    def parse(k):
        m = re.fullmatch(r"(\d{1,2})([AB])", k or "")
        return (int(m.group(1)), m.group(2)) if m else None

    pa, pb = parse(a), parse(b)
    if pa is None or pb is None:
        return None
    ha, la = pa
    hb, lb = pb
    ring = min((ha - hb) % 12, (hb - ha) % 12)
    return ring + (0 if la == lb else 1)


@dataclass
class ClipRow:
    set: str
    set_id: str
    arr_track: str
    slot: str
    stem: str
    coarse: int
    fine_dialed: float
    cents_agg: float
    residual: float
    peak_corr: float
    saturated: int
    bpm_native: float | None
    bpm_local: float | None
    camelot: str | None
    section: str  # solo | layered
    n_cotracks: int
    co_slots: str
    co_camelots: str
    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float
    dur_s: float
    ref_path: str


def _bpm_local(tempo_pts, beat: float) -> float | None:
    if not tempo_pts:
        return None
    return envelope_value(tempo_pts, beat)


def _audible_frac(clip: ParsedClip, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    curve = clip_gain_breakpoints(clip.vol_points, lo, hi)
    frac, _, _ = audible_from_curve(curve)
    return frac


def measure_set(name: str, set_id: str, als_path: str) -> list[ClipRow]:
    root = cst.load_als_xml(Path(os.path.expanduser(als_path)))
    mix_track = next(
        tr
        for tr in root.xpath(".//LiveSet/Tracks/*")
        if read.track_display_name(tr).startswith(("1-mix", "2-mix"))
    )
    mix_path = read.clip_original_path(mix_track.xpath(".//AudioClip")[0])
    clips = read.parse_layer_clips(root)
    tempo_pts = parse_master_tempo(root)
    label_arr_max = max((c.arr_end for c in clips), default=0.0)
    mapper = select_arrangement_mapper(
        root, mix_track, mix_duration_s=1e9, label_arr_max=label_arr_max
    )
    fine_map = raw_fine_map(root)

    print(f"[{name}] loading mix {os.path.basename(mix_path)} ...", flush=True)
    mix_y, _ = librosa.load(mix_path, sr=SR, mono=True)
    print(f"[{name}] mix {mix_y.size / SR:.0f}s; {len(clips)} clips", flush=True)

    ref_cache: dict[str, np.ndarray] = {}

    def ref_full(path: str) -> np.ndarray:
        if path not in ref_cache:
            try:
                y, _ = librosa.load(path, sr=SR, mono=True)
            except Exception:
                y = np.zeros(0, dtype=np.float32)
            ref_cache[path] = y.astype(np.float32)
        return ref_cache[path]

    rows: list[ClipRow] = []
    for i, c in enumerate(clips):
        s0 = mapper.arr_to_set_sec(c.arr_start)
        s1 = mapper.arr_to_set_sec(c.arr_end)
        if s0 is None or s1 is None or s1 - s0 < MIN_MIX_SPAN_S:
            continue
        r0, r1 = c.ref_start_s(), c.ref_end_s()
        if r1 - r0 < MIN_MIX_SPAN_S:
            continue
        # mix span from the preloaded array
        mix_slice = mix_y[int(s0 * SR) : int(s1 * SR)]
        ref_y = ref_full(c.path)
        if ref_y.size == 0:
            continue
        ref_slice = ref_y[int(r0 * SR) : int(r1 * SR)]
        # acappellas are buried under the bed in the mix span → measure on the
        # vocal band so the bass doesn't drag the correlation down (see
        # pitch_detune.estimate_cents); instrumentals/full tracks use full range.
        stem = stem_of(c.path)
        band = (VOCAL_FMIN, VOCAL_FMAX) if stem == "acappella" else (None, None)
        kw = {"fmin": band[0], "fmax": band[1]} if band[0] else {}
        res = estimate_cents(ref_slice, mix_slice, pitch_coarse=c.pitch_coarse, **kw)
        if not res.ok or res.peak_corr < MIN_CORR:
            continue

        bpm_native, camelot = tag_of(c.path, c.track_name)
        bpm_local = _bpm_local(tempo_pts, (c.arr_start + c.arr_end) / 2.0)
        slot = slot_of(c.path, c.track_name)

        # co-play: other-song clips overlapping this clip's arrangement span,
        # audible in the overlap
        co_slots: list[str] = []
        co_cam: list[str] = []
        for d in clips:
            if d is c:
                continue
            ds = slot_of(d.path, d.track_name)
            if ds == slot:
                continue
            lo = max(c.arr_start, d.arr_start)
            hi = min(c.arr_end, d.arr_end)
            if hi - lo < 1.0:  # <~ a beat of overlap doesn't count
                continue
            if _audible_frac(d, lo, hi) < 0.2 or _audible_frac(c, lo, hi) < 0.2:
                continue
            co_slots.append(ds)
            _, dcam = tag_of(d.path, d.track_name)
            if dcam:
                co_cam.append(dcam)
        co_slots = sorted(set(co_slots))
        section = "layered" if co_slots else "solo"

        fine = fine_map.get(
            (c.path, round(c.arr_start, 3), round(c.loop_start, 3)),
            float(c.pitch_fine),
        )
        rows.append(
            ClipRow(
                set=name,
                set_id=set_id,
                arr_track=c.track_name,
                slot=slot,
                stem=stem,
                coarse=c.pitch_coarse,
                fine_dialed=fine,
                cents_agg=round(res.cents_agg, 2),
                residual=round(res.residual, 2),
                peak_corr=round(res.peak_corr, 3),
                saturated=int(res.saturated),
                bpm_native=bpm_native,
                bpm_local=round(bpm_local, 2) if bpm_local else None,
                camelot=camelot,
                section=section,
                n_cotracks=len(co_slots),
                co_slots=";".join(co_slots),
                co_camelots=";".join(sorted(set(co_cam))),
                mix_start_s=round(s0, 2),
                mix_end_s=round(s1, 2),
                ref_start_s=round(r0, 2),
                ref_end_s=round(r1, 2),
                dur_s=round(s1 - s0, 2),
                ref_path=c.path,
            )
        )
        if (i + 1) % 40 == 0:
            print(
                f"[{name}] {i + 1}/{len(clips)} clips scanned, {len(rows)} kept",
                flush=True,
            )
    print(f"[{name}] done: {len(rows)} rows", flush=True)
    return rows


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[ClipRow] = []
    for name, (set_id, als) in SETS.items():
        if not os.path.exists(os.path.expanduser(als)):
            print(f"[{name}] als not found: {als}", file=sys.stderr)
            continue
        all_rows.extend(measure_set(name, set_id, als))
    if not all_rows:
        print("no rows measured", file=sys.stderr)
        return 1
    fields = list(asdict(all_rows[0]).keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))
    print(f"wrote {len(all_rows)} rows -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
