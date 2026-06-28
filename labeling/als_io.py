"""Ableton `.als` parsing helpers for ground-truth export.

Uses `lxml` (Py3.14 venv lacks working stdlib expat). Always re-read the
`.als` from disk — never cache a parse across runs.
"""

from __future__ import annotations

import gzip
import html
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from lxml import etree

from core.identity import normalize_stem

_USER_TAG_PATTERN = re.compile(
    r"\[\s*\d+\s*bpm\b|\[no-features\]",
    re.IGNORECASE,
)
_SLOT_FROM_PATH = re.compile(r"(?:^|[/\\])(\d{3}(?:w\d+)?)__")
_BRACKET_TAG = re.compile(r"\s*\[[^\]]*\]\s*")


@dataclass(frozen=True)
class WarpMarkers:
    points: tuple[tuple[float, float], ...]  # (beat, sec)

    @classmethod
    def from_clip(cls, clip: etree._Element) -> WarpMarkers:
        pts = sorted(
            (float(w.get("BeatTime")), float(w.get("SecTime")))
            for w in clip.xpath(".//WarpMarker")
        )
        return cls(points=tuple(pts))

    def beat_to_sec(self, beat: float) -> float:
        if not self.points:
            return beat
        if len(self.points) == 1:
            return self.points[0][1]
        pts = self.points
        if beat <= pts[0][0]:
            # extrapolate before the first marker — use the first marker and the
            # nearest one with a DISTINCT beat for the slope (duplicated/clustered
            # warp markers, e.g. Aftershock's 2 pairs at beats 0 & 0.03125, would
            # otherwise give b1==b0 → a clamped, zero-span ref).
            b0, s0 = pts[0]
            b1, s1 = next(((b, s) for b, s in pts if b > b0), (b0, s0))
        elif beat >= pts[-1][0]:
            b1, s1 = pts[-1]
            b0, s0 = next(((b, s) for b, s in reversed(pts) if b < b1), (b1, s1))
        else:
            for i in range(len(pts) - 1):
                if pts[i][0] <= beat <= pts[i + 1][0]:
                    b0, s0 = pts[i]
                    b1, s1 = pts[i + 1]
                    break
            else:
                b0, s0 = pts[-2]
                b1, s1 = pts[-1]
        if b1 == b0:
            return s0
        return s0 + (beat - b0) / (b1 - b0) * (s1 - s0)


@dataclass(frozen=True)
class MixClipSpan:
    arr_start: float
    arr_end: float
    loop_start: float
    warp: WarpMarkers

    def arr_to_set_sec(self, arr: float) -> float:
        # 1-mix clips are unwarped with markers whose beat 0 == the clip's
        # LEFT EDGE (first marker sec == loop_start sec) — so the map is
        # simply beat_to_sec(arr - arr_start). The old version added the
        # first marker's beat as an anchor: harmless when that beat is 0
        # (clips 1/3 of the BB12 fast project) but clips 2/4 carry markers
        # extending BEFORE the clip (anchor beats -41.5 / -724), which
        # shifted every late-set GT time ~430 s early (found 2026-06-11).
        # NOTE loop values on unwarped clips are SECONDS, marker beats are
        # clip-relative — do not mix the domains.
        return self.warp.beat_to_sec(arr - self.arr_start)


@dataclass(frozen=True)
class ArrangementMapper:
    """Map Ableton arrangement beats → mix file seconds via 1-mix warp spans."""

    spans: tuple[MixClipSpan, ...]
    mix_duration_s: float

    @classmethod
    def from_mix_track(
        cls, mix_track: etree._Element, *, mix_duration_s: float
    ) -> ArrangementMapper:
        spans: list[MixClipSpan] = []
        for clip in mix_track.xpath(".//AudioClip"):
            spans.append(
                MixClipSpan(
                    arr_start=float(clip.find("CurrentStart").get("Value")),
                    arr_end=float(clip.find("CurrentEnd").get("Value")),
                    loop_start=float(clip.find(".//Loop/LoopStart").get("Value")),
                    warp=WarpMarkers.from_clip(clip),
                )
            )
        spans.sort(key=lambda s: s.arr_start)
        return cls(spans=tuple(spans), mix_duration_s=mix_duration_s)

    @property
    def arr_min(self) -> float:
        return self.spans[0].arr_start if self.spans else 0.0

    @property
    def arr_max(self) -> float:
        return self.spans[-1].arr_end if self.spans else 0.0

    def arr_to_set_sec(self, arr: float) -> float | None:
        for span in self.spans:
            if span.arr_start <= arr <= span.arr_end + 1e-3:
                return span.arr_to_set_sec(arr)
        # Bridge short gaps between contiguous mix clips.
        for left, right in zip(self.spans, self.spans[1:]):
            if left.arr_end < arr < right.arr_start:
                left_sec = left.arr_to_set_sec(left.arr_end)
                right_sec = right.arr_to_set_sec(right.arr_start)
                frac = (arr - left.arr_end) / (right.arr_start - left.arr_end)
                return left_sec + frac * (right_sec - left_sec)
        return None


def parse_master_tempo(root: etree._Element) -> tuple[tuple[float, float], ...]:
    """Master-track tempo automation as sorted ``(beat, bpm)`` breakpoints.

    The newer alignment convention leaves the ``1-mix`` clip *unwarped* and
    encodes the mix's (varying) tempo as explicit master-tempo automation, so
    arrangement-beats map to seconds by integrating this curve — not via the
    clip's warp markers. Ableton represents a tempo *step* as two FloatEvents at
    the same Time; the integrator treats zero-width segments as instantaneous.
    Sentinel "before-start" times (large negative) are clamped to beat 0.
    """
    tempo = root.find(".//MasterTrack//Tempo")
    if tempo is None:
        return ()
    at = tempo.find("AutomationTarget")
    target_id = at.get("Id") if at is not None else None
    pts: list[tuple[float, float]] = []
    if target_id is not None:
        for env in root.xpath(
            ".//MasterTrack//AutomationEnvelopes/Envelopes/AutomationEnvelope"
        ):
            pid = env.find("EnvelopeTarget/PointeeId")
            if pid is None or pid.get("Value") != target_id:
                continue
            for fe in env.xpath(".//FloatEvent"):
                t = fe.get("Time")
                v = fe.get("Value")
                if t is None or v is None:
                    continue
                pts.append((max(0.0, float(t)), float(v)))
    if not pts:
        manual = tempo.find("Manual")
        if manual is not None and manual.get("Value"):
            pts.append((0.0, float(manual.get("Value"))))
    pts.sort(key=lambda p: p[0])
    return tuple(pts)


def tempo_beat_to_sec(pts: tuple[tuple[float, float], ...], beat: float) -> float:
    """Integrate ``60/bpm`` over a piecewise-linear tempo curve → seconds.

    Between consecutive breakpoints Ableton ramps tempo linearly, so the exact
    integral of 60/bpm over a linear ramp v0→v1 is
    ``60 * dbeat / (v1 - v0) * ln(v1 / v0)`` (and ``60 * dbeat / v0`` when flat).
    """
    if not pts:
        return beat
    if beat <= pts[0][0]:
        return beat * 60.0 / pts[0][1]
    sec = 0.0
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if beat <= b0:
            return sec
        if b1 <= b0:
            continue  # step (duplicate Time) — zero-width, instantaneous jump
        e = min(beat, b1)
        v_e = v0 + (v1 - v0) * ((e - b0) / (b1 - b0))
        if abs(v_e - v0) < 1e-9:
            sec += 60.0 * (e - b0) / v0
        else:
            sec += 60.0 * (e - b0) / (v_e - v0) * math.log(v_e / v0)
        if beat <= b1:
            return sec
    return sec + (beat - pts[-1][0]) * 60.0 / pts[-1][1]


@dataclass(frozen=True)
class TempoArrangementMapper:
    """Map arrangement-beats → mix-seconds via master-tempo automation.

    For the unwarped-mix convention: mix-second 0 is anchored at the ``1-mix``
    clip's left edge (its ``CurrentStart``), and any arrangement beat maps
    through the integrated tempo curve. Duck-types ``ArrangementMapper`` so the
    export uses it interchangeably."""

    tempo_pts: tuple[tuple[float, float], ...]
    anchor_beat: float
    content_offset_s: float
    mix_duration_s: float
    _anchor_sec: float

    @classmethod
    def from_root(
        cls,
        root: etree._Element,
        mix_track: etree._Element,
        *,
        mix_duration_s: float,
    ) -> TempoArrangementMapper | None:
        pts = parse_master_tempo(root)
        if not pts:
            return None
        clips = mix_track.xpath(".//AudioClip")
        if not clips:
            return None
        clip = clips[0]
        anchor = float(clip.find("CurrentStart").get("Value"))
        loop_el = clip.find(".//Loop/LoopStart")
        # unwarped clips carry loop values in SECONDS (see MixClipSpan note);
        # tiny float noise (~3e-15) rounds to 0.
        content = float(loop_el.get("Value")) if loop_el is not None else 0.0
        return cls(
            tempo_pts=pts,
            anchor_beat=anchor,
            content_offset_s=content,
            mix_duration_s=mix_duration_s,
            _anchor_sec=tempo_beat_to_sec(pts, anchor),
        )

    @property
    def arr_min(self) -> float:
        return self.anchor_beat

    @property
    def arr_max(self) -> float:
        return self.tempo_pts[-1][0] if self.tempo_pts else self.anchor_beat

    def arr_to_set_sec(self, arr: float) -> float | None:
        if arr < self.anchor_beat - 1e-3:
            return None  # before mix-second 0
        return (
            tempo_beat_to_sec(self.tempo_pts, arr)
            - self._anchor_sec
            + self.content_offset_s
        )


def select_arrangement_mapper(
    root: etree._Element,
    mix_track: etree._Element,
    *,
    mix_duration_s: float,
    label_arr_max: float,
) -> ArrangementMapper | TempoArrangementMapper:
    """Pick the arrangement→mix-seconds map for a session.

    Default to the clip-warp ``ArrangementMapper`` (warped-mix convention, e.g.
    BB12). Fall back to the master-tempo mapper only when the clip-warp domain
    fails to cover the labeled clips — i.e. the unwarped-mix / varying-BPM
    convention where the mix clip is a stub. This keeps existing warped sessions
    bit-identical while supporting the new convention."""
    clip_mapper = ArrangementMapper.from_mix_track(
        mix_track, mix_duration_s=mix_duration_s
    )
    if clip_mapper.spans and clip_mapper.arr_max + 1.0 >= label_arr_max:
        return clip_mapper
    tempo_mapper = TempoArrangementMapper.from_root(
        root, mix_track, mix_duration_s=mix_duration_s
    )
    return tempo_mapper if tempo_mapper is not None else clip_mapper


MUTE_THR = 0.05  # track-volume below this is effectively silent (≈ -26 dB)


@dataclass(frozen=True)
class AudibleSpan:
    """Audible portion of a clip's arrangement span (from volume automation)."""

    fraction: float
    arr_start: float
    arr_end: float


@dataclass(frozen=True)
class ParsedClip:
    group_name: str
    track_name: str
    path: str
    arr_start: float
    arr_end: float
    loop_start: float
    loop_end: float
    pitch_coarse: int
    pitch_fine: int
    warp: WarpMarkers
    vol_points: tuple[tuple[float, float], ...] = ()

    @property
    def content_beat_start(self) -> float:
        return self.loop_start

    @property
    def content_beat_end(self) -> float:
        return self.loop_start + (self.arr_end - self.arr_start)

    def ref_start_s(self) -> float:
        # Content at the clip's (possibly trimmed) left edge — loop_start
        # through the warp map. The old anchor-based version returned the
        # FIRST WARP MARKER's position (~file start), so every trimmed clip
        # exported ref_start≈0; the aligner head trained on those labels
        # learned to predict ~0 ref offsets (found 2026-06-11 when the
        # matched-filter detector disagreed with GT at peak 0.99-1.00 and
        # loop_start mapped exactly to the detector's answer).
        return self.warp.beat_to_sec(self.content_beat_start)

    def ref_end_s(self) -> float:
        return self.warp.beat_to_sec(self.content_beat_end)


def _find_mix_splice_beat(
    mapper: ArrangementMapper,
    arr_lo: float,
    arr_hi: float,
) -> float | None:
    """Return the earliest arrangement beat in (arr_lo, arr_hi] where mix-sec jumps back."""
    sec_lo = mapper.arr_to_set_sec(arr_lo)
    sec_hi = mapper.arr_to_set_sec(arr_hi)
    if sec_lo is None or sec_hi is None or sec_hi >= sec_lo:
        return None
    while arr_hi - arr_lo > 1e-4:
        mid = (arr_lo + arr_hi) / 2.0
        sec_mid = mapper.arr_to_set_sec(mid)
        if sec_mid is None:
            return arr_hi
        if sec_mid < sec_lo:
            arr_hi = mid
        else:
            arr_lo = mid
    return arr_hi


def _split_monotonic_arr_interval(
    clip: ParsedClip,
    mapper: ArrangementMapper,
    arr_lo: float,
    arr_hi: float,
) -> tuple[ParsedClip, ...]:
    sec_lo = mapper.arr_to_set_sec(arr_lo)
    sec_hi = mapper.arr_to_set_sec(arr_hi)
    if sec_lo is not None and sec_hi is not None and sec_hi >= sec_lo:
        return (
            replace(
                clip,
                arr_start=arr_lo,
                arr_end=arr_hi,
                loop_start=clip.loop_start + (arr_lo - clip.arr_start),
            ),
        )
    splice = _find_mix_splice_beat(mapper, arr_lo, arr_hi)
    if splice is None or splice <= arr_lo + 1e-6 or splice >= arr_hi - 1e-6:
        return (
            replace(
                clip,
                arr_start=arr_lo,
                arr_end=arr_hi,
                loop_start=clip.loop_start + (arr_lo - clip.arr_start),
            ),
        )
    left_end = splice - 1e-4
    if left_end <= arr_lo + 1e-6:
        return _split_monotonic_arr_interval(clip, mapper, splice, arr_hi)
    return (
        *_split_monotonic_arr_interval(clip, mapper, arr_lo, left_end),
        *_split_monotonic_arr_interval(clip, mapper, splice, arr_hi),
    )


def split_clip_at_mix_span_edges(
    clip: ParsedClip,
    mapper: ArrangementMapper,
) -> tuple[ParsedClip, ...]:
    """Split a layer clip when mix-second mapping jumps at a ``1-mix`` splice."""
    parts = _split_monotonic_arr_interval(clip, mapper, clip.arr_start, clip.arr_end)
    return tuple(p for p in parts if p.arr_end - p.arr_start > 1e-6) or (clip,)


@dataclass(frozen=True)
class ManifestSlot:
    slot_label: str
    track_id: str | None
    display: str
    local_path: str = ""


@dataclass(frozen=True)
class ManifestIndex:
    by_slot: dict[str, ManifestSlot]
    by_path: dict[str, ManifestSlot]
    rows: tuple[ManifestSlot, ...]


def strip_user_tags(name: str) -> str:
    return _BRACKET_TAG.sub("", name).strip()


def slot_from_path(path: str) -> str | None:
    m = _SLOT_FROM_PATH.search(path)
    return m.group(1) if m else None


def clip_original_path(clip: etree._Element) -> str:
    ps = clip.xpath(".//SourceContext//OriginalFileRef//Path")
    if not ps:
        ps = clip.xpath(".//Path")
    if not ps:
        return ""
    return html.unescape(ps[0].get("Value") or "")


def track_display_name(track_el: etree._Element) -> str:
    for tag in ("EffectiveName", "Name", "UserName"):
        n = track_el.find(f".//{tag}")
        if n is not None and n.get("Value"):
            return n.get("Value")
    return ""


def load_als_xml(als_path: Path) -> etree._Element:
    raw = gzip.decompress(als_path.read_bytes())
    return etree.fromstring(raw)


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    return str(Path(path.replace("\\", "/")).expanduser())


def _stem_folder_name(path: str) -> str | None:
    """Return the ``tracks/`` or ``stems/`` child folder name, if any."""
    parts = Path(path.replace("\\", "/")).parts
    for idx, part in enumerate(parts):
        if part in ("tracks", "stems") and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def build_manifest_index(manifest_path: Path) -> ManifestIndex:
    payload = json.loads(manifest_path.read_text())
    by_slot: dict[str, ManifestSlot] = {}
    by_path: dict[str, ManifestSlot] = {}
    rows: list[ManifestSlot] = []
    for row in payload.get("tracks") or []:
        local_path = str(row.get("local_path") or "")
        slot = str(row.get("label") or "").strip() or (slot_from_path(local_path) or "")
        if not slot and not local_path:
            continue
        artist = str(row.get("artist") or "").strip()
        title = str(row.get("title") or "").strip()
        version = row.get("version_tag")
        display = f"{artist} - {title}"
        if version:
            display = f"{display} ({version})"
        slot_row = ManifestSlot(
            slot_label=slot,
            track_id=str(row.get("track_id") or "").strip() or None,
            display=display,
            local_path=local_path,
        )
        rows.append(slot_row)
        if slot:
            by_slot[slot] = slot_row
        if local_path:
            by_path[_normalize_path(local_path)] = slot_row
    return ManifestIndex(by_slot=by_slot, by_path=by_path, rows=tuple(rows))


def match_manifest_for_path(path: str, manifest: ManifestIndex) -> ManifestSlot | None:
    """Exact manifest row for an ALS clip path (file or same stems folder only).

    No label guessing — the ALS path is canonical; manifest is a pull inventory
    used only when the clip points at the exact file (or stem tree) we synced.
    """
    norm = _normalize_path(path)
    if norm in manifest.by_path:
        return manifest.by_path[norm]

    folder = _stem_folder_name(path)
    if folder:
        for row in manifest.rows:
            if row.local_path and _stem_folder_name(row.local_path) == folder:
                return row

    for row in manifest.rows:
        if not row.local_path:
            continue
        stem_root = (
            _normalize_path(row.local_path)
            .replace("/tracks/", "/stems/")
            .rsplit(".", 1)[0]
        )
        if norm.startswith(stem_root + "/"):
            return row

    return None


def _filename_stem_marker(fname: str) -> str | None:
    """Explicit stem qualifier in a master filename, e.g. ``... (Acappella).m4a``
    or ``... (Instrumental Mix).m4a``. A downloaded acappella/instrumental master
    lives in ``tracks/`` too — the qualifier, not the folder, names the stem."""
    if "acappella" in fname or "acapella" in fname:
        return "acappella"
    if "instrumental" in fname:
        return "instrumental"
    return None


def classify_path(path: str) -> tuple[str, str]:
    """Return (claimed_stem, ref_source) from the clip's referenced AUDIO FILE.

    The ``.als`` is the canonical stem oracle: the file the human placed decides
    the stem, in precedence order — Demucs stems and candidate downloads are
    unambiguous; a master is ``regular`` UNLESS its filename carries an explicit
    ``(Acappella)`` / ``(Instrumental)`` qualifier.

    The folder is NOT authoritative: the old code returned ``regular`` for
    everything under ``/tracks/`` *before* reading the filename, silently
    dropping the stem of every ``tracks/... (Acappella).m4a`` master (45 BB12 GT
    rows landed as untagged-regular, incl. the real ``Bad Day (Acappella)``).
    See ``test_classify_path_tracks_master_stem_marker``.
    """
    p = path.replace("\\", "/").lower()
    fname = p.rsplit("/", 1)[-1]

    # 1. Demucs separated stems — unambiguous, regardless of parent folder name.
    if p.endswith("/vocals.flac"):
        return "acappella", "demucs"
    if p.endswith("/instrumental.flac"):
        return "instrumental", "demucs"
    # 2. Downloaded candidate stems.
    if "/candidates/vocals/" in p:
        return "acappella", "online_candidate"
    if "/candidates/instrumental/" in p:
        return "instrumental", "online_candidate"
    if "/candidates/" in p:
        if "instrumental" in fname:
            return "instrumental", "online_candidate"
        return "acappella", "online_candidate"
    # 3. Phase-cancel extractions.
    if "/phase_cancel/" in p or "phase_cancel" in p:
        if "vocals" in p or "acap" in p:
            return "acappella", "phase_cancel"
        return "instrumental", "phase_cancel"
    # 4. Master file (tracks/ or anywhere else): the filename qualifier is the
    #    oracle; default regular. Version tags like (Remix)/(Rework) do NOT flip
    #    the stem.
    marker = _filename_stem_marker(fname)
    if marker:
        return marker, "reference"
    return "regular", "reference"


def display_from_path(path: str) -> str:
    """Human label inferred from an aligning-folder path (filename or parent dir)."""
    p = Path(path.replace("\\", "/"))
    name = p.name
    if name in ("vocals.flac", "instrumental.flac"):
        name = p.parent.name
        name = re.sub(r"^\d+(?:w\d+)?__", "", name)
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        else:
            name = Path(name).stem
    elif name.startswith("cand") and "__" in name:
        name = name.split("__", 1)[1]
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        else:
            name = Path(name).stem
    else:
        name = re.sub(r"^\d+(?:w\d+)?__", "", name)
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        else:
            name = Path(name).stem
    return strip_user_tags(name)


def labels_overlap(left: str, right: str, *, min_tokens: int = 2) -> bool:
    """True when two display labels share enough distinctive tokens."""

    def _tokens(label: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", label.lower())
        return {w for w in cleaned.split() if len(w) > 2}

    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    shared = a & b
    if len(shared) >= min_tokens:
        return True
    shorter = min(len(a), len(b))
    return shorter > 0 and len(shared) / shorter >= 0.4


def build_vol_envelopes(root: etree._Element) -> dict[str, list[tuple[float, float]]]:
    """PointeeId -> sorted (arr-beat, value) breakpoints for volume automation."""
    envs: dict[str, list[tuple[float, float]]] = {}
    for env_el in root.xpath(".//AutomationEnvelope"):
        pid = env_el.find(".//PointeeId")
        if pid is None:
            continue
        envs[pid.get("Value")] = sorted(
            (max(float(fe.get("Time")), -1e6), float(fe.get("Value")))
            for fe in env_el.xpath(".//FloatEvent")
        )
    return envs


def volume_automation_id(track_el: etree._Element) -> str | None:
    at = track_el.find(".//DeviceChain/Mixer/Volume/AutomationTarget")
    return at.get("Id") if at is not None else None


def envelope_value(
    pts: tuple[tuple[float, float], ...] | list[tuple[float, float]], x: float
) -> float:
    if not pts:
        return 1.0
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        (b0, v0), (b1, v1) = pts[i], pts[i + 1]
        if b0 <= x <= b1:
            return v0 if b1 == b0 else v0 + (x - b0) / (b1 - b0) * (v1 - v0)
    return pts[-1][1]


def audible_span(
    pts: tuple[tuple[float, float], ...],
    arr_lo: float,
    arr_hi: float,
    *,
    thr: float = MUTE_THR,
    n: int = 60,
) -> AudibleSpan:
    """Fraction of [arr_lo, arr_hi] where track volume exceeds the mute floor."""
    if not pts or arr_hi <= arr_lo:
        return AudibleSpan(1.0, arr_lo, arr_hi)
    step = (arr_hi - arr_lo) / max(n - 1, 1)
    audible = 0
    arr_a = arr_hi
    arr_b = arr_lo
    t = arr_lo
    for _ in range(n):
        if envelope_value(pts, t) > thr:
            audible += 1
            arr_a = min(arr_a, t)
            arr_b = max(arr_b, t)
        t += step
    frac = audible / n
    if frac == 0:
        return AudibleSpan(0.0, arr_lo, arr_lo)
    if frac >= 1.0 - 1e-9:
        return AudibleSpan(1.0, arr_lo, arr_hi)
    return AudibleSpan(frac, arr_a, arr_b)


def clip_gain_breakpoints(
    pts: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    arr_lo: float,
    arr_hi: float,
) -> list[tuple[float, float]]:
    """Volume breakpoints (arr-beat, linear-gain) ACROSS one clip's span.

    The exact piecewise-linear fader curve the DJ rode over [arr_lo, arr_hi]:
    every automation breakpoint strictly inside the span, bracketed by
    interpolated values at the two endpoints so the curve is closed and
    self-contained. With no automation the track plays at unity, so we return
    a flat [(lo, 1.0), (hi, 1.0)]. Gain is Ableton's linear Mixer/Volume value
    (1.0 = unity / 0 dB; the mute floor is `MUTE_THR`)."""
    if arr_hi <= arr_lo:
        return [(arr_lo, envelope_value(pts, arr_lo))]
    inner = [(b, v) for (b, v) in pts if arr_lo < b < arr_hi]
    curve = [(arr_lo, envelope_value(pts, arr_lo))]
    curve.extend(inner)
    curve.append((arr_hi, envelope_value(pts, arr_hi)))
    return curve


def audible_from_curve(
    curve: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    *,
    thr: float = MUTE_THR,
    n: int = 200,
) -> tuple[float, float | None, float | None]:
    """(fraction, first_audible_x, last_audible_x) of a gain curve above mute.

    The single source of truth for `audible_frac` / `audible_start` /
    `audible_end`: integrating ONE curve guarantees the three agree (the old
    per-field `min`/`max` merge let a muted sibling clip zero the fraction while
    the window stayed populated — slots 066/112). x is whatever domain the curve
    is in (arr-beats or set-seconds); the caller chooses."""
    if not curve:
        return 1.0, None, None
    if len(curve) == 1:
        x, g = curve[0]
        return (1.0, x, x) if g > thr else (0.0, None, None)
    lo, hi = curve[0][0], curve[-1][0]
    if hi <= lo:
        return 1.0, lo, hi
    step = (hi - lo) / (n - 1)
    above = 0
    start: float | None = None
    end: float | None = None
    x = lo
    for _ in range(n):
        if envelope_value(curve, x) > thr:
            above += 1
            start = x if start is None else start
            end = x
        x += step
    return above / n, start, end


def parse_layer_clips(root: etree._Element) -> list[ParsedClip]:
    vol_envs = build_vol_envelopes(root)
    tracks = root.xpath(".//LiveSet/Tracks/*")
    current_group: str | None = None
    out: list[ParsedClip] = []
    for track_el in tracks:
        if track_el.tag == "GroupTrack":
            current_group = track_display_name(track_el) or None
            continue
        if track_el.tag != "AudioTrack":
            continue
        track_name = track_display_name(track_el)
        if track_name.startswith("1-mix") or track_name.startswith("2-mix"):
            continue
        for clip_el in track_el.xpath(".//AudioClip"):
            path = clip_original_path(clip_el)
            if not path:
                continue
            cs_el = clip_el.find("CurrentStart")
            ce_el = clip_el.find("CurrentEnd")
            ls_el = clip_el.find(".//Loop/LoopStart")
            le_el = clip_el.find(".//Loop/LoopEnd")
            if cs_el is None or ce_el is None or ls_el is None or le_el is None:
                continue
            pc_el = clip_el.find("PitchCoarse")
            pf_el = clip_el.find("PitchFine")
            vol_id = volume_automation_id(track_el)
            vol_pts = tuple(vol_envs.get(vol_id, ())) if vol_id else ()
            out.append(
                ParsedClip(
                    group_name=current_group or "",
                    track_name=track_name,
                    path=path,
                    arr_start=float(cs_el.get("Value")),
                    arr_end=float(ce_el.get("Value")),
                    loop_start=float(ls_el.get("Value")),
                    loop_end=float(le_el.get("Value")),
                    # PitchFine is detune in cents and can be fractional
                    # (e.g. "25.5"); round rather than assume int.
                    pitch_coarse=int(round(float(pc_el.get("Value") or 0)))
                    if pc_el is not None
                    else 0,
                    pitch_fine=int(round(float(pf_el.get("Value") or 0)))
                    if pf_el is not None
                    else 0,
                    warp=WarpMarkers.from_clip(clip_el),
                    vol_points=vol_pts,
                )
            )
    return out


def resolve_identity(
    clip: ParsedClip,
    manifest: ManifestIndex,
) -> tuple[str | None, str | None, str, str]:
    """Return (recording_id, slot_label, display_label, claimed_stem).

    Identity is ALS-canonical: display/stem/slot come from the clip path.
    ``track_id`` is filled only on an exact manifest path match (pull inventory),
    never from scrape slot or title guessing.
    """
    claimed_stem, _ = classify_path(clip.path)
    path_label = display_from_path(clip.path)
    path_slot = slot_from_path(clip.path) or ""

    matched = match_manifest_for_path(clip.path, manifest)
    track_id = matched.track_id if matched is not None else None

    return track_id, path_slot, path_label or clip.track_name, claimed_stem


def tempo_ratio(set_span: float, ref_span: float) -> float | None:
    if set_span <= 0 or ref_span <= 0:
        return None
    return ref_span / set_span


def normalize_stem_value(raw: str) -> str:
    return normalize_stem(raw.strip() or None)


# Ableton's "before-start" sentinel time for an automation's initial value.
_ENV_INIT_TIME = "-63072000"


def write_tempo_envelope(
    root: etree._Element, breakpoints: list[tuple[float, float]]
) -> int:
    """Populate the MasterTrack tempo AutomationEnvelope with (beat, bpm) points.

    Reuses the template's *existing* tempo AutomationTarget + AutomationEnvelope
    (matched by PointeeId), so no new PointeeId is allocated — this is deliberately
    clear of the deep-copy id-duplication path that crashes Live (the seeder's
    strip_automation/renumber_pointee_ids machinery touches copied audio tracks,
    never the MasterTrack). `breakpoints` are (beat_time, bpm), any order; beats
    are arrangement musical time. Returns the number of points written; raises if
    the tempo target/envelope is missing from the template.
    """
    tempo = root.find(".//MasterTrack//Tempo")
    if tempo is None:
        raise ValueError("no MasterTrack/Tempo in document")
    at = tempo.find("AutomationTarget")
    if at is None or at.get("Id") is None:
        raise ValueError("Tempo has no AutomationTarget Id")
    target_id = at.get("Id")
    env = None
    for ae in root.findall(
        ".//MasterTrack//AutomationEnvelopes/Envelopes/AutomationEnvelope"
    ):
        pid = ae.find("EnvelopeTarget/PointeeId")
        if pid is not None and pid.get("Value") == target_id:
            env = ae
            break
    if env is None:
        raise ValueError(f"no tempo AutomationEnvelope (PointeeId={target_id})")
    events = env.find("Automation/Events")
    if events is None:
        raise ValueError("tempo envelope has no Automation/Events")
    for child in list(events):
        events.remove(child)
    pts = sorted(breakpoints)
    first_bpm = pts[0][1] if pts else 120.0
    # leading initial-value event, then one FloatEvent per breakpoint
    init = etree.SubElement(events, "FloatEvent")
    init.set("Id", "0")
    init.set("Time", _ENV_INIT_TIME)
    init.set("Value", f"{first_bpm:.6f}")
    for i, (beat, bpm) in enumerate(pts, start=1):
        fe = etree.SubElement(events, "FloatEvent")
        fe.set("Id", str(i))
        fe.set("Time", f"{max(0.0, beat):.6f}")
        fe.set("Value", f"{bpm:.6f}")
    manual = tempo.find("Manual")
    if manual is not None:
        manual.set("Value", f"{first_bpm:.6f}")
    return len(pts)


def write_locators(root: etree._Element, markers: list[tuple[float, float]]) -> int:
    """Replace the arrangement Locators (markers) with `(beat_time, name)` pairs.

    Clones the document's own <Locator> element when present so the schema matches
    Live exactly (versions differ). Times are arrangement beats. Returns count.
    `name` is a float here only by signature convenience — callers pass
    (beat, label_str); we coerce label to str.
    """
    container = root.find(".//Locators/Locators")
    if container is None:
        outer = root.find(".//Locators")
        if outer is None:
            raise ValueError("no Locators block in document")
        container = outer
    existing = container.findall("Locator")
    proto = existing[0] if existing else None
    for loc in existing:
        container.remove(loc)
    from copy import deepcopy

    for i, (beat, name) in enumerate(sorted(markers, key=lambda m: m[0])):
        if proto is not None:
            el = deepcopy(proto)
        else:
            el = etree.SubElement(container, "Locator")
            for tag in ("LomId", "Time", "Name", "Annotation", "IsSongStart"):
                etree.SubElement(el, tag)
            container.remove(el)
        el.set("Id", str(i))

        def _set(tag: str, val: str, _el=el) -> None:
            e = _el.find(tag)
            if e is None:
                e = etree.SubElement(_el, tag)
            e.set("Value", val)

        _set("Time", f"{max(0.0, beat):.6f}")
        _set("Name", str(name))
        _set("IsSongStart", "false")
        lom = el.find("LomId")
        if lom is not None:
            lom.set("Value", "0")
        container.append(el)
    return len(markers)
