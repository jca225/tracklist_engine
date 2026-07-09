"""Timeline semantics of a session — the codec's evaluators.

The denotation of a session is a family of maps out of arrangement-beats:
→ mix-seconds (the arrangement mappers), → linear gain (the envelope
functions), → audibility (the audible-* functions). Everything here is pure
over already-parsed values except the mapper constructors, which read their
inputs straight off the CST.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from lxml import etree

from labeling.als.models import AudibleSpan, MixClipSpan, ParsedClip, WarpMarkers

MUTE_THR = 0.05  # track-volume below this is effectively silent (≈ -26 dB)


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
            try:
                spans.append(
                    MixClipSpan(
                        arr_start=float(clip.find("CurrentStart").get("Value")),
                        arr_end=float(clip.find("CurrentEnd").get("Value")),
                        loop_start=float(clip.find(".//Loop/LoopStart").get("Value")),
                        warp=WarpMarkers.from_clip(clip),
                    )
                )
            except (AttributeError, TypeError, ValueError):
                continue  # malformed mix clip — validate reports it
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
                try:
                    beat, bpm = max(0.0, float(t)), float(v)
                except ValueError:
                    continue  # malformed event — validate reports tempo-malformed
                if math.isfinite(beat) and math.isfinite(bpm) and bpm > 0:
                    pts.append((beat, bpm))
    if not pts:
        manual = tempo.find("Manual")
        if manual is not None and manual.get("Value"):
            try:
                bpm = float(manual.get("Value"))
            except ValueError:
                bpm = 0.0
            if math.isfinite(bpm) and bpm > 0:
                pts.append((0.0, bpm))
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


def tempo_sec_to_beat(pts: tuple[tuple[float, float], ...], sec: float) -> float:
    """Inverse of :func:`tempo_beat_to_sec` — seconds → arrangement beats.

    THE tempo-breakpoint placement primitive: any event meant to happen at
    mix-second T (a tempo change at a song boundary, a clip start) must be
    written at beat ``tempo_sec_to_beat(pts, T)``. Placing it at ``T`` or at
    ``T·bpm/60`` without integrating the curve before it is the Jun-16 seeder
    bug — every breakpoint after the first tempo change lands wrong and the
    error ripples. Over a linear ramp v0→v1 (slope m per beat) the inverse of
    the ``60/m·ln(v/v0)`` integral is ``b0 + v0·(exp(m·sec/60) − 1)/m``.
    """
    if not pts:
        return sec
    b0, v0 = pts[0]
    first_sec = b0 * 60.0 / v0
    if sec <= first_sec:
        return sec * v0 / 60.0
    acc = first_sec
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if b1 <= b0:
            continue  # zero-width step — instantaneous jump, no time passes
        if abs(v1 - v0) < 1e-9:
            seg = 60.0 * (b1 - b0) / v0
            if sec <= acc + seg:
                return b0 + (sec - acc) * v0 / 60.0
        else:
            m = (v1 - v0) / (b1 - b0)
            seg = 60.0 / m * math.log(v1 / v0)
            if sec <= acc + seg:
                return b0 + v0 * (math.exp(m * (sec - acc) / 60.0) - 1.0) / m
        acc += seg
    b_last, v_last = pts[-1]
    return b_last + (sec - acc) * v_last / 60.0


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
        try:
            anchor = float(clip.find("CurrentStart").get("Value"))
            loop_el = clip.find(".//Loop/LoopStart")
            # unwarped clips carry loop values in SECONDS (see MixClipSpan note);
            # tiny float noise (~3e-15) rounds to 0.
            content = float(loop_el.get("Value")) if loop_el is not None else 0.0
        except (AttributeError, TypeError, ValueError):
            return None  # malformed 1-mix clip — validate reports it
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


def _trim_to_interval(
    clip: ParsedClip,
    mapper: ArrangementMapper,
    arr_lo: float,
    arr_hi: float,
) -> ParsedClip:
    if arr_lo == clip.arr_start and arr_hi == clip.arr_end:
        return clip
    if clip.is_warped:
        return replace(
            clip,
            arr_start=arr_lo,
            arr_end=arr_hi,
            loop_start=clip.loop_start + (arr_lo - clip.arr_start),
        )
    # Unwarped clips index content in SECONDS at 1:1 with wall clock, so the
    # trim advances loop values by elapsed mix-seconds, not arrangement beats.
    # Caveat: across a 1-mix splice mix-sec jumps while the clip's wall clock
    # doesn't — no GT session has an unwarped clip straddling a splice
    # (BB11/BB12 scan 2026-07-02); when unmappable, keep loop values untrimmed.
    s0 = mapper.arr_to_set_sec(clip.arr_start)
    s1 = mapper.arr_to_set_sec(arr_lo)
    s2 = mapper.arr_to_set_sec(arr_hi)
    loop_start = clip.loop_start
    if s0 is not None and s1 is not None and s1 >= s0:
        loop_start = clip.loop_start + (s1 - s0)
    loop_end = clip.loop_end
    if s1 is not None and s2 is not None and s2 >= s1:
        loop_end = loop_start + (s2 - s1)
    return replace(
        clip, arr_start=arr_lo, arr_end=arr_hi, loop_start=loop_start, loop_end=loop_end
    )


def _split_monotonic_arr_interval(
    clip: ParsedClip,
    mapper: ArrangementMapper,
    arr_lo: float,
    arr_hi: float,
) -> tuple[ParsedClip, ...]:
    sec_lo = mapper.arr_to_set_sec(arr_lo)
    sec_hi = mapper.arr_to_set_sec(arr_hi)
    if sec_lo is not None and sec_hi is not None and sec_hi >= sec_lo:
        return (_trim_to_interval(clip, mapper, arr_lo, arr_hi),)
    splice = _find_mix_splice_beat(mapper, arr_lo, arr_hi)
    if splice is None or splice <= arr_lo + 1e-6 or splice >= arr_hi - 1e-6:
        return (_trim_to_interval(clip, mapper, arr_lo, arr_hi),)
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


def tempo_ratio(set_span: float, ref_span: float) -> float | None:
    if set_span <= 0 or ref_span <= 0:
        return None
    return ref_span / set_span
