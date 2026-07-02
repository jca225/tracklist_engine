"""Structured records of the `.als` codec — the "AST" side of the grammar.

Frozen dataclasses only; parsing lives in `read`, path/manifest identity in
`identity`, session mutation in `write`.
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree


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
