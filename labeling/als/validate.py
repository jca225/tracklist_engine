"""Semantic well-formedness pass over a parsed session — diagnostics as values.

`validate_session` walks the CST once and reports every violated invariant as
a `Diagnostic` with a clip-level location; it never raises, and it never fixes
anything silently. Each check corresponds to a bug class this repo has
actually hit (see docs/als_interpreter_plan.md §3):

- ``warp-*`` — duplicated/cluster warp markers (zero-span ref bug), malformed
  or non-monotonic marker pairs
- ``tempo-*`` — missing/non-positive/malformed tempo automation
- ``pointee-dup`` — duplicate AutomationTarget ids (the Live-crash class)
- ``clip-*`` — clips the extractor would silently skip or mis-handle
- ``gain-out-of-range`` — volume envelope values outside Live's fader range
- ``clip-envelope-ignored`` — clip-local automation the track-fader reader skips
- ``version-unknown`` — session from an untested Live version

CLI edge (fail-fast): ``python -m labeling.als.validate <session.als>`` exits
non-zero iff any error-severity diagnostic is found.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lxml import etree

from labeling.als.read import track_display_name

_KNOWN_MAJOR = {"5"}  # Live 9–12 series schema
_GAIN_MAX = 2.0 + 1e-6  # Live fader top (+6 dB) is ~1.995


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str  # "error" | "warning"
    message: str
    location: str

    def render(self) -> str:
        return f"{self.severity}[{self.code}] {self.location}: {self.message}"


def _clip_location(clip_el: etree._Element) -> str:
    name_el = clip_el.find("Name")
    clip_name = (name_el.get("Value") or "") if name_el is not None else ""
    track = next(
        (
            a
            for a in clip_el.iterancestors()
            if a.tag in ("AudioTrack", "GroupTrack", "MidiTrack")
        ),
        None,
    )
    track_name = track_display_name(track) if track is not None else ""
    return f"track '{track_name}' clip '{clip_name}'"


def _float_or_none(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _check_version(root: etree._Element, out: list[Diagnostic]) -> None:
    if root.tag != "Ableton":
        out.append(
            Diagnostic(
                "version-unknown",
                "error",
                f"root element is <{root.tag}>, not <Ableton>",
                "document",
            )
        )
        return
    major = root.get("MajorVersion")
    creator = root.get("Creator") or "unknown creator"
    if major not in _KNOWN_MAJOR:
        out.append(
            Diagnostic(
                "version-unknown",
                "warning",
                f"untested schema MajorVersion={major!r} ({creator})",
                "document",
            )
        )


def _check_pointee_ids(root: etree._Element, out: list[Diagnostic]) -> None:
    seen: dict[str, int] = {}
    for at in root.iter("AutomationTarget"):
        target_id = at.get("Id")
        if target_id is None:
            continue
        seen[target_id] = seen.get(target_id, 0) + 1
    for target_id, count in seen.items():
        if count > 1:
            out.append(
                Diagnostic(
                    "pointee-dup",
                    "error",
                    f"AutomationTarget Id={target_id} appears {count} times — "
                    "duplicate PointeeIds crash Live on open",
                    "document",
                )
            )


def _check_tempo(root: etree._Element, out: list[Diagnostic]) -> None:
    tempo = root.find(".//MasterTrack//Tempo")
    if tempo is None:
        out.append(
            Diagnostic(
                "tempo-missing", "warning", "no MasterTrack tempo block", "MasterTrack"
            )
        )
        return
    at = tempo.find("AutomationTarget")
    target_id = at.get("Id") if at is not None else None
    if target_id is None:
        out.append(
            Diagnostic(
                "tempo-missing",
                "warning",
                "tempo has no AutomationTarget Id",
                "MasterTrack",
            )
        )
        return
    for env in root.xpath(
        ".//MasterTrack//AutomationEnvelopes/Envelopes/AutomationEnvelope"
    ):
        pid = env.find("EnvelopeTarget/PointeeId")
        if pid is None or pid.get("Value") != target_id:
            continue
        for fe in env.xpath(".//FloatEvent"):
            t = _float_or_none(fe.get("Time"))
            v = _float_or_none(fe.get("Value"))
            if t is None or v is None:
                out.append(
                    Diagnostic(
                        "tempo-malformed",
                        "error",
                        f"tempo FloatEvent Time={fe.get('Time')!r} Value={fe.get('Value')!r} not finite floats",
                        "MasterTrack tempo envelope",
                    )
                )
            elif v <= 0:
                out.append(
                    Diagnostic(
                        "tempo-nonpositive",
                        "error",
                        f"tempo breakpoint at beat {t} has BPM {v} <= 0",
                        "MasterTrack tempo envelope",
                    )
                )


def _check_clips(root: etree._Element, out: list[Diagnostic]) -> None:
    for clip_el in root.iter("AudioClip"):
        loc = _clip_location(clip_el)
        cs = clip_el.find("CurrentStart")
        ce = clip_el.find("CurrentEnd")
        ls = clip_el.find(".//Loop/LoopStart")
        le = clip_el.find(".//Loop/LoopEnd")
        if cs is None or ce is None or ls is None or le is None:
            out.append(
                Diagnostic(
                    "clip-incomplete",
                    "warning",
                    "missing CurrentStart/CurrentEnd/Loop — extractor skips this clip silently",
                    loc,
                )
            )
            continue
        start = _float_or_none(cs.get("Value"))
        end = _float_or_none(ce.get("Value"))
        if start is None or end is None:
            out.append(
                Diagnostic(
                    "clip-malformed",
                    "error",
                    "CurrentStart/CurrentEnd not finite floats",
                    loc,
                )
            )
        elif end < start:
            out.append(
                Diagnostic(
                    "clip-negative-span",
                    "error",
                    f"arr span [{start}, {end}] is negative",
                    loc,
                )
            )

        pairs: list[tuple[float, float]] = []
        malformed = False
        for w in clip_el.xpath(".//WarpMarker"):
            b = _float_or_none(w.get("BeatTime"))
            s = _float_or_none(w.get("SecTime"))
            if b is None or s is None:
                malformed = True
                continue
            pairs.append((b, s))
        if malformed:
            out.append(
                Diagnostic(
                    "warp-malformed",
                    "error",
                    "WarpMarker with non-finite Beat/SecTime",
                    loc,
                )
            )
        pairs.sort()
        if any(s1 < s0 for (_, s0), (_, s1) in zip(pairs, pairs[1:])):
            out.append(
                Diagnostic(
                    "warp-sec-nonmonotonic",
                    "warning",
                    "SecTime decreases along sorted BeatTimes — beat↔sec map is not a function",
                    loc,
                )
            )
        distinct_beats = len({b for b, _ in pairs})
        if pairs and distinct_beats == 1 and len(pairs) > 1:
            out.append(
                Diagnostic(
                    "warp-duplicate-beats",
                    "warning",
                    "all warp markers share one BeatTime — beat_to_sec degenerates to a constant",
                    loc,
                )
            )


def _check_volume_envelopes(root: etree._Element, out: list[Diagnostic]) -> None:
    vol_ids = {
        at.get("Id")
        for at in root.xpath(".//DeviceChain/Mixer/Volume/AutomationTarget")
        if at.get("Id")
    }
    for env_el in root.xpath(".//AutomationEnvelope"):
        pid = env_el.find(".//PointeeId")
        if pid is None or pid.get("Value") not in vol_ids:
            continue
        for fe in env_el.xpath(".//FloatEvent"):
            v = _float_or_none(fe.get("Value"))
            if v is None:
                out.append(
                    Diagnostic(
                        "gain-malformed",
                        "error",
                        f"volume FloatEvent Value={fe.get('Value')!r} not a finite float",
                        f"volume envelope PointeeId={pid.get('Value')}",
                    )
                )
            elif not (0.0 <= v <= _GAIN_MAX):
                out.append(
                    Diagnostic(
                        "gain-out-of-range",
                        "warning",
                        f"volume value {v} outside Live's fader range [0, 2]",
                        f"volume envelope PointeeId={pid.get('Value')}",
                    )
                )


def _check_clip_envelopes(root: etree._Element, out: list[Diagnostic]) -> None:
    """Flag clip-LOCAL automation the track-fader reader ignores.

    GT audibility is read from the TRACK fader (``Mixer/Volume``). A per-clip
    automation envelope — gain/pan/transpose drawn INSIDE the clip, stored in
    the clip's own ``<Envelopes><Envelopes>`` block — is invisible to that
    reader, so a clip faded via a clip envelope would export as fully audible.
    No real labeling session uses one (annotators ride the track fader); this
    fence fires only if that assumption ever breaks, so the gap can't pass
    silently. Empty ``<Envelopes><Envelopes/></Envelopes>`` scaffolding (on
    every real clip) carries no FloatEvent and does not trip it.
    """
    for clip_el in root.iter("AudioClip"):
        inner = clip_el.find("Envelopes/Envelopes")
        if inner is not None and inner.xpath(".//FloatEvent"):
            out.append(
                Diagnostic(
                    "clip-envelope-ignored",
                    "warning",
                    "clip carries a clip-local automation envelope the "
                    "track-fader reader ignores — per-clip gain/pan is not "
                    "captured in ground truth",
                    _clip_location(clip_el),
                )
            )


_CHECKS = (
    _check_version,
    _check_pointee_ids,
    _check_tempo,
    _check_clips,
    _check_volume_envelopes,
    _check_clip_envelopes,
)


def validate_session(root: etree._Element) -> list[Diagnostic]:
    """All diagnostics for a session tree. Never raises — a check that blows
    up on hostile input becomes a ``validator-error`` diagnostic itself."""
    out: list[Diagnostic] = []
    for check in _CHECKS:
        try:
            check(root, out)
        except Exception as exc:  # noqa: BLE001 — the whole point is totality
            out.append(
                Diagnostic(
                    "validator-error",
                    "error",
                    f"{check.__name__} crashed: {type(exc).__name__}: {exc}",
                    "document",
                )
            )
    return out


def has_errors(diags: list[Diagnostic]) -> bool:
    return any(d.severity == "error" for d in diags)


def main() -> None:
    import sys
    from pathlib import Path

    from labeling.als.cst import load_als_xml

    if len(sys.argv) != 2:
        sys.exit("usage: python -m labeling.als.validate <session.als>")
    als_path = Path(sys.argv[1]).expanduser()
    diags = validate_session(load_als_xml(als_path))
    for d in diags:
        print(d.render())
    if has_errors(diags):
        sys.exit(1)
    print(f"OK — {len(diags)} warning(s)" if diags else "OK — clean")


if __name__ == "__main__":
    main()
