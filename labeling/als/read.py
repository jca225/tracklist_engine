"""Read side of the `.als` codec: CST → AST extraction.

Pulls typed records (`labeling.als.models`) off a parsed session tree. Loading
the tree itself lives in `cst`; timeline evaluation lives in `semantics`.
"""

from __future__ import annotations

import html

from lxml import etree

from labeling.als.models import ParsedClip, WarpMarkers


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


def build_vol_envelopes(root: etree._Element) -> dict[str, list[tuple[float, float]]]:
    """PointeeId -> sorted (arr-beat, value) breakpoints for volume automation."""
    envs: dict[str, list[tuple[float, float]]] = {}
    for env_el in root.xpath(".//AutomationEnvelope"):
        pid = env_el.find(".//PointeeId")
        if pid is None:
            continue
        pts: list[tuple[float, float]] = []
        for fe in env_el.xpath(".//FloatEvent"):
            try:
                pts.append((max(float(fe.get("Time")), -1e6), float(fe.get("Value"))))
            except (TypeError, ValueError):
                continue  # malformed event — validate reports it
        envs[pid.get("Value")] = sorted(pts)
    return envs


def volume_automation_id(track_el: etree._Element) -> str | None:
    at = track_el.find(".//DeviceChain/Mixer/Volume/AutomationTarget")
    return at.get("Id") if at is not None else None


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
            try:
                clip = ParsedClip(
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
            except (TypeError, ValueError, OverflowError):
                continue  # malformed clip numerics — validate reports clip-malformed
            out.append(clip)
    return out
