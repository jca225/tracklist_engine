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


def _active_fileref(clip: etree._Element) -> etree._Element | None:
    """The clip's live sample reference — the FileRef Live actually loads from,
    NOT the historical `SourceContext/OriginalFileRef` copy. Identity reads must
    use this one (Operation Crush §9)."""
    for fr in clip.xpath(".//SampleRef/FileRef"):
        if not any(a.tag == "SourceContext" for a in fr.iterancestors()):
            return fr
    return None


def clip_content_identity(clip: etree._Element) -> tuple[int | None, int | None]:
    """(OriginalFileSize, OriginalCrc) off the active FileRef — Ableton's own
    record of the exact bytes labeled against. None when the ref omits them."""
    fr = _active_fileref(clip)
    if fr is None:
        return None, None

    def _int(tag: str) -> int | None:
        el = fr.find(f".//{tag}")
        if el is None or el.get("Value") is None:
            return None
        try:
            return int(el.get("Value"))
        except (TypeError, ValueError):
            return None

    return _int("OriginalFileSize"), _int("OriginalCrc")


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


def track_static_gain(track_el: etree._Element) -> float:
    """The static ``Mixer/Volume/Manual`` fader value, linear (1.0 = unity).

    Defaults to unity when absent or malformed — a missing fader element means
    Live left it at default, not that the track is silent.
    """
    man = track_el.find(".//DeviceChain/Mixer/Volume/Manual")
    if man is None:
        return 1.0
    try:
        return float(man.get("Value"))
    except (TypeError, ValueError):
        return 1.0


# Static fader gains at/below this are treated as silence. Kept tiny (a true
# "0" fader, not merely quiet) so audible-but-low GT clips are never dropped;
# a merely-low static fader flows into the gain curve instead, where the
# MUTE_THR floor decides audibility just like an automated fade.
_FADER_SILENCE = 1e-4


def _silence_reason(
    track_el: etree._Element,
    clip_el: etree._Element,
    vol_pts: tuple[tuple[float, float], ...],
    static_gain: float,
) -> str:
    """Why this clip is inaudible, or "" if it plays.

    Three hard-off switches Live records independently of clip extent: the
    Track Activator (``Mixer/Speaker/Manual=false``), per-clip deactivation
    (``AudioClip/Disabled=true``), and a static fader at 0. A clip under any of
    them is silent and must not become a ground-truth span. A merely-low (not
    zero) static fader is NOT dropped here — it rides into the gain curve.
    """
    spk = track_el.find(".//DeviceChain/Mixer/Speaker/Manual")
    if spk is not None and spk.get("Value") == "false":
        return "track-deactivated"
    dis = clip_el.find("Disabled")
    if dis is not None and dis.get("Value") == "true":
        return "clip-disabled"
    # A static fader at 0 silences only when no automation overrides it.
    if not vol_pts and static_gain <= _FADER_SILENCE:
        return "track-fader-zero"
    return ""


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
            iw_el = clip_el.find("IsWarped")
            vol_id = volume_automation_id(track_el)
            vol_pts = tuple(vol_envs.get(vol_id, ())) if vol_id else ()
            static_gain = track_static_gain(track_el)
            file_size, crc = clip_content_identity(clip_el)
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
                    # missing element defaults to warped (pre-Live-8 sets
                    # don't occur here; warped is the 295/301 common case)
                    is_warped=iw_el is None or iw_el.get("Value") == "true",
                    track_gain=static_gain,
                    silence_reason=_silence_reason(
                        track_el, clip_el, vol_pts, static_gain
                    ),
                    file_size=file_size,
                    crc=crc,
                )
            except (TypeError, ValueError, OverflowError):
                continue  # malformed clip numerics — validate reports clip-malformed
            out.append(clip)
    return out
