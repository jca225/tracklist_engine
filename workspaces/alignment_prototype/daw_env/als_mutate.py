"""ALS mutators for daw_env — place/nudge clips + Mode B locators."""

from __future__ import annotations

import gzip
from pathlib import Path

from lxml import etree

from labeling.als import load_als_xml, save_als_xml, write_locators
from workspaces.alignment_prototype.daw_env.session import SpanGeom
from workspaces.alignment_prototype.review.seed_als_from_timeline import rewrite_clip


def _find_track_for_slot(root: etree._Element, slot: str) -> etree._Element | None:
    """Match AudioTrack EffectiveName starting with ``{slot}__`` or exact slot."""
    for t in root.findall(".//LiveSet/Tracks/AudioTrack"):
        name_el = t.find(".//Name/EffectiveName")
        nm = name_el.get("Value") if name_el is not None else ""
        if nm == slot or nm.startswith(f"{slot}__"):
            return t
    return None


def sync_span_to_als(als_path: Path, geom: SpanGeom) -> None:
    """Rewrite arrangement + warp markers for ``geom.slot`` (60 BPM = 1 beat/s)."""
    root = load_als_xml(als_path)
    track = _find_track_for_slot(root, geom.slot)
    if track is None:
        return  # cold session without that track — geometry stays in-memory
    clip = track.find(".//ArrangerAutomation/Events/AudioClip")
    if clip is None:
        return
    # Preserve existing file path / duration from the clip when possible.
    fref = clip.find("SampleRef/FileRef")
    path_el = fref.find("Path") if fref is not None else None
    file_path = (
        Path(path_el.get("Value"))
        if path_el is not None and path_el.get("Value")
        else Path(".")
    )
    sref = clip.find("SampleRef")
    sr = 44100
    file_dur = max(geom.duration_s + geom.ref_start_s, 1.0)
    if sref is not None:
        dsr = sref.find("DefaultSampleRate")
        ddur = sref.find("DefaultDuration")
        if dsr is not None and dsr.get("Value"):
            sr = int(float(dsr.get("Value")))
        if ddur is not None and ddur.get("Value") and sr > 0:
            file_dur = max(float(ddur.get("Value")) / sr, file_dur)
    name_el = clip.find("Name")
    name = name_el.get("Value") if name_el is not None else geom.slot
    color_el = clip.find("Color")
    color = (
        int(color_el.get("Value"))
        if color_el is not None and color_el.get("Value")
        else 5
    )
    ref_end = geom.ref_start_s + geom.duration_s
    rewrite_clip(
        clip,
        name=name or geom.slot,
        color=color,
        arr_start=geom.set_start_s,
        arr_end=geom.set_end_s,
        file_path=file_path,
        file_dur_s=file_dur,
        sample_rate=sr,
        ref_start_s=geom.ref_start_s,
        ref_end_s=ref_end,
        is_warped=True,
    )
    save_als_xml(root, als_path)


def write_agent_propose_locator(als_path: Path, beat_s: float, slot: str) -> None:
    """Stamp a Mode-B locator: ``AGENT PROPOSE <slot>`` at arrangement beat (== seconds @ 60 BPM)."""
    root = load_als_xml(als_path)
    existing: list[tuple[float, str]] = []
    for loc in root.findall(".//Locators/Locators/Locator"):
        t = loc.find("Time")
        n = loc.find("Name")
        if t is not None and n is not None:
            existing.append((float(t.get("Value") or 0), n.get("Value") or ""))
    # Drop prior propose for this slot; keep others.
    keep = [(b, n) for b, n in existing if not n.startswith(f"AGENT PROPOSE {slot}")]
    keep.append((beat_s, f"AGENT PROPOSE {slot}"))
    write_locators(root, [(b, n) for b, n in keep])  # type: ignore[arg-type]
    save_als_xml(root, als_path)


def make_minimal_als(path: Path, *, slot: str = "001") -> Path:
    """Tiny gzipped .als with one warped AudioClip for unit tests (no Live)."""
    # Minimal Ableton-like tree — enough for load_als_xml + rewrite_clip + locators.
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="11" MinorVersion="0" Creator="daw_env_test">
  <LiveSet>
    <Tracks>
      <AudioTrack Id="1">
        <Name>
          <EffectiveName Value="{slot}__test" />
          <UserName Value="{slot}__test" />
        </Name>
        <DeviceChain>
          <MainSequencer>
            <ClipTimeable>
              <ArrangerAutomation>
                <Events>
                  <AudioClip Id="0" Time="10.000000">
                    <CurrentStart Value="10.000000" />
                    <CurrentEnd Value="18.000000" />
                    <Name Value="{slot}__test" />
                    <Color Value="5" />
                    <IsWarped Value="true" />
                    <PitchCoarse Value="0" />
                    <PitchFine Value="0" />
                    <Disabled Value="false" />
                    <Loop>
                      <LoopStart Value="0" />
                      <LoopEnd Value="8.000000" />
                      <StartRelative Value="0" />
                      <LoopOn Value="false" />
                      <OutMarker Value="8.000000" />
                      <HiddenLoopStart Value="0" />
                      <HiddenLoopEnd Value="8.000000" />
                    </Loop>
                    <SampleRef>
                      <FileRef>
                        <Path Value="/tmp/daw_env_test.wav" />
                        <RelativePath Value="/tmp/daw_env_test.wav" />
                        <RelativePathType Value="0" />
                      </FileRef>
                      <DefaultDuration Value="352800" />
                      <DefaultSampleRate Value="44100" />
                    </SampleRef>
                    <WarpMarkers>
                      <WarpMarker Id="0" SecTime="0.000000" BeatTime="0.000000" />
                      <WarpMarker Id="1" SecTime="8.000000" BeatTime="8.000000" />
                    </WarpMarkers>
                  </AudioClip>
                </Events>
              </ArrangerAutomation>
            </ClipTimeable>
          </MainSequencer>
        </DeviceChain>
      </AudioTrack>
    </Tracks>
    <Locators>
      <Locators>
        <Locator Id="0">
          <LomId Value="0" />
          <Time Value="0.000000" />
          <Name Value="SEEDED" />
          <Annotation Value="" />
          <IsSongStart Value="false" />
        </Locator>
      </Locators>
    </Locators>
  </LiveSet>
</Ableton>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(xml.encode("utf-8"))
    return path
