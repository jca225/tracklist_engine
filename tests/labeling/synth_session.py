"""Synthetic minimal `.als` sessions for codec tests.

Builds the smallest session tree the codec meaningfully operates on: one
master track with a tempo AutomationTarget + envelope, a Locators block, a
1-mix track, and N layer AudioClips with warp markers and volume automation.
Test-only — real sessions are never regenerated from scratch (see
labeling/als/cst.py); this exists so repo tests don't depend on Mac-local
Ableton projects.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass

from lxml import etree

TEMPO_TARGET_ID = "8770"
VOL_TARGET_ID_BASE = 9000


@dataclass(frozen=True)
class LayerSpec:
    """One layer track for a synthetic session (see ``layer_specs``)."""

    name: str
    speaker_on: bool = True
    volume_manual: float = 1.0
    clip_disabled: bool = False
    # Draw a clip-LOCAL automation envelope inside the AudioClip (the per-clip
    # gain/pan the track-fader reader ignores) — used only by the validate fence
    # test. Empty by default, matching every real labeling session.
    clip_envelope: bool = False


def _clip_xml(
    idx: int,
    path: str,
    arr_start: float,
    arr_end: float,
    loop_start: float = 0.0,
    warp_points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (64.0, 32.0)),
    disabled: bool = False,
    clip_envelope: bool = False,
) -> str:
    warps = "".join(
        f'<WarpMarker Id="{i}" SecTime="{s}" BeatTime="{b}"/>'
        for i, (b, s) in enumerate(warp_points)
    )
    loop_end = loop_start + (arr_end - arr_start)
    disabled_el = f'<Disabled Value="{"true" if disabled else "false"}"/>'
    # Live nests clip-local automation under <Envelopes><Envelopes>…; the empty
    # form (real sessions) is <Envelopes><Envelopes/></Envelopes>.
    if clip_envelope:
        envelopes = (
            "<Envelopes><Envelopes><AutomationEnvelope><Automation><Events>"
            '<FloatEvent Id="0" Time="0" Value="1"/>'
            '<FloatEvent Id="1" Time="8" Value="0"/>'
            "</Events></Automation></AutomationEnvelope></Envelopes></Envelopes>"
        )
    else:
        envelopes = "<Envelopes><Envelopes/></Envelopes>"
    return f"""
      <AudioClip Id="{idx}" Time="{arr_start}">
        <CurrentStart Value="{arr_start}"/>
        <CurrentEnd Value="{arr_end}"/>
        {disabled_el}
        <Loop>
          <LoopStart Value="{loop_start}"/>
          <LoopEnd Value="{loop_end}"/>
        </Loop>
        <Name Value="clip{idx}"/>
        <PitchCoarse Value="0"/>
        <PitchFine Value="0"/>
        {envelopes}
        <SourceContext>
          <SourceContext Id="0">
            <OriginalFileRef>
              <FileRef Id="0"><Path Value="{path}"/></FileRef>
            </OriginalFileRef>
          </SourceContext>
        </SourceContext>
        <WarpMarkers>{warps}</WarpMarkers>
      </AudioClip>"""


def _audio_track_xml(
    track_id: int,
    name: str,
    clips: str,
    vol_target_id: int,
    vol_events: tuple[tuple[float, float], ...] = (),
    *,
    speaker_on: bool = True,
    volume_manual: float = 1.0,
) -> str:
    """One AudioTrack.

    ``speaker_on=False`` emits the Track-Activator-off state
    (``Mixer/Speaker/Manual=false``) that Live writes when a track is
    deactivated; ``volume_manual`` sets the static fader value.
    """
    envelope = ""
    if vol_events:
        events = "".join(
            f'<FloatEvent Id="{i}" Time="{t}" Value="{v}"/>'
            for i, (t, v) in enumerate(vol_events)
        )
        envelope = f"""
        <AutomationEnvelopes>
          <Envelopes>
            <AutomationEnvelope Id="0">
              <EnvelopeTarget><PointeeId Value="{vol_target_id}"/></EnvelopeTarget>
              <Automation><Events>{events}</Events></Automation>
            </AutomationEnvelope>
          </Envelopes>
        </AutomationEnvelopes>"""
    speaker = (
        f'<Speaker><Manual Value="{"true" if speaker_on else "false"}"/></Speaker>'
    )
    return f"""
    <AudioTrack Id="{track_id}">
      <Name><EffectiveName Value="{name}"/></Name>
      {envelope}
      <DeviceChain>
        <Mixer>
          {speaker}
          <Volume>
            <Manual Value="{volume_manual}"/>
            <AutomationTarget Id="{vol_target_id}"/>
          </Volume>
        </Mixer>
        <MainSequencer>
          <Sample>
            <ArrangerAutomation>
              <Events>{clips}</Events>
            </ArrangerAutomation>
          </Sample>
        </MainSequencer>
      </DeviceChain>
    </AudioTrack>"""


def session_xml(
    *,
    tempo_events: tuple[tuple[float, float], ...] = ((0.0, 120.0),),
    layer_tracks: tuple[str, ...] = ("layer-a",),
    layer_clip_path: str = "/aligning/set/tracks/001__Artist - Title.m4a",
    mix_warp_points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (512.0, 256.0)),
    layer_specs: tuple[LayerSpec, ...] | None = None,
) -> bytes:
    """Uncompressed XML for a minimal warped-mix session.

    ``layer_specs`` overrides ``layer_tracks`` when given: each spec becomes one
    fully-audible layer track (no fade automation) whose activation / fader /
    clip-disable state is set from the spec — for testing the GT-export silence
    filter.
    """
    tempo_fes = "".join(
        f'<FloatEvent Id="{i + 1}" Time="{t}" Value="{v}"/>'
        for i, (t, v) in enumerate(tempo_events)
    )
    mix_clip = _clip_xml(
        0, "/aligning/set/mix.wav", 0.0, 512.0, warp_points=mix_warp_points
    )
    if layer_specs is not None:
        layers = "".join(
            _audio_track_xml(
                10 + i,
                spec.name,
                _clip_xml(
                    100 + i,
                    f"/aligning/set/tracks/{100 + i:03d}__{spec.name}.m4a",
                    16.0 + 8 * i,
                    48.0 + 8 * i,
                    disabled=spec.clip_disabled,
                    clip_envelope=spec.clip_envelope,
                ),
                VOL_TARGET_ID_BASE + i,
                speaker_on=spec.speaker_on,
                volume_manual=spec.volume_manual,
            )
            for i, spec in enumerate(layer_specs)
        )
    else:
        layers = "".join(
            _audio_track_xml(
                10 + i,
                name,
                _clip_xml(100 + i, layer_clip_path, 16.0 + 8 * i, 48.0 + 8 * i),
                VOL_TARGET_ID_BASE + i,
                vol_events=(
                    (-63072000.0, 1.0),
                    (16.0 + 8 * i, 1.0),
                    (40.0 + 8 * i, 0.0),
                ),
            )
            for i, name in enumerate(layer_tracks)
        )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="11.0_11202" Creator="Ableton Live 11.3.13" Revision="synthetic">
  <LiveSet>
    <Tracks>
      <AudioTrack Id="1">
        <Name><EffectiveName Value="1-mix"/></Name>
        <DeviceChain>
          <Mixer>
            <Volume>
              <Manual Value="1.0"/>
              <AutomationTarget Id="8999"/>
            </Volume>
          </Mixer>
          <MainSequencer>
            <Sample>
              <ArrangerAutomation>
                <Events>{mix_clip}</Events>
              </ArrangerAutomation>
            </Sample>
          </MainSequencer>
        </DeviceChain>
      </AudioTrack>
      {layers}
    </Tracks>
    <MasterTrack>
      <AutomationEnvelopes>
        <Envelopes>
          <AutomationEnvelope Id="1">
            <EnvelopeTarget><PointeeId Value="{TEMPO_TARGET_ID}"/></EnvelopeTarget>
            <Automation><Events>{tempo_fes}</Events></Automation>
          </AutomationEnvelope>
        </Envelopes>
      </AutomationEnvelopes>
      <DeviceChain>
        <Mixer>
          <Tempo>
            <Manual Value="{tempo_events[0][1] if tempo_events else 120.0}"/>
            <AutomationTarget Id="{TEMPO_TARGET_ID}"/>
          </Tempo>
        </Mixer>
      </DeviceChain>
    </MasterTrack>
    <Locators>
      <Locators>
        <Locator Id="0">
          <LomId Value="0"/>
          <Time Value="16"/>
          <Name Value="001 first song"/>
          <Annotation Value=""/>
          <IsSongStart Value="false"/>
        </Locator>
      </Locators>
    </Locators>
  </LiveSet>
</Ableton>
"""
    return xml.encode("utf-8")


def session_root(**kwargs) -> etree._Element:
    return etree.fromstring(session_xml(**kwargs))


def session_als_file(tmp_path, **kwargs):
    """Write a gzipped synthetic session to tmp_path and return the Path."""
    p = tmp_path / "synthetic.als"
    p.write_bytes(gzip.compress(session_xml(**kwargs)))
    return p
