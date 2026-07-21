"""Content-identity fields on ParsedClip (Operation Crush §9 — path/identity fix).

Ableton records per-clip content identity (`OriginalFileSize` + `OriginalCrc`)
on the ACTIVE FileRef — the exact bytes labeled against. The codec must surface
these so identity can bind by content, not by mutable path/slot strings.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from labeling.als import clip_content_identity, load_als_xml, parse_layer_clips

FIX = Path(__file__).parent / "fixtures" / "als"


def test_content_identity_reads_active_ref_not_sourcecontext():
    # A collected clip keeps its ORIGINAL external identity inside SourceContext.
    # Identity must come from the ACTIVE FileRef (999/888 below is the historical
    # ghost and must be ignored) — the SourceContext-preference inversion that
    # caused the path-MISS mess (Operation Crush §9).
    clip = etree.fromstring(
        """
        <AudioClip>
          <SampleRef><FileRef>
            <Path Value="/now/Samples/Imported/x.flac"/>
            <OriginalFileSize Value="111"/>
            <OriginalCrc Value="222"/>
          </FileRef></SampleRef>
          <SourceContext><SourceContext><OriginalFileRef><FileRef>
            <Path Value="/old/elsewhere/x.flac"/>
            <OriginalFileSize Value="999"/>
            <OriginalCrc Value="888"/>
          </FileRef></OriginalFileRef></SourceContext></SourceContext>
        </AudioClip>
        """
    )
    assert clip_content_identity(clip) == (111, 222)


def test_parsed_clip_carries_active_fileref_content_identity():
    root = load_als_xml(FIX / "warped_basic.als")
    clips = parse_layer_clips(root)

    assert clips, "fixture parsed no clips"
    assert hasattr(clips[0], "file_size"), (
        "ParsedClip missing content-identity field file_size"
    )
    assert hasattr(clips[0], "crc"), "ParsedClip missing content-identity field crc"

    # warped_basic.als has one reference clip (the other AudioClip is the mix,
    # which parse_layer_clips correctly skips). Its active FileRef records this
    # exact (OriginalFileSize, OriginalCrc) — read off the ACTIVE ref, never the
    # historical SourceContext copy.
    ref = clips[0]
    assert ref.file_size == 3829708
    assert ref.crc == 44972
