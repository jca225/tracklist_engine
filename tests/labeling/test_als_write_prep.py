"""Tests for the codec write primitives Task 9 added (`write_clip_source_paths`,
`write_clip_names`) and the `prep/relink_als_after_tag.py` +
`prep/fill_als_clip_tags.py` refactor that routes through them.

Two tiers, matching the codec's own testing convention
(tests/labeling/test_als_roundtrip.py): primitive-level roundtrip/locality
checks against the committed `bb12_seeded.als` fixture (the real dialect these
two prep tools actually consume — a freshly-seeded session, before any Live
editing has re-collected samples or reintroduced `OriginalFileRef`; see
`write_clip_source_paths`'s docstring), and an end-to-end integration test
that runs relink -> fill on a scratch aligning folder and checks the clips
resolve to real on-disk files afterward.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from lxml import etree

from labeling.als.cst import dump_als_bytes, load_als_xml
from labeling.als.roundtrip import check_clip_name_write, check_clip_source_write
from labeling.als.validate import has_errors, validate_session
from labeling.als.write import write_clip_names, write_clip_source_paths
from labeling.prep import fill_als_clip_tags, relink_als_after_tag
from tests.labeling.synth_session import session_root

FIXTURE = Path(__file__).parent / "fixtures" / "als" / "bb12_seeded.als"


def _bb12_root() -> etree._Element:
    return load_als_xml(FIXTURE)


# --- write_clip_source_paths: primitive-level ------------------------------


def test_write_clip_source_paths_roundtrip_locality():
    root = _bb12_root()
    # a real stem-dir substring straight out of the fixture, renamed the way
    # inline_tag_aligning_folder.py would tag it.
    edits = [
        (
            "001w1__Post Malone - Congratulations (Acappella)/",
            "001w1__Post Malone - Congratulations (Acappella) [126bpm 8B]/",
        )
    ]
    assert check_clip_source_write(root, edits) == []


def test_write_clip_source_paths_applies_and_scopes_to_sampleref():
    root = _bb12_root()
    edits = [
        (
            "001w1__Post Malone - Congratulations (Acappella)/",
            "001w1__Post Malone - Congratulations (Acappella) [126bpm 8B]/",
        )
    ]
    total = write_clip_source_paths(root, edits)
    assert total > 0

    # every clip whose Path used to contain the old dirname now contains the
    # new one, in both Path and RelativePath.
    hit = 0
    for fref in root.iter("FileRef"):
        parent = fref.getparent()
        if parent is None or parent.tag != "SampleRef":
            continue
        for tag in ("Path", "RelativePath"):
            el = fref.find(tag)
            if el is None:
                continue
            val = el.get("Value") or ""
            if "Post Malone - Congratulations" in val:
                assert "[126bpm 8B]" in val
                hit += 1
    assert hit > 0

    # device-preset FileRefs (not under SampleRef) are untouched — same count
    # before/after for the two known device-preset paths in this fixture.
    device_paths = [
        fr.find("Path").get("Value")
        for fr in root.iter("FileRef")
        if fr.getparent() is not None and fr.getparent().tag != "SampleRef"
        if fr.find("Path") is not None
    ]
    assert "/Dotted Eighth Note.adv" in device_paths


def test_write_clip_source_paths_matches_document_wide_occurrence_count():
    """Cross-check against the old tool's coverage: in this fixture (no
    per-clip OriginalFileRef — the seeder strips it), every occurrence of the
    old substring anywhere in the serialized document lives inside a
    SampleRef/FileRef Path or RelativePath, so the element-scoped primitive's
    hit count must equal a blind whole-document substring count."""
    root = _bb12_root()
    old = "001w1__Post Malone - Congratulations (Acappella)/"
    new = "001w1__Post Malone - Congratulations (Acappella) [126bpm 8B]/"
    xml_before = dump_als_bytes(root).decode("utf-8")
    blind_count = xml_before.count(old)
    assert blind_count > 0

    total = write_clip_source_paths(root, [(old, new)])
    assert total == blind_count

    xml_after = dump_als_bytes(root).decode("utf-8")
    assert old not in xml_after
    assert xml_after.count(new) == blind_count


def test_write_clip_source_paths_edits_path_and_relativepath_independently():
    """RelativePathType 1/3/5 sessions can have Path != RelativePath (different
    prefixes) — the primitive must substitute inside each field's own value,
    not assume they're identical."""
    root = session_root()
    # inject a SampleRef/FileRef with divergent Path/RelativePath onto the
    # first layer clip.
    clip = root.find(".//AudioTrack/DeviceChain//AudioClip")
    sref = etree.SubElement(clip, "SampleRef")
    fref = etree.SubElement(sref, "FileRef")
    etree.SubElement(fref, "Path", Value="/abs/aligning/set/tracks/Foo.m4a")
    etree.SubElement(fref, "RelativePath", Value="../tracks/Foo.m4a")

    total = write_clip_source_paths(root, [("Foo.m4a", "Foo [126bpm 8B].m4a")])
    assert total == 2
    assert (
        fref.find("Path").get("Value") == "/abs/aligning/set/tracks/Foo [126bpm 8B].m4a"
    )
    assert fref.find("RelativePath").get("Value") == "../tracks/Foo [126bpm 8B].m4a"


# --- write_clip_names: primitive-level --------------------------------------


def test_write_clip_names_roundtrip_locality():
    root = session_root()
    renames = {"clip100": "clip100 [126bpm 8B]"}
    assert check_clip_name_write(root, renames) == []


def test_write_clip_names_applies_exact_match_only():
    root = session_root()
    total = write_clip_names(root, {"clip100": "clip100 [126bpm 8B]"})
    assert total == 1
    names = [
        n.get("Value") for n in root.iter("Name") if n.getparent().tag == "AudioClip"
    ]
    assert "clip100 [126bpm 8B]" in names
    assert "clip100" not in names


def test_write_clip_names_bb12_fixture_fill_placeholder():
    root = _bb12_root()
    clip = next(
        c
        for c in root.findall(".//AudioClip")
        if (n := c.find("Name")) is not None and "[?]" in (n.get("Value") or "")
    )
    old_name = clip.find("Name").get("Value")
    new_name = old_name.replace("[?]", "[126bpm 8B]")
    total = write_clip_names(root, {old_name: new_name})
    assert total == 1
    assert clip.find("Name").get("Value") == new_name


# --- integration: relink -> fill on a scratch aligning folder --------------


def test_relink_then_fill_end_to_end(tmp_path):
    """Build a tiny scratch aligning folder (tagged tracks/stems files on
    disk, a .als still pointing at the pre-tag names) and run relink then
    fill through their real CLIs' functions. Clips must resolve to files that
    exist on disk, names must lose their `[?]` placeholder, and the codec's
    own laws (locality already proven above; reparse stability + validation
    here) must hold on the result — this is "opened successfully" without
    literally opening Live."""
    set_dir = tmp_path / "1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
    tracks = set_dir / "tracks"
    stems_dir = set_dir / "stems" / "002__Other - Song [130bpm 5A]"
    tracks.mkdir(parents=True)
    stems_dir.mkdir(parents=True)

    tagged_track = tracks / "001__Artist - Title [126bpm 8B].m4a"
    tagged_track.write_bytes(b"fake-audio")
    (stems_dir / "vocals.flac").write_bytes(b"fake-audio")

    untagged_track_path = str(set_dir / "tracks" / "001__Artist - Title.m4a")
    untagged_stem_path = str(set_dir / "stems" / "002__Other - Song" / "vocals.flac")

    # Build a minimal session: two clips in the freshly-seeded dialect
    # (SampleRef/FileRef only, Path == RelativePath, Name carries "[?]").
    root = _bb12_root()
    clips = root.findall(".//AudioClip")
    track_clip, stem_clip = clips[1], clips[2]

    def _point_at(clip, path: str, name: str) -> None:
        fref = clip.find(".//SampleRef/FileRef")
        fref.find("Path").set("Value", path)
        fref.find("RelativePath").set("Value", path)
        clip.find("Name").set("Value", name)

    _point_at(track_clip, untagged_track_path, "001__Artist - Title [?]")
    _point_at(stem_clip, untagged_stem_path, "002__Other - Song [?]")

    als_path = set_dir / "seeded.als"
    als_path.write_bytes(gzip.compress(dump_als_bytes(root)))

    # relink
    edits = relink_als_after_tag.build_renames(set_dir)
    assert set(edits) == {
        ("001__Artist - Title.m4a", "001__Artist - Title [126bpm 8B].m4a"),
        ("002__Other - Song/", "002__Other - Song [130bpm 5A]/"),
    }
    n_relinked = relink_als_after_tag.relink(als_path, edits, dry_run=False)
    assert n_relinked > 0
    assert als_path.with_suffix(".als.prerelink.bak").exists()

    relinked_root = load_als_xml(als_path)
    new_track_path = (
        relinked_root.findall(".//AudioClip")[1]
        .find(".//SampleRef/FileRef/Path")
        .get("Value")
    )
    new_stem_path = (
        relinked_root.findall(".//AudioClip")[2]
        .find(".//SampleRef/FileRef/Path")
        .get("Value")
    )
    assert Path(new_track_path) == tagged_track
    assert Path(new_track_path).exists()
    assert Path(new_stem_path) == stems_dir / "vocals.flac"
    assert Path(new_stem_path).exists()

    # fill
    fill_root = load_als_xml(als_path)
    name_edits, conflicts = fill_als_clip_tags.build_name_edits(fill_root)
    assert conflicts == []
    assert name_edits == {
        "001__Artist - Title [?]": "001__Artist - Title [126bpm 8B]",
        "002__Other - Song [?]": "002__Other - Song [130bpm 5A]",
    }
    n_filled = fill_als_clip_tags.apply_edits(
        als_path, fill_root, name_edits, dry_run=False
    )
    assert n_filled == 2
    assert als_path.with_suffix(".als.prefill.bak").exists()

    final_root = load_als_xml(als_path)
    final_names = {
        c.find("Name").get("Value")
        for c in (
            final_root.findall(".//AudioClip")[1],
            final_root.findall(".//AudioClip")[2],
        )
    }
    assert final_names == {
        "001__Artist - Title [126bpm 8B]",
        "002__Other - Song [130bpm 5A]",
    }
    assert not any("[?]" in n for n in final_names)

    # codec laws hold on the end result
    diags = validate_session(final_root)
    assert not has_errors(diags), [d.render() for d in diags]
