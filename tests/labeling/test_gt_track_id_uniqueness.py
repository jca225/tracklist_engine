"""#63: a `track_id` must denote ONE recording within a set.

Two slots sharing a track_id while naming DIFFERENT recordings is the
identity-by-string collision: under the old acquisition scheme the two merged
into a single case and minted a corrupt cross-track preference pair (BB12
`2p25k23p` = Garrix "In The Name Of Love" AND Beatles "Can't Buy Me Love").
`schema.load` must reject it loudly; reusing a track_id for the SAME recording
(played twice, or a base+stem pair) stays legal.
"""

from __future__ import annotations

from pathlib import Path

from labeling.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    dump,
    load,
)

_REPO = Path(__file__).resolve().parents[2]


def _track(**kw) -> GroundTruthTrack:
    base = dict(
        label="X",
        track_id="rec1",
        claimed_stem="regular",
        set_start_s=1.0,
        set_end_s=2.0,
        ref_start_s=0.0,
    )
    base.update(kw)
    return GroundTruthTrack(**base)


def _load_dumped(tmp_path: Path, *tracks: GroundTruthTrack):
    gt = GroundTruthSet(set_id="s", tracks=tracks)
    p = tmp_path / "gt.yaml"
    p.write_text(dump(gt))
    return load(p)


def test_shared_track_id_distinct_names_is_rejected(tmp_path: Path) -> None:
    r = _load_dumped(
        tmp_path,
        _track(
            label="Martin Garrix - In The Name Of Love",
            track_id="2p25k23p",
            slot_label="012",
        ),
        _track(
            label="The Beatles - Can't Buy Me Love",
            track_id="2p25k23p",
            slot_label="028",
        ),
    )
    assert not r.is_ok()
    assert r.error.kind == "identity_collision"
    # the message must name the offending id and both recordings
    assert "2p25k23p" in r.error.detail
    assert "Garrix" in r.error.detail and "Beatles" in r.error.detail


def test_same_track_id_same_name_is_legal(tmp_path: Path) -> None:
    # a track played twice (or a base+stem pair) shares the track_id — fine.
    r = _load_dumped(
        tmp_path,
        _track(label="Two Friends - Emily", track_id="rec9", slot_label="003"),
        _track(label="Two Friends - Emily", track_id="rec9", slot_label="050"),
    )
    assert r.is_ok(), r.error.detail if not r.is_ok() else ""


def test_missing_track_ids_never_collide(tmp_path: Path) -> None:
    # abstained rows carry no track_id and must not be conflated with each other.
    r = _load_dumped(
        tmp_path,
        _track(label="A - one", track_id=None, slot_label="001"),
        _track(label="B - two", track_id=None, slot_label="002"),
    )
    assert r.is_ok(), r.error.detail if not r.is_ok() else ""


def test_whitespace_only_label_difference_is_not_a_collision(tmp_path: Path) -> None:
    r = _load_dumped(
        tmp_path,
        _track(
            label="Post Malone - Congratulations", track_id="rec7", slot_label="002"
        ),
        _track(
            label="Post Malone - Congratulations ", track_id="rec7", slot_label="040"
        ),
    )
    assert r.is_ok(), r.error.detail if not r.is_ok() else ""


def test_canonical_fixtures_pass_the_gate() -> None:
    # regression guard: the gate must not false-fail on the committed GT.
    for name in ("bb11_ground_truth.yaml", "bb12_ground_truth.yaml"):
        r = load(_REPO / "labeling" / "fixtures" / name)
        assert r.is_ok(), (
            f"{name}: {r.error.kind}: {r.error.detail}" if not r.is_ok() else ""
        )
