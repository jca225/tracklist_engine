"""A2: `claimed_variant` survives GT-yaml emit -> load symmetrically.

Mirrors `claimed_stem`'s emit-when-non-default / load-with-fallback
contract (see test_gt_id_source.py for the same pattern applied to
`id_source`). The emitter and loader MUST stay symmetric or the
`anchor_check` / `gt_als_gate` roundtrip (yaml == fresh re-export) breaks.
"""

from __future__ import annotations

from pathlib import Path

from labeling.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    dump,
    load,
)


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


def test_claimed_variant_defaults_regular() -> None:
    assert _track().claimed_variant == "regular"


def test_claimed_variant_extended_round_trips(tmp_path: Path) -> None:
    gt = GroundTruthSet(set_id="s", tracks=(_track(claimed_variant="extended"),))
    p = tmp_path / "gt.yaml"
    p.write_text(dump(gt))
    back = load(p)
    assert back.is_ok()
    assert back.value.tracks[0].claimed_variant == "extended"


def test_claimed_variant_regular_omitted_from_yaml_and_loads_back_regular(
    tmp_path: Path,
) -> None:
    gt = GroundTruthSet(set_id="s", tracks=(_track(claimed_variant="regular"),))
    text = dump(gt)
    assert "claimed_variant" not in text

    p = tmp_path / "gt.yaml"
    p.write_text(text)
    back = load(p)
    assert back.is_ok()
    assert back.value.tracks[0].claimed_variant == "regular"
