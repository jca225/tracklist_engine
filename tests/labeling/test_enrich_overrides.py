"""Identity-override + tie-resolving name-match (slot-renumber/generic-import)."""
from __future__ import annotations

from labeling.enrich_gt_track_ids import SlotRow, load_identity_overrides, lookup_db_label


def test_load_identity_overrides_bb12():
    ov = load_identity_overrides("1fsnxchk")
    assert ov.get("instrumental-2") == "lvzr9s5"        # Mako (generic import)
    assert ov.get("155") == "mtck04x"                   # Manse (degenerate label)
    assert ov.get("Outside Official Acapella") == "y10w66f"
    assert load_identity_overrides("nonexistent_set") == {}


def test_resolve_ties_prefers_base_recording():
    # Vicetone appears as a regular AND an acappella scrape row -> a tie that
    # the old unique-top logic dropped. resolve_ties picks the GT stem, else base.
    slots = (
        SlotRow(track_id="reg9", claimed_stem="regular", display="Vicetone - Nothing Stopping Me"),
        SlotRow(track_id="acap9", claimed_stem="acappella", display="Vicetone - Nothing Stopping Me (Acappella)"),
    )
    # GT instrumental -> no stem match -> falls back to base/regular recording
    assert lookup_db_label("Vicetone - Nothing Stopping Me", "instrumental", slots,
                           require_stem=False, resolve_ties=True) == "reg9"
    # GT acappella -> prefers the acappella recording
    assert lookup_db_label("Vicetone - Nothing Stopping Me", "acappella", slots,
                           require_stem=False, resolve_ties=True) == "acap9"
    # without resolve_ties the tie is dropped (old conservative behavior)
    assert lookup_db_label("Vicetone - Nothing Stopping Me", "instrumental", slots,
                           require_stem=False) is None
