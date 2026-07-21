"""The GT-yaml ⟵ .als drift gate (labeling/gt_als_gate.py).

Hermetic: uses the committed BB11 ``.als`` (labeling/fixtures/als/2nvzlh2k.als) and
the committed GT yaml — no ``~/aligning/`` and no manifest. Regression for main's
BB11 slot-013 drift (commit 518065f hand-edited "Kill FM - Fresh" 013w1 -> 013w3
while the .als says 013w1).
"""

from __future__ import annotations

import pytest

from labeling.gt_als_gate import (
    GATED_GT_YAMLS,
    GATED_SETS,
    check_yaml_matches_als,
    iter_gated,
    slot_labels_from_als,
    slot_labels_from_yaml,
)

BB11 = "2nvzlh2k"


def test_bb11_yaml_matches_committed_als():
    ok, reason = check_yaml_matches_als(GATED_GT_YAMLS[BB11], GATED_SETS[BB11])
    assert ok, reason
    assert "match" in reason


def test_committed_als_is_present():
    # The gate is inert without the committed .als — pin that it stays checked in.
    assert GATED_SETS[BB11].is_file(), GATED_SETS[BB11]


def test_planted_013w3_fails_with_reason(tmp_path):
    # Reproduce the exact main-branch drift on a copy: one 013w1 -> 013w3.
    src = GATED_GT_YAMLS[BB11].read_text()
    planted = src.replace('slot_label:  "013w1"', 'slot_label:  "013w3"', 1)
    assert planted != src, "fixture no longer contains slot 013w1 to plant on"
    bad = tmp_path / "bb11_planted.yaml"
    bad.write_text(planted)

    ok, reason = check_yaml_matches_als(bad, GATED_SETS[BB11])

    assert not ok
    assert "013w3" in reason  # invented label present in yaml, absent from .als
    assert "013w1" in reason  # real label present in .als, dropped from yaml


def test_planted_extra_slot_fails(tmp_path):
    # A hand-invented slot with no basis in the .als must be caught too.
    src = GATED_GT_YAMLS[BB11].read_text()
    planted = src.replace('slot_label:  "013w1"', 'slot_label:  "999w9"', 1)
    bad = tmp_path / "bb11_extra.yaml"
    bad.write_text(planted)

    ok, reason = check_yaml_matches_als(bad, GATED_SETS[BB11])

    assert not ok
    assert "999w9" in reason


def test_slot_sets_are_equal_but_multiset_is_not():
    # Design invariant: the gate compares SETS, not multisets. The .als's hygiene
    # collapse (loop/sliver/split) means the raw multiset is larger, but the sets
    # of distinct slot labels coincide.
    als_slots = slot_labels_from_als(GATED_SETS[BB11])
    yaml_slots = slot_labels_from_yaml(GATED_GT_YAMLS[BB11])
    assert als_slots == yaml_slots
    assert als_slots  # non-empty (guards against a silently-empty extraction)
    assert "013w1" in als_slots
    assert "013w3" not in als_slots


def test_missing_als_returns_not_ok(tmp_path):
    ok, reason = check_yaml_matches_als(GATED_GT_YAMLS[BB11], tmp_path / "nope.als")
    assert not ok
    assert "not found" in reason


def test_registry_keys_are_consistent():
    # Every gated set must carry both a committed .als and a GT yaml.
    assert set(GATED_SETS) == set(GATED_GT_YAMLS)
    assert BB11 in GATED_SETS
    # BB12 is deferred pending its canonical .als — must NOT be gated yet.
    assert "1fsnxchk" not in GATED_SETS


@pytest.mark.parametrize("set_id,yaml_path,als_path", iter_gated())
def test_all_gated_sets_pass(set_id, yaml_path, als_path):
    ok, reason = check_yaml_matches_als(yaml_path, als_path)
    assert ok, f"{set_id}: {reason}"
