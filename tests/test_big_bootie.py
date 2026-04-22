"""Tests for the Big Bootie helper — focus on the cue-section resolver,
which is the non-trivial piece of logic that downstream section-level
alignment depends on.
"""
from __future__ import annotations

import pandas as pd
import pytest

from big_bootie import _resolve_cue_sections


def _rows(records) -> pd.DataFrame:
    return pd.DataFrame(records)


def test_resolver_ffills_cue_within_a_group():
    """`w/` rows (cue_seconds = 0) should inherit the parent's cue time."""
    df = _rows([
        {"set_id": "S1", "row_index": 0, "row_kind": "track", "cue_seconds": 100.0},
        {"set_id": "S1", "row_index": 1, "row_kind": "track", "cue_seconds": 0.0},
        {"set_id": "S1", "row_index": 2, "row_kind": "track", "cue_seconds": 0.0},
        {"set_id": "S1", "row_index": 3, "row_kind": "track", "cue_seconds": 200.0},
        {"set_id": "S1", "row_index": 4, "row_kind": "track", "cue_seconds": 0.0},
    ])
    out = _resolve_cue_sections(df)
    assert out["cue_seconds_section"].tolist() == [100.0, 100.0, 100.0, 200.0, 200.0]


def test_resolver_does_not_cross_set_boundaries():
    """Two different sets must not share cue anchors."""
    df = _rows([
        {"set_id": "A", "row_index": 0, "row_kind": "track", "cue_seconds": 10.0},
        {"set_id": "A", "row_index": 1, "row_kind": "track", "cue_seconds": 0.0},
        {"set_id": "B", "row_index": 0, "row_kind": "track", "cue_seconds": 0.0},
        {"set_id": "B", "row_index": 1, "row_kind": "track", "cue_seconds": 50.0},
    ])
    out = _resolve_cue_sections(df).sort_values(["set_id", "row_index"]).reset_index(drop=True)
    values = out["cue_seconds_section"].tolist()
    assert values[0] == 10.0 and values[1] == 10.0   # A ffills within itself
    assert pd.isna(values[2])                        # B has no parent yet
    assert values[3] == 50.0


def test_resolver_all_zero_produces_all_nan():
    """Pathological case: the 10 cueless Big Bootie sets. No parent ever fires,
    so every row stays NaN and downstream analysis can skip the set cleanly."""
    df = _rows([
        {"set_id": "X", "row_index": i, "row_kind": "track", "cue_seconds": 0.0}
        for i in range(5)
    ])
    out = _resolve_cue_sections(df)
    assert out["cue_seconds_section"].isna().all()


def test_resolver_ignores_non_track_rows_when_anchoring():
    """Non-track rows (player_widget, text, etc.) must not overwrite the anchor
    — even if they have a spurious cue_seconds value."""
    df = _rows([
        {"set_id": "S", "row_index": 0, "row_kind": "player_widget", "cue_seconds": 999.0},
        {"set_id": "S", "row_index": 1, "row_kind": "track",          "cue_seconds": 50.0},
        {"set_id": "S", "row_index": 2, "row_kind": "track",          "cue_seconds": 0.0},
    ])
    out = _resolve_cue_sections(df).sort_values("row_index").reset_index(drop=True)
    # player_widget cannot be the anchor because the mask is `row_kind == 'track'`
    assert pd.isna(out.loc[0, "cue_seconds_section"])
    assert out.loc[1, "cue_seconds_section"] == 50.0
    assert out.loc[2, "cue_seconds_section"] == 50.0


def test_resolver_preserves_sort_order():
    """Output must be ordered by (set_id, row_index) even if input isn't."""
    df = _rows([
        {"set_id": "S", "row_index": 3, "row_kind": "track", "cue_seconds": 30.0},
        {"set_id": "S", "row_index": 1, "row_kind": "track", "cue_seconds": 10.0},
        {"set_id": "S", "row_index": 2, "row_kind": "track", "cue_seconds": 0.0},
    ])
    out = _resolve_cue_sections(df)
    assert out["row_index"].tolist() == [1, 2, 3]
    assert out["cue_seconds_section"].tolist() == [10.0, 10.0, 30.0]
