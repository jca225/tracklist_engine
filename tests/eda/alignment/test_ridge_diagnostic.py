from __future__ import annotations

from eda.alignment.ridge_diagnostic.cases import CaseRecord, _rank_candidates


def test_rank_candidates_prefers_large_place_err_with_id_correct() -> None:
    rows = [
        {"id_correct": True, "place_err_s": 40.0, "slot": "a"},
        {"id_correct": True, "place_err_s": 5.0, "slot": "b"},
        {"id_correct": False, "place_err_s": 99.0, "slot": "c"},
        {"id_correct": True, "place_err_s": None, "slot": "d"},
    ]
    ranked = _rank_candidates(rows, min_place_err_s=15.0)
    assert [r["slot"] for r in ranked] == ["a"]
