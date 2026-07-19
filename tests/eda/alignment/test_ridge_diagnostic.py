from __future__ import annotations

import numpy as np

from eda.alignment.ridge_diagnostic.cases import CaseRecord, _rank_candidates
from eda.alignment.ridge_diagnostic.features import cosine_sim_matrix


def test_rank_candidates_prefers_large_place_err_with_id_correct() -> None:
    rows = [
        {"id_correct": True, "place_err_s": 40.0, "slot": "a"},
        {"id_correct": True, "place_err_s": 5.0, "slot": "b"},
        {"id_correct": False, "place_err_s": 99.0, "slot": "c"},
        {"id_correct": True, "place_err_s": None, "slot": "d"},
    ]
    ranked = _rank_candidates(rows, min_place_err_s=15.0)
    assert [r["slot"] for r in ranked] == ["a"]


def test_cosine_sim_matrix_identity_has_strong_diagonal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    m = cosine_sim_matrix(x, x)
    assert m.shape == (20, 20)
    assert float(np.mean(np.diag(m))) > 0.9
