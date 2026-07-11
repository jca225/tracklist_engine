from __future__ import annotations
from workspaces.alignment_prototype.experiments.matrix import Cell, cell_hash, PAPER


def test_cell_hash_stable_and_order_independent():
    a = Cell(driver="classical", set_id="1fsnxchk", decoder="looptrace")
    b = Cell(driver="classical", set_id="1fsnxchk", decoder="looptrace")
    assert cell_hash(a) == cell_hash(b)
    assert cell_hash(a) != cell_hash(
        Cell(driver="classical", set_id="1fsnxchk", decoder="legacy")
    )


def test_paper_matrix_covers_c4_and_both_sets():
    sets = {c.set_id for c in PAPER}
    assert sets == {"2nvzlh2k", "1fsnxchk"}  # BB11, BB12
    # C4 ablation present: classical looptrace vs legacy on each set
    for sid in sets:
        decoders = {
            c.decoder for c in PAPER if c.driver == "classical" and c.set_id == sid
        }
        assert {"looptrace", "legacy"} <= decoders
    # all three drivers represented
    assert {c.driver for c in PAPER} == {"classical", "agentic", "ml"}
