from __future__ import annotations
import ast
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "alignment/experiments"


def test_only_score_spans_supplies_metrics():
    """No experiments module may import the raw trajectory_acc or a duplicate
    scorer — headline metrics come solely through score_spans()."""
    banned = {"trajectory_acc"}
    for py in P.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
                assert not (names & banned), f"{py.name} imports {names & banned}"
            elif isinstance(node, ast.Import):
                names = {a.name.split(".")[-1] for a in node.names}
                assert not (names & banned), f"{py.name} bare-imports {names & banned}"
