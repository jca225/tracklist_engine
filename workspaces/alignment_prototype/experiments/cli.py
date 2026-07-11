"""Run the PAPER ablation matrix → store → print headline + ablation tables.

Base classical timelines are computed once per set and reused for agentic/ml
(their refinement source), mirroring drivers/race.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from workspaces.alignment_prototype.experiments.matrix import PAPER, Cell
from workspaces.alignment_prototype.experiments.run import (
    _cached_timeline,
    build_driver,
    run_cell,
)
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.experiments import report

_RESULTS = Path(__file__).resolve().parent / "results" / "scores.db"

_MATRICES = {"paper": PAPER}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fibers", action="store_true", help="fiber-aware scoring (needs audio)"
    )
    p.add_argument("--matrix", default="paper", choices=list(_MATRICES))
    args = p.parse_args(argv)

    cells = _MATRICES[args.matrix]
    store = Store(_RESULTS)
    # ensure a classical/looptrace base per set exists first (reused downstream)
    bases: dict[str, Path] = {}
    for sid in sorted({c.set_id for c in cells}):
        base_cell = Cell("classical", sid, decoder="looptrace")
        run_cell(base_cell, store, fibers=args.fibers)
        bases[sid] = _cached_timeline(base_cell)

    for cell in cells:
        base = bases[cell.set_id] if cell.driver in ("agentic", "ml") else None
        run_cell(cell, store, fibers=args.fibers, base_timeline=base)

    rows = store.fetch()
    print("\n## Headline (baseline classical/looptrace)\n")
    print(report.headline_table(rows))
    print("\n## Ablations (paired span-bootstrap CI on the delta)\n")
    print("| ablation | left | right | Δ [95% CI] |")
    print("|---|---|---|---|")
    print(report.ablation_table(rows, "decoder", "looptrace", "legacy"))  # C4
    print(report.ablation_table(rows, "driver", "agentic", "classical"))
    print(report.ablation_table(rows, "driver", "ml", "classical"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
