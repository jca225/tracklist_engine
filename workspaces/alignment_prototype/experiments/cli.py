"""Run the PAPER ablation matrix → store → print headline + ablation tables.

Base classical timelines are computed once per set and reused for agentic/ml
(their refinement source), mirroring drivers/race.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from workspaces.alignment_prototype.experiments.matrix import PAPER, Cell
from workspaces.alignment_prototype.experiments.run import (
    _cached_timeline,
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
    # A failed cell (e.g. a transient pi-storage SSH drop mid-infer) must NOT
    # abort the whole matrix — the store is per-cell, so skip + report + continue;
    # re-running the CLI cache-hits the survivors and retries only the failures.
    failed: list[tuple[str, str]] = []
    # ensure a classical/looptrace base per set exists first (reused downstream)
    bases: dict[str, Path] = {}
    for sid in sorted({c.set_id for c in cells}):
        base_cell = Cell("classical", sid, decoder="looptrace")
        try:
            run_cell(base_cell, store, fibers=args.fibers)
            bases[sid] = _cached_timeline(base_cell)
        except Exception as e:  # noqa: BLE001 — one cell must not sink the matrix
            failed.append((f"base[{sid}]", str(e).splitlines()[-1][:160]))
            print(f"[skip] base {sid}: {type(e).__name__}")

    for cell in cells:
        needs_base = cell.driver in ("agentic", "ml")
        if needs_base and cell.set_id not in bases:
            failed.append((f"{cell.label}[{cell.set_id}]", "base timeline unavailable"))
            continue
        base = bases.get(cell.set_id) if needs_base else None
        try:
            run_cell(cell, store, fibers=args.fibers, base_timeline=base)
        except Exception as e:  # noqa: BLE001 — one cell must not sink the matrix
            failed.append(
                (f"{cell.label}[{cell.set_id}]", str(e).splitlines()[-1][:160])
            )
            print(f"[skip] {cell.set_id} {cell.label}: {type(e).__name__}")

    rows = store.fetch()
    print("\n## Headline (baseline classical/looptrace)\n")
    print(report.headline_table(rows))
    print("\n## Ablations (paired span-bootstrap CI on the delta)\n")
    print("| ablation | left | right | Δ [95% CI] |")
    print("|---|---|---|---|")
    print(report.ablation_table(rows, "decoder", "looptrace", "legacy"))  # C4
    print(report.ablation_table(rows, "driver", "agentic", "classical"))
    print(report.ablation_table(rows, "driver", "ml", "classical"))
    if failed:
        print("\n## Cells skipped (re-run to retry — cache-hits the rest):")
        for label, why in failed:
            print(f"  - {label}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
