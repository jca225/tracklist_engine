"""Run a Cell: build its driver (with toggles), produce a timeline (cached by
cell_hash), score via the single scorer, write per-span rows to the store.

Reuse only — drivers are invoked through their existing config kwargs. Agentic
and ml cells need a classical base timeline (their refinement source), same as
drivers/race.py."""

from __future__ import annotations

import shutil
from pathlib import Path

from workspaces.alignment_prototype.drivers.base import SetContext
from workspaces.alignment_prototype.drivers.classical import ClassicalDriver
from workspaces.alignment_prototype.drivers.agentic import AgenticDriver
from workspaces.alignment_prototype.drivers.ml import HybridMlDriver
from workspaces.alignment_prototype.experiments.matrix import Cell, cell_hash
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans

_CACHE = Path(__file__).resolve().parent / "cache"

# gate_margin note: HybridMlDriver gates on `score < gate_margin` (ml.py:149).
# gate_margin=0.0 is not None (gate "enabled") but trajectory scores are
# non-negative, so score < 0.0 is never True — functionally ungated.
# Use 0.05 for the ml_gate=True cell so gated vs ungated cells actually differ.
_ML_GATE_MARGIN = 0.05  # ml_gate=True  → gate spans whose ml conf < 0.05
_ML_NOGATE_MARGIN = None  # ml_gate=False → replace unconditionally


def build_driver(cell: Cell, base_timeline: Path | None):
    if cell.driver == "classical":
        return ClassicalDriver(decoder=cell.decoder)
    if cell.driver == "agentic":
        assert base_timeline is not None, "agentic needs a classical base"
        return AgenticDriver(base_timeline, live=cell.live)
    if cell.driver == "ml":
        assert base_timeline is not None, "ml needs a classical base"
        margin = _ML_GATE_MARGIN if cell.ml_gate else _ML_NOGATE_MARGIN
        return HybridMlDriver(base_timeline, gate_margin=margin)
    raise ValueError(f"unknown driver {cell.driver!r}")


def _cached_timeline(cell: Cell) -> Path:
    return _CACHE / f"{cell_hash(cell)}.json"


def run_cell(
    cell: Cell,
    store: Store,
    *,
    fibers: bool,
    driver_factory=build_driver,
    base_timeline: Path | None = None,
) -> int:
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached = _cached_timeline(cell)
    if not cached.exists():
        ctx = SetContext.for_set(cell.set_id)
        driver = driver_factory(cell, base_timeline)
        produced = Path(driver.align_set(ctx))
        shutil.copyfile(produced, cached)
    rows = score_spans(cell.set_id, cached, fibers=fibers)
    store.upsert(cell, rows)
    return len(rows)
