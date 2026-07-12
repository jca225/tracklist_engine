from __future__ import annotations
import json
from pathlib import Path
import workspaces.alignment_prototype.experiments.run as _run_mod
from workspaces.alignment_prototype.experiments.run import run_cell
from workspaces.alignment_prototype.experiments.matrix import Cell
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.drivers.base import SetContext


class _StubDriver:
    calls = 0

    def __init__(self, tl: Path):
        self._tl = tl

    def align_set(self, ctx):
        _StubDriver.calls += 1
        return self._tl


def test_run_cell_scores_and_caches(tmp_path, monkeypatch):
    # Redirect the module-level cache dir into tmp_path so the test is
    # hermetic and never pollutes the source tree.
    monkeypatch.setattr(_run_mod, "_CACHE", tmp_path / "cache")
    # Stub out SetContext.for_set so the test never touches ~/aligning on disk.
    # The stub driver_factory ignores ctx, so a sentinel object suffices.
    monkeypatch.setattr(SetContext, "for_set", staticmethod(lambda set_id: object()))
    # a minimal timeline JSON with one span that has no same-rec GT → strict None,
    # but score_spans still returns one SpanScore row.
    tl = tmp_path / "tl.json"
    tl.write_text(
        json.dumps(
            {
                "set_id": "1fsnxchk",
                "spans": [
                    {
                        "slot_label": "6",
                        "recording_id": "nope",
                        "set_start_s": 1.0,
                        "set_end_s": 9.0,
                        "name": "x",
                        "ref_start_s": 0.0,
                        "claimed_stem": "regular",
                    }
                ],
            }
        )
    )
    store = Store(tmp_path / "s.db")
    cell = Cell("classical", "1fsnxchk", decoder="looptrace")
    factory = lambda c, base: _StubDriver(tl)
    n1 = run_cell(cell, store, fibers=False, driver_factory=factory)
    assert n1 == 1 and len(store.fetch(set_id="1fsnxchk")) == 1
    # second run hits the timeline cache: driver not called again
    before = _StubDriver.calls
    run_cell(cell, store, fibers=False, driver_factory=factory)
    assert _StubDriver.calls == before  # cached, no re-inference
