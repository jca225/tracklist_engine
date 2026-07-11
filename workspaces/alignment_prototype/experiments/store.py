"""Long-format sqlite results store: one row per (cell × span). Tidy — every
paper table is a GROUP BY. Idempotent on (cell_hash, slot, recording_id)."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

from workspaces.alignment_prototype.experiments.matrix import Cell, cell_hash
from workspaces.alignment_prototype.score_timeline_vs_gt import SpanScore

_COLS = [
    "cell_hash",
    "driver",
    "set_id",
    "decoder",
    "ml_gate",
    "live",
    "slot",
    "recording_id",
    "stem",
    "span_class",
    "id_correct",
    "place_err_s",
    "strict",
    "fiber",
    "ref_err_s",
    "density",
]


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cols = ", ".join(f"{c}" for c in _COLS)
            cx.execute(f"CREATE TABLE IF NOT EXISTS scores ({cols})")
            cx.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_span "
                "ON scores (cell_hash, slot, recording_id)"
            )

    def _conn(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path)
        cx.row_factory = sqlite3.Row
        return cx

    def upsert(self, cell: Cell, rows: list[SpanScore]) -> None:
        h = cell_hash(cell)
        cd = asdict(cell)
        placeholders = ", ".join("?" for _ in _COLS)
        with self._conn() as cx:
            for r in rows:
                rd = asdict(r)
                vals = [
                    h,
                    cd["driver"],
                    cd["set_id"],
                    cd["decoder"],
                    int(cd["ml_gate"]),
                    int(cd["live"]),
                    rd["slot"],
                    rd["recording_id"],
                    rd["stem"],
                    rd["span_class"],
                    (None if rd["id_correct"] is None else int(rd["id_correct"])),
                    rd["place_err_s"],
                    rd["strict"],
                    rd["fiber"],
                    rd["ref_err_s"],
                    rd["density"],
                ]
                cx.execute(
                    f"INSERT OR REPLACE INTO scores ({', '.join(_COLS)}) "
                    f"VALUES ({placeholders})",
                    vals,
                )

    def fetch(
        self, *, driver: str | None = None, set_id: str | None = None
    ) -> list[dict]:
        q, args = "SELECT * FROM scores", []
        clauses = []
        if driver is not None:
            clauses.append("driver = ?")
            args.append(driver)
        if set_id is not None:
            clauses.append("set_id = ?")
            args.append(set_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(q, args).fetchall()]
