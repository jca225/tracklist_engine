"""Declarative ablation matrix: one Cell = one (driver-config, set) to run+score.

An ablation is a pair of cells differing by exactly one field. Sets:
BB11=2nvzlh2k (Episode 11), BB12=1fsnxchk (Volume 12).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict

BB11 = "2nvzlh2k"
BB12 = "1fsnxchk"


@dataclass(frozen=True)
class Cell:
    driver: str  # "classical" | "agentic" | "ml"
    set_id: str
    decoder: str = "looptrace"  # classical only: "looptrace" | "legacy"
    ml_gate: bool = True  # ml only: gated vs ungated decode
    live: bool = True  # agentic only

    @property
    def label(self) -> str:
        extra = {
            "classical": self.decoder,
            "ml": "gated" if self.ml_gate else "ungated",
            "agentic": "live" if self.live else "replay",
        }[self.driver]
        return f"{self.driver}:{extra}"


def cell_hash(cell: Cell) -> str:
    payload = repr(sorted(asdict(cell).items()))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _per_set(sid: str) -> tuple[Cell, ...]:
    return (
        Cell("classical", sid, decoder="looptrace"),  # baseline
        Cell("classical", sid, decoder="legacy"),  # C4 ablation
        Cell("agentic", sid),
        Cell("ml", sid, ml_gate=True),
        Cell("ml", sid, ml_gate=False),
    )


PAPER: tuple[Cell, ...] = _per_set(BB11) + _per_set(BB12)
