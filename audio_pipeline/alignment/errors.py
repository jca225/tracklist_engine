from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentError:
    kind: str
    detail: str
