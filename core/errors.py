from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DbError:
    kind: str  # 'not_found' | 'query_failed' | 'integrity'
    detail: str
