from __future__ import annotations

from dataclasses import dataclass

from core.errors import DbError  # canonical home is core.errors; re-exported here for the PipelineError union and ingest callers


@dataclass(frozen=True)
class DownloadError:
    kind: str  # 'unavailable' | 'network' | 'parse' | 'disk' | 'unsupported_platform' | 'tool_missing'
    url: str
    detail: str


@dataclass(frozen=True)
class SpotifyApiError:
    kind: str  # 'auth' | 'rate_limit' | 'not_found' | 'network'
    detail: str


type PipelineError = DbError | DownloadError | SpotifyApiError
