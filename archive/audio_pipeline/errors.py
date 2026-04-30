from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DbError:
    kind: str  # 'not_found' | 'query_failed' | 'integrity'
    detail: str


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
