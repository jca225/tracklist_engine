"""Download-side projection: collapse a canonical `full_name` into a YT Music
search query.

This is the *download view* of the tokenizer's lossless record. The tokenizer
keeps `full_name` verbatim (it carries `(Instrumental)`/`(Acappella)` markers so
the alignment-X spine can see them); the search query must drop those markers
because YT Music's `filter='songs'` index doesn't reliably carry isolated
vocal/instrumental cuts as separate releases — a bare query resolves to the
canonical (vocal) master, and Demucs splits stems downstream.

Two collapses happen here, both download-specific (they must NOT live in the
tokenizer):

  1. Strip vocal/instrumental qualifiers — `(Instrumental)`, `(Acappella)`, etc.
  2. Fall back to bare `"Artist - Title"` when `full_name` carries 1001tracklists'
     "ID" placeholder (`(ID Remix)`, `(ID Bootleg)`, …): a literal "ID" in the
     query derails search to random uploads.

The remixer qualifier (e.g. `(Madison Mars Remix)`) is deliberately preserved —
it steers search to the remix release rather than the original cut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VOCAL_QUALIFIER_RE: re.Pattern[str] = re.compile(
    r"\s*\(\s*(a[\s-]*cappella|acapella|instrumental|inst\.?|instr\.?)\s*\)",
    re.IGNORECASE,
)

_ID_PLACEHOLDER_RE: re.Pattern[str] = re.compile(
    r"\bID\b\s*(?:Remix|Bootleg|Edit|Mashup|VIP|Rework|Mix|Flip)?\b",
    re.IGNORECASE,
)

_ACAPELLA_QUERY_SUFFIXES = ("acapella", "acappella", "vocals only", "a cappella")
_INSTRUMENTAL_QUERY_SUFFIX = "instrumental"


@dataclass(frozen=True)
class TrackSearchMeta:
    full_name: str | None = None
    artists_csv: str | None = None
    title: str | None = None
    version: str | None = None
    claimed_stem: str = "regular"
    layer_role: str = "solo"


def to_search_query(
    full_name: str | None,
    artists_csv: str | None,
    title: str | None,
) -> str:
    """Build the YT Music search query for a track.

    Prefers the canonical `full_name` (with vocal qualifiers stripped) so the
    remix release is hit; falls back to `"Artist - Title"` when `full_name` is
    absent or carries an "ID" placeholder.
    """
    if full_name:
        stripped = _VOCAL_QUALIFIER_RE.sub("", full_name).strip()
        if stripped and not _ID_PLACEHOLDER_RE.search(stripped):
            return stripped
    if artists_csv:
        return f"{artists_csv} - {title}"
    return title or ""


def to_search_query_for_claim(
    *,
    full_name: str | None,
    artists_csv: str | None,
    title: str | None,
    claimed_stem: str = "regular",
    layer_role: str = "solo",
    version: str | None = None,
) -> str:
    """Role-aware YT Music search query."""
    base = to_search_query(full_name, artists_csv, title)

    if layer_role == "bed" or (version or "") in ("mashup", "bootleg"):
        return base

    if layer_role == "payload" or claimed_stem == "acappella":
        stripped = _VOCAL_QUALIFIER_RE.sub("", base).strip()
        if " - " in stripped:
            return f"{stripped} {_ACAPELLA_QUERY_SUFFIXES[0]}"
        return f"{base} {_ACAPELLA_QUERY_SUFFIXES[0]}"

    if claimed_stem == "instrumental":
        stripped = _VOCAL_QUALIFIER_RE.sub("", base).strip()
        return f"{stripped} {_INSTRUMENTAL_QUERY_SUFFIX}"

    return base


def to_search_query_for_meta(
    meta: TrackSearchMeta,
    *,
    layer_role: str = "solo",
) -> str:
    """Build query from metadata + optional layer role (redownload helper)."""
    return to_search_query_for_claim(
        full_name=meta.full_name,
        artists_csv=meta.artists_csv,
        title=meta.title,
        claimed_stem=meta.claimed_stem,
        layer_role=layer_role,
        version=meta.version,
    )
