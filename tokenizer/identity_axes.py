"""Parse 1001tracklists row text into identity axes (version vs stem vs variant).

Re-exports canonical types from core.identity. Scrape-time Title Case tags are
converted to DB lowercase via core.identity helpers in materialize.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from core.identity import (
    DEFAULT_STEM,
    Variant,
    normalize_stem,
    normalize_variant,
    scrape_version_to_db,
)

VersionTag = Optional[Literal["Remix", "Rework", "AltVersion"]]
ClaimedStem = Literal["regular", "acappella", "instrumental"]

_ACAPELLA_RE = re.compile(
    r"\bacappella\b|\b(vocal|vox)\s+(only|version|mix)\b|\(\s*acap\s*\)",
    re.IGNORECASE,
)
_INSTRUMENTAL_RE = re.compile(
    r"\(\s*(instrumental|inst\.?|instr\.?)\s*\)"
    r"|\binstrumental(?:\s+(?:mix|version|edit))?\b"
    r"|\bdub\s*(?:mix|version)?\b"
    r"|\bkaraoke\b"
    r"|\(\s*instr\s*\)",
    re.IGNORECASE,
)


def derive_claimed_stem(
    full_name: str | None,
    row_text: str | None = None,
) -> ClaimedStem:
    """Stem axis from verbatim full_name + optional row text blob."""
    blob = " ".join(filter(None, (full_name or "", row_text or "")))
    if _ACAPELLA_RE.search(blob):
        return "acappella"
    if _INSTRUMENTAL_RE.search(blob):
        return "instrumental"
    return "regular"


def derive_version_flags(
    row_text: str,
    *,
    remix_flag: bool = False,
    has_recycle_rework: bool = False,
) -> tuple[bool, VersionTag]:
    """Version axis only — never returns Acappella (that's `derive_claimed_stem`)."""
    text_blob = row_text.lower()
    has_rework = " rework" in text_blob or has_recycle_rework
    has_remix = " remix" in text_blob

    is_remixish = remix_flag or has_rework or has_remix

    version_tag: VersionTag = None
    if has_rework or has_recycle_rework:
        version_tag = "Rework"
    elif has_remix:
        version_tag = "Remix"
    elif remix_flag:
        version_tag = "AltVersion"

    return is_remixish, version_tag


def scrape_claimed_version(scrape_tag: str | None) -> str:
    """DB lowercase version for set_track_slots.claimed_version."""
    return scrape_version_to_db(scrape_tag)


def scrape_claimed_stem(full_name: str | None, row_text: str | None = None) -> str:
    return normalize_stem(derive_claimed_stem(full_name, row_text))


def derive_claimed_variant(full_name: str | None, row_text: str | None = None) -> Variant:
    blob = " ".join(filter(None, (full_name or "", row_text or "")))
    if re.search(r"\bextended\s+(?:mix|version|edit)\b", blob, re.IGNORECASE):
        return "extended"
    if re.search(r"\(extended\)", blob, re.IGNORECASE):
        return "extended"
    return "regular"
