"""Three-axis recording identity — version, stem, variant.

Axes are orthogonal and can be concatenated into a single lookup key::

    {version}__{stem}__{variant}
    remix__acappella__extended

When a remixer is known, version_artist is stored separately on ``recording``
(not in the key) — the UNIQUE constraint is
(work_id, version, version_artist, stem, variant).

Vocabulary (canonical lowercase in DB/code):

| Axis    | Values |
|---------|--------|
| version | original, remix, rework, altversion, edit, bootleg, mashup |
| stem    | regular, acappella, instrumental |
| variant | regular, extended |

Legacy mapping (Phase 1 migration / ingest compat):

- ``track_audio.variant_tag='original'`` → stem ``regular``
- ``full`` (tokenizer interim) → ``regular``
- scrape ``version_tag='Remix'`` → version ``remix``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Optional

Version = Literal[
    "original", "remix", "rework", "altversion", "edit", "bootleg", "mashup",
]
Stem = Literal["regular", "acappella", "instrumental"]
Variant = Literal["regular", "extended"]

DEFAULT_VERSION: Final[Version] = "original"
DEFAULT_STEM: Final[Stem] = "regular"
DEFAULT_VARIANT: Final[Variant] = "regular"

AXES_SEP: Final[str] = "__"

# Scrape-time Title Case → DB lowercase
_VERSION_FROM_SCRAPE: Final[dict[str | None, Version]] = {
    None: "original",
    "Remix": "remix",
    "Rework": "rework",
    "AltVersion": "altversion",
    "Acappella": "original",  # mis-tagged scrape rows: stem handled separately
}

_STEM_ALIASES: Final[dict[str, Stem]] = {
    "full": "regular",
    "original": "regular",
    "regular": "regular",
    "acappella": "acappella",
    "acapella": "acappella",
    "instrumental": "instrumental",
}


@dataclass(frozen=True)
class RecordingAxes:
    version: Version = DEFAULT_VERSION
    stem: Stem = DEFAULT_STEM
    variant: Variant = DEFAULT_VARIANT
    version_artist: str | None = None

    def key(self) -> str:
        """Concatenated axes: ``version__stem__variant`` (all lowercase)."""
        return f"{self.version}{AXES_SEP}{self.stem}{AXES_SEP}{self.variant}"

    def display_suffix(self) -> str:
        """Human-readable parenthetical for filenames, e.g. `` (Quintino Remix)``."""
        if self.version == "remix" and self.version_artist:
            return f" ({self.version_artist} Remix)"
        if self.version == "rework" and self.version_artist:
            return f" ({self.version_artist} Rework)"
        if self.version not in ("original",) and self.version_artist:
            return f" ({self.version_artist} {self.version.title()})"
        if self.version == "remix":
            return " (Remix)"
        if self.version == "rework":
            return " (Rework)"
        if self.stem == "acappella":
            return " (Acappella)"
        if self.stem == "instrumental":
            return " (Instrumental)"
        if self.variant == "extended":
            return " (Extended Mix)"
        return ""


def normalize_version(raw: str | None) -> Version:
    if raw is None or not str(raw).strip():
        return DEFAULT_VERSION
    s = str(raw).strip()
    if s in _VERSION_FROM_SCRAPE:
        return _VERSION_FROM_SCRAPE[s]
    low = s.lower()
    if low in (
        "original", "remix", "rework", "altversion", "edit", "bootleg", "mashup",
    ):
        return low  # type: ignore[return-value]
    return DEFAULT_VERSION


def normalize_stem(raw: str | None) -> Stem:
    if raw is None:
        return DEFAULT_STEM
    return _STEM_ALIASES.get(str(raw).strip().lower(), DEFAULT_STEM)


def normalize_variant(raw: str | None) -> Variant:
    if raw is None or not str(raw).strip():
        return DEFAULT_VARIANT
    low = str(raw).strip().lower()
    return "extended" if low == "extended" else DEFAULT_VARIANT


def parse_axes_key(key: str) -> RecordingAxes:
    """Parse ``version__stem__variant``; extra segments are ignored."""
    parts = key.split(AXES_SEP)
    if len(parts) < 3:
        return RecordingAxes()
    return RecordingAxes(
        version=normalize_version(parts[0]),
        stem=normalize_stem(parts[1]),
        variant=normalize_variant(parts[2]),
    )


def scrape_version_to_db(scrape_tag: str | None) -> Version:
    """Map tokenizer Title Case ``version_tag`` to DB ``version``."""
    return normalize_version(_VERSION_FROM_SCRAPE.get(scrape_tag, scrape_tag))
