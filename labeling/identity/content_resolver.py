"""Content-addressed clip → identity resolution (Operation Crush §9).

The one resolver, with **no fallback ladder**: a clip binds to identity by
content — Ableton's own `OriginalFileSize`+`OriginalCrc`, then a head hash of the
resolved bytes — and on a miss it **abstains loudly** with a diagnostic. It never
guesses identity from a filename or slot number. Deleting the old guess-ladder
(the slot-label id bridge, the weak tiers of `match_manifest_for_path`) in favour
of this is what kills the GT-poisoning class by construction.

Paths are locators, never identity: the only role `clip.path` plays here is as
the argument handed to an injected `head_hash_of` when content bytes must be read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.result import Err, Ok, Result
from labeling.als.models import ParsedClip


@dataclass(frozen=True)
class CatalogEntry:
    """One catalogued audio artifact, keyed by content."""

    track_audio_id: str
    recording_id: str | None
    stem: str  # regular | acappella | instrumental
    file_size: int | None = None
    crc: int | None = None
    head_hash: str | None = None
    variant: str = "regular"  # regular | extended
    id_source: str = "content"  # content | historical-content
    cert: str | None = None  # payload | derivation | None (live row)


@dataclass(frozen=True)
class ContentCatalog:
    by_size_crc: dict[tuple[int, int], CatalogEntry]
    by_head_hash: dict[str, CatalogEntry]

    @classmethod
    def from_entries(cls, entries: Iterable[CatalogEntry]) -> "ContentCatalog":
        """Build the two content-hash indices, hard-abstaining on axis conflicts.

        A key (size,crc) or head_hash that maps to entries disagreeing on the
        identity axis tuple `(recording_id, stem, variant)` is dropped entirely
        rather than last-writer-wins: getting the work right but the
        stem/variant wrong is a wrong label, so an ambiguous key must resolve
        to nothing (a lookup miss → abstain), not an arbitrarily chosen entry.
        """
        by_sc: dict[tuple[int, int], CatalogEntry] = {}
        by_hh: dict[str, CatalogEntry] = {}
        sc_axes: dict[tuple[int, int], set[tuple[str | None, str, str]]] = {}
        hh_axes: dict[str, set[tuple[str | None, str, str]]] = {}
        sc_ambiguous: set[tuple[int, int]] = set()
        hh_ambiguous: set[str] = set()
        for e in entries:
            axis = (e.recording_id, e.stem, e.variant)
            if e.file_size is not None and e.crc is not None:
                sc_key = (e.file_size, e.crc)
                axes = sc_axes.setdefault(sc_key, set())
                axes.add(axis)
                if len(axes) > 1:
                    sc_ambiguous.add(sc_key)
                by_sc[sc_key] = e
            if e.head_hash:
                axes = hh_axes.setdefault(e.head_hash, set())
                axes.add(axis)
                if len(axes) > 1:
                    hh_ambiguous.add(e.head_hash)
                by_hh[e.head_hash] = e
        for key in sc_ambiguous:
            del by_sc[key]
        for key in hh_ambiguous:
            del by_hh[key]
        return cls(by_size_crc=by_sc, by_head_hash=by_hh)


@dataclass(frozen=True)
class ClipIdentity:
    track_audio_id: str
    recording_id: str | None
    stem: str
    variant: str
    matched_by: str  # "size_crc" | "head_hash"
    id_source: str = "content"  # content | historical-content
    cert: str | None = None


@dataclass(frozen=True)
class ResolveDiagnostic:
    """Why a clip could not be content-bound — a worklist entry, not a guess."""

    reason: str  # "no_content_match"
    file_size: int | None
    crc: int | None
    path: str


def resolve_clip_identity(
    clip: ParsedClip,
    catalog: ContentCatalog,
    *,
    head_hash_of: Callable[[str], str | None] | None = None,
) -> Result[ClipIdentity, ResolveDiagnostic]:
    """Bind a clip to identity by content, or abstain. No filename/slot guessing.

    Order: exact (OriginalFileSize, OriginalCrc) → head hash of the resolved
    bytes (only if `head_hash_of` is supplied) → Err. There is no further tier.
    """
    if clip.file_size is not None and clip.crc is not None:
        entry = catalog.by_size_crc.get((clip.file_size, clip.crc))
        if entry is not None:
            return Ok(
                ClipIdentity(
                    track_audio_id=entry.track_audio_id,
                    recording_id=entry.recording_id,
                    stem=entry.stem,
                    variant=entry.variant,
                    matched_by="size_crc",
                    id_source=entry.id_source,
                    cert=entry.cert,
                )
            )

    if head_hash_of is not None:
        digest = head_hash_of(clip.path)
        if digest is not None:
            entry = catalog.by_head_hash.get(digest)
            if entry is not None:
                return Ok(
                    ClipIdentity(
                        track_audio_id=entry.track_audio_id,
                        recording_id=entry.recording_id,
                        stem=entry.stem,
                        variant=entry.variant,
                        matched_by="head_hash",
                        id_source=entry.id_source,
                        cert=entry.cert,
                    )
                )

    return Err(
        ResolveDiagnostic(
            reason="no_content_match",
            file_size=clip.file_size,
            crc=clip.crc,
            path=clip.path,
        )
    )
