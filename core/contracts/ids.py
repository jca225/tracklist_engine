"""Typed id namespaces + the canonical slot-label normal form.

Four id namespaces coexist in this repo and share raw value spaces (`str`),
which produced an 18-fix bug class (tlp id looked up as recording id, manifest
keyed one way and probed the other — see docs/entropy_reduction_plan.md).
`NewType` brands make the namespace part of a signature under mypy; they are
minted ONLY at trust boundaries (DB row adapters, artifact loaders) and stay
branded through interior code.

Known limit (accepted): NewType is one-way — a branded id still satisfies a
bare ``str`` parameter, and string ops degrade the brand. The guardrail is
signatures: never take a bare ``str`` where an id is meant.
"""

from __future__ import annotations

import re
from typing import NewType

SetId = NewType("SetId", str)
"""A 1001tracklists set id, e.g. ``1fsnxchk`` (BB12)."""

RecordingId = NewType("RecordingId", str)
"""Canonical recording id (``recording`` table; ≈ legacy track_id)."""

TlpId = NewType("TlpId", str)
"""Scrape-namespace id: ``tlp<data-id>`` for sided rows without data-trackid."""

TrackAudioId = NewType("TrackAudioId", int)
"""A physical download — ``track_audio.track_audio_id``."""

SlotLabel = NewType("SlotLabel", str)
"""A slot label in CANONICAL NORMAL FORM (see normalize_slot_label)."""


_SLOT_RE = re.compile(r"^0*(\d+)(w\d+)?$")


def normalize_slot_label(raw: str | None) -> SlotLabel | None:
    """Canonical slot-label form: strip zero-padding, keep the w-suffix.

    Three lexicons exist in the wild — DB/GT ``001w2``, timeline ``1w2``, and
    annotator-tagged names — and every consumer used to re-normalize its own
    way. One normal form, applied at artifact-load time: ``001w2`` -> ``1w2``,
    ``042`` -> ``42``. Returns None for empty/unparseable labels (callers
    decide whether that is a diagnostic).
    """
    if not raw:
        return None
    m = _SLOT_RE.match(raw.strip())
    if not m:
        return None
    return SlotLabel(m.group(1) + (m.group(2) or ""))
