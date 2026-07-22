"""Same-song guard: refuse attaching a stem to a recording that is a different song.

Pure decision function. All I/O (fingerprinting, DB lookups, title probing) is
the caller's job — this module only decides. Two channels, REFUSE if either
fires (fail-closed on a mismatch signal, not on absence of signal):

  1. title-token (primary): acquired source title vs target recording title.
  2. stem-aware chromaprint (corroboration): classify() vs the regular reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.labels import labels_overlap
from ingest.adapters.fingerprint import Fingerprint, classify, similarity

# classify() verdicts that mean "different recording" (not a wrong-stem signal).
_CONTENT_REFUSE = {"WRONG_SONG", "DURATION_MISMATCH"}


@dataclass(frozen=True)
class GuardVerdict:
    accept: bool
    channel: str | None  # "title" | "content" | None (which channel refused)
    reason: str


def same_song_guard(
    acquired_title: str,
    recording_title: str,
    stem_axis: str,
    fp_regular: Fingerprint | None,
    fp_candidate: Fingerprint | None,
) -> GuardVerdict:
    # Channel 1 — title-token (primary). Only decisive when both titles present.
    if acquired_title and recording_title:
        if not labels_overlap(acquired_title, recording_title):
            return GuardVerdict(
                False,
                "title",
                f"title-token disjoint: {acquired_title!r} vs {recording_title!r}",
            )

    # Channel 2 — content (corroboration). Only when both fingerprints present.
    if fp_regular is not None and fp_candidate is not None:
        sim = similarity(fp_regular.raw, fp_candidate.raw)
        dur_ratio = (
            fp_candidate.duration_s / fp_regular.duration_s
            if fp_regular.duration_s
            else 0.0
        )
        verdict, detail = classify(stem_axis, sim, dur_ratio)
        if verdict in _CONTENT_REFUSE:
            return GuardVerdict(False, "content", f"{verdict}: {detail}")

    return GuardVerdict(True, None, "accept")
