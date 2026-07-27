from __future__ import annotations

"""Stem "separation of powers" router for the PWS aligner (Workstream F).

Each operation LF has JURISDICTION over one or more set-audio stems: running
echo-tail on a vocal stem is valid; running sidechain-pump on a vocal stem is
not (the pump lives in the instrumental bed, not the vocal).  This module
encodes that competence table and provides a thin dispatcher that calls each LF
only on the stems where it is competent.

Design principles:
  - Pure routing / wrapping layer.  No audio loading, no DSP.
  - Wraps votes rather than mutating OperationVote (no breaking change to
    existing vote contracts).
  - Stem-agnostic LFs (cue_time, claimed_axes_identity, llm, …) are NOT
    listed in LF_COMPETENCE; ``competent_stems()`` returns the safe default
    ``(Stem.FULL,)`` for any unknown LF name.  This is the right sentinel
    because the full mix is the universal fallback: a stem-agnostic LF that
    receives the full mix behaves identically to the original stem-blind call.

Payoffs (see plan §F):
  1. Correctness — e.g. echo on a vocal-only stem is meaningful; noise-sweep
     on a vocal stem is not (broadband noise = separation bleed, not a riser).
  2. FABLE instance feature — the label model learns per-(LF, stem) accuracy.
  3. Vote density — running each LF on N_competent_stems multiplies co-voting
     coverage, the direct fix for the Gate-v3 σ-sparsity bottleneck.

Caveat (document in fit_corpus): Roformer separation bleed means mix_vocal /
mix_instrumental votes carry artifact noise → lower learned per-(LF,stem)
accuracy.  The label model absorbs this rather than hiding it.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Stem enum
# ---------------------------------------------------------------------------


class Stem(Enum):
    """Which set-audio stem an LF is running on."""

    FULL = "full"
    VOCAL = "vocal"
    INSTRUMENTAL = "instrumental"


# ---------------------------------------------------------------------------
# RoutedVote — stem-tagged vote wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutedVote:
    """A stem-tagged wrapper around any underlying LF vote.

    Does NOT mutate or subclass the wrapped vote; downstream consumers can
    inspect ``rv.stem`` for the stem tag and ``rv.vote`` for the original
    OperationVote / ClaimedIdentityVote / etc.

    Fields:
        stem: which stem the underlying vote was computed on.
        vote: the underlying vote object (OperationVote, ClaimedIdentityVote,
              or any future vote type — typed as ``object`` to avoid coupling).
    """

    stem: Stem
    vote: object


# ---------------------------------------------------------------------------
# Competence table
# ---------------------------------------------------------------------------

# Jurisdiction: which stems each named LF is allowed to run on.
# Rationale per LF:
#
#   echo_tail         FULL + VOCAL — echo throws appear on the vocal channel;
#                     the full mix preserves them too.  Instrumental bed rarely
#                     carries a throw (it would blend into sustained pads).
#
#   noise_sweep       FULL only — a broadband riser is a full-mix bed element.
#                     On an isolated vocal stem, broadband noise = separation
#                     bleed (Roformer artifact), not a genuine riser.
#
#   filter_sweep      FULL + INSTRUMENTAL — tonal filter sweeps live in the
#                     instrumental bed (synth/bass cutoff sweeping).  Present
#                     in the full mix too.  Vocal-only: filter sweeps on a
#                     voice are rare and separation quality too low to trust.
#
#   sidechain         INSTRUMENTAL + FULL — sidechain pump is a bed effect
#                     (kick-ducking the sustained pad/bass).  Present in full
#                     mix.  Vocal stem: ducking rarely applied to lead vocal.
#
#   loop              FULL + VOCAL + INSTRUMENTAL — DJ loops can be applied to
#                     any stem (vocal loop, instrumental bed loop, or the whole
#                     mix), so all three are valid and maximise vote density.
#
#   keylock_vs_varispeed  INSTRUMENTAL only — pitch-shift detection requires a
#                         tonal carrier (pitched instruments).  A vocal stem
#                         works in principle, but pitch-tracking is far less
#                         reliable on voice under separation bleed; INSTRUMENTAL
#                         is the cleaner carrier and beats FULL (vocals muddy
#                         the CQT profile).
#
# Stem-agnostic LFs (cue_time, claimed_axes_identity, llm, ordering_soft, …)
# are NOT in this table.  competent_stems() returns (Stem.FULL,) for them —
# the full mix is the universal stand-in for a stem-blind call.

LF_COMPETENCE: dict[str, tuple[Stem, ...]] = {
    "echo_tail": (Stem.FULL, Stem.VOCAL),
    "noise_sweep": (Stem.FULL,),
    "filter_sweep": (Stem.FULL, Stem.INSTRUMENTAL),
    "sidechain": (Stem.INSTRUMENTAL, Stem.FULL),
    "loop": (Stem.FULL, Stem.VOCAL, Stem.INSTRUMENTAL),
    "keylock_vs_varispeed": (Stem.INSTRUMENTAL,),
}

# Default competence for any LF not in the table (stem-agnostic).
_DEFAULT_COMPETENCE: tuple[Stem, ...] = (Stem.FULL,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def competent_stems(lf_name: str) -> tuple[Stem, ...]:
    """Return the stems this LF is competent to run on.

    For any LF name not in LF_COMPETENCE (including stem-agnostic LFs such as
    ``cue_time`` and ``claimed_axes_identity``), returns ``(Stem.FULL,)`` — the
    universal fallback that replicates the original stem-blind call behaviour.
    """
    return LF_COMPETENCE.get(lf_name, _DEFAULT_COMPETENCE)


def route_audio_lf(
    lf_name: str,
    span_id: str,
    stem_audios: Mapping[Stem, tuple[np.ndarray, int]],
    call_fn: Callable[[Stem, np.ndarray, int], Any],
) -> list[RoutedVote]:
    """Dispatch an LF to each of its competent stems that are present.

    For each stem in ``competent_stems(lf_name)`` that also appears in
    ``stem_audios``, calls ``call_fn(stem, y, sr)`` and wraps the result in a
    ``RoutedVote`` tagged with that stem.  Stems absent from ``stem_audios``
    are silently skipped (no error — the caller controls which stems are
    available at runtime).

    Args:
        lf_name:     name of the LF being dispatched (used for competence
                     lookup via ``competent_stems``).
        span_id:     span identifier forwarded to call_fn if needed; the
                     dispatcher itself does not inspect it.
        stem_audios: mapping from Stem → (audio_array, sample_rate).  Only
                     stems present here will produce votes.
        call_fn:     ``(stem, y, sr) → vote`` — the caller binds the specific
                     LF and any extra parameters (e.g. beat_period_s) before
                     passing it here.  Returns any vote object; the dispatcher
                     wraps it without inspecting it.

    Returns:
        List of RoutedVote, one per competent stem that was present in
        stem_audios.  Empty list if no competent stems are present.
    """
    results: list[RoutedVote] = []
    for stem in competent_stems(lf_name):
        if stem not in stem_audios:
            continue
        y, sr = stem_audios[stem]
        vote = call_fn(stem, y, sr)
        results.append(RoutedVote(stem=stem, vote=vote))
    return results
