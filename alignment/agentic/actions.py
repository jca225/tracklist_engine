"""Action registry: probes as uniform actions with cost tags + dominance pruning.

Each probe registers an ``ActionSpec`` (cost in relative units, calibrated
precision, applicability by stem). ``plan_for(stem)`` returns the measured
dominance order — cheapest high-information first, dominated probes pruned —
so the loop never spends a call we've already proven won't help (design doc
§action-space optimization). Runners are bound at loop time: the spec is the
schema, the runner is the implementation (replay adapters today, live probe
calls later) — new probes plug in without touching the loop.

Costs/precisions are measured on the BB11/BB12 chains: fp ≈ free when cached
but only ~53% clean overall (recalibrated 2026-07-06 by recording_id join —
see the fp spec below; the old 0.90 was optimistic and set-unstable); lyrics
~76% clean (also recalibrated 2026-07-06, was 0.90) with cached transcripts
(first transcription is the expensive part); HuBERT minutes-scale; cue prior
measures ~0.76 but is pinned to a 0.50 trust-floor (see its spec — `precision`
doubles as the auto-commit gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from alignment.agentic.belief import Observation

STEMS = ("regular", "acappella", "instrumental")


@dataclass(frozen=True)
class ActionSpec:
    name: str
    cost: float  # relative units (fp-cached ≈ 0.1)
    precision: float  # calibrated P(correct | probe fired)
    stems: tuple[str, ...]  # where this action is NOT dominated
    validated: bool = True  # False = precision is a PROVISIONAL guess, not GT-measured


REGISTRY: dict[str, ActionSpec] = {
    s.name: s
    for s in (
        # cue prior: free, and structurally blind on w-rows (cue=0.0 there).
        # Its measured precision is actually ~0.76 (BB11 74% / BB12 88%), but it
        # is deliberately kept LOW here: `precision` doubles as the auto-commit
        # gate, and at 0.76 the free scraped cue clears 0.75 SOLO and rubber-
        # stamps nearly every span (measured 2026-07-06: BB11 auto 27→149 spans,
        # cleanliness 100%→74%, loop stops running real probes). 0.50 is a trust
        # FLOOR that keeps cue out of solo auto-commit — do not "correct" it up
        # without separating fusion-weight from the auto-commit gate.
        ActionSpec("cue_prior", cost=0.0, precision=0.50, stems=STEMS),
        # MERT decode: the anchored coarse placement (always available)
        ActionSpec("mert_decode", cost=0.5, precision=0.55, stems=STEMS),
        # landmark fingerprint: cached, sharp on regular content.
        # fp fires on vocals too in practice (MERT-gated) — weaker there, kept
        # as the cheap fallback between lyrics and HuBERT.
        # precision MEASURED 2026-07-06 across BB11+BB12 (recording_id join,
        # n=230): 121/230 = 0.53, NOT the 0.90 prior — and set-UNSTABLE
        # (BB11 45%, BB12 62%). At 0.53 fp can no longer clear the 0.75
        # auto-commit bar alone (correct — 53% is not commit-grade); it now
        # commits only when a high-precision probe agrees in its cluster.
        ActionSpec("fp", cost=0.1, precision=0.53, stems=STEMS),
        # lyrics (Whisper + IDF diagonals): THE vocal channel; cached after first
        # run. precision MEASURED 2026-07-06 (BB11+BB12, recording_id join,
        # n=96): 73/96 = 0.76 (stable: BB11 77% / BB12 75%), down from the 0.90
        # prior. Still clears the 0.75 solo auto-commit bar (verified auto-
        # neutral: rung unchanged at 100% clean), but the margin is now thin.
        ActionSpec(
            "lyrics", cost=1.0, precision=0.76, stems=("acappella",)
        ),  # EXPERIMENT
        # banded HuBERT joint placement: vocal fallback when lyrics abstains
        ActionSpec("stem_hubert", cost=3.0, precision=0.75, stems=("acappella",)),
        # chroma matched-filter refine (instrumental/regular ref offsets)
        ActionSpec(
            "chroma_refine", cost=0.5, precision=0.70, stems=("regular", "instrumental")
        ),
        # --- auditory-neuroscience probes (auditory.py) — PROVISIONAL precisions,
        #     unvalidated until an eval pass measures P(correct|fired) on GT. The
        #     ladder must treat validated=False as suggest-only, never auto-commit.
        # onset-envelope cross-correlation placement: the flagship onset-alignment
        # probe. Cheap; a DJ's beatmatch preserves the onset grid.
        ActionSpec(
            # precision MEASURED on BB11 (eval_auditory): full-ref xcorr; the
            # peak-gated short-excerpt variant is the untested improvement path
            "onset_align",
            cost=0.3,
            precision=0.36,
            stems=STEMS,
            validated=False,
        ),
        # onset-asynchrony source-count: "is a second (overlay) onset stream here"
        ActionSpec(
            "onset_async",
            cost=0.2,
            precision=0.55,
            stems=("acappella", "instrumental"),
            validated=False,
        ),
        # harmonic-sieve pitch-invariant identity: robust to DJ re-pitching
        ActionSpec(
            "harmonic_sieve",
            cost=0.4,
            precision=0.60,
            stems=("acappella",),
            validated=False,
        ),
        # modulation-spectrum tempo/stretch estimate (first-class ref→mix ratio)
        ActionSpec(
            "modulation", cost=0.3, precision=0.55, stems=STEMS, validated=False
        ),
        # predictive-coding surprise: Foote novelty on mix MERT, stem-routed
        # (acappella reads mix_vocals novelty), snapped to cue else mert band
        # center (novelty.py). Precision MEASURED vs BB11+BB12 hand GT
        # (2026-07-06) per (stem, center): reg/instr@cue 0.73, acap@cue 0.58,
        # reg/instr@mert 0.48, acap@mert 0.37; the runner stamps each
        # observation with its own population's precision — this registry
        # value is only the prior/bandit seed.
        ActionSpec("surprise", cost=0.1, precision=0.60, stems=STEMS),
        # tempogram: onset-autocorrelation beat-grid → tempo + pulse strength
        ActionSpec(
            "tempogram",
            cost=0.2,
            precision=0.55,
            stems=("regular", "instrumental"),
            validated=False,
        ),
        # common-fate comodulation source-count (how a source evolves)
        ActionSpec(
            "common_fate",
            cost=0.3,
            precision=0.55,
            stems=("acappella", "instrumental"),
            validated=False,
        ),
        # cocktail-party belief-shaping action: subtract a committed bed, re-probe
        # the residual for the overlay (old-plus-new). Not a placement probe on its
        # own — it re-runs a stem probe on the residual signal.
        ActionSpec(
            "cocktail_party",
            cost=2.0,
            precision=0.65,
            stems=("acappella",),
            validated=False,
        ),
    )
}

# Measured dominance order per stem (design doc table): run left-to-right,
# commit early when the ladder's gate clears. Pruned probes simply absent.
# surprise sits LAST in every lane: spans that clear the ladder never reach it
# (a ~0.6-precision vote diluting an already-confident belief costs auto
# coverage — measured), so it fires only on unresolved spans, where it sharpens
# the review queue. Live-mode ordering also needs mert's proposal to exist
# first (the w-row band center; scraped cue is fake 0.0 there).
DOMINANCE: dict[str, tuple[str, ...]] = {
    "regular": ("cue_prior", "mert_decode", "fp", "chroma_refine", "surprise"),
    "instrumental": ("cue_prior", "mert_decode", "fp", "chroma_refine", "surprise"),
    "acappella": (
        "cue_prior",
        "mert_decode",
        "lyrics",
        "fp",
        "stem_hubert",
        "surprise",
    ),
}

# Composite skills — one decision covers a proven sequence.
SKILLS: dict[str, tuple[str, ...]] = {
    "resolve_vocal_span": ("lyrics", "stem_hubert"),
    "resolve_host_span": ("fp", "chroma_refine"),
}

# A runner executes one action for one span context and returns an Observation.
Runner = Callable[[dict], Observation]


def plan_for(stem: str) -> tuple[ActionSpec, ...]:
    names = DOMINANCE.get(stem or "regular", DOMINANCE["regular"])
    return tuple(REGISTRY[n] for n in names)


def bind(runners: dict[str, Runner]) -> dict[str, Runner]:
    """Validate a runner set against the registry (unknown names are bugs)."""
    unknown = set(runners) - set(REGISTRY)
    if unknown:
        raise ValueError(f"runners for unregistered actions: {sorted(unknown)}")
    return runners
