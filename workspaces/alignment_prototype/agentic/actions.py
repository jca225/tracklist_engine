"""Action registry: probes as uniform actions with cost tags + dominance pruning.

Each probe registers an ``ActionSpec`` (cost in relative units, calibrated
precision, applicability by stem). ``plan_for(stem)`` returns the measured
dominance order — cheapest high-information first, dominated probes pruned —
so the loop never spends a call we've already proven won't help (design doc
§action-space optimization). Runners are bound at loop time: the spec is the
schema, the runner is the implementation (replay adapters today, live probe
calls later) — new probes plug in without touching the loop.

Costs/precisions are the 2026-07-02 measured values (BB11/BB12 chains):
fp ≈ free when cached and ~90% clean where sharp; lyrics ~90% clean with
cached transcripts (first transcription is the expensive part); HuBERT
minutes-scale; cue prior is free but blind on w-rows (cue=0.0 by scraping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from workspaces.alignment_prototype.agentic.belief import Observation

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
        # cue prior: free, weak, and structurally blind on w-rows
        ActionSpec("cue_prior", cost=0.0, precision=0.50, stems=STEMS),
        # MERT decode: the anchored coarse placement (always available)
        ActionSpec("mert_decode", cost=0.5, precision=0.55, stems=STEMS),
        # landmark fingerprint: cached, sharp on regular content
        # fp fires on vocals too in practice (MERT-gated) — weaker there, kept
        # as the cheap fallback between lyrics and HuBERT
        ActionSpec("fp", cost=0.1, precision=0.90, stems=STEMS),
        # lyrics (Whisper + IDF diagonals): THE vocal channel; cached after first run
        ActionSpec("lyrics", cost=1.0, precision=0.90, stems=("acappella",)),
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
            "onset_align", cost=0.3, precision=0.60, stems=STEMS, validated=False
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
        # predictive-coding surprise: order-free "a new track entered here" prior
        ActionSpec("surprise", cost=0.1, precision=0.45, stems=STEMS, validated=False),
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
DOMINANCE: dict[str, tuple[str, ...]] = {
    "regular": ("cue_prior", "mert_decode", "fp", "chroma_refine"),
    "instrumental": ("cue_prior", "mert_decode", "fp", "chroma_refine"),
    "acappella": ("cue_prior", "mert_decode", "lyrics", "fp", "stem_hubert"),
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
