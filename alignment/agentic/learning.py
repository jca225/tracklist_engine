"""A learning harness: the probe registry as a Bayesian bandit.

The hand-set precisions in ``actions.py`` are the POMDP's observation model
P(correct | probe fired) — but hand-set. This module makes them *learned*: each
probe's precision is a Beta posterior updated from outcomes, so the harness

  1. **calibrates** each probe's precision from evidence (posterior mean feeds
     the ladder's quality gate), and
  2. **selects** which probe to run next by Thompson sampling (draw a precision
     from each candidate's posterior, pick best info-per-cost) — the classic
     bandit answer to "learn which tools to use", exploring uncertain probes and
     exploiting proven ones.

Why this is safe where RL is not (design doc tier 3): a conjugate Bayesian
bandit *calibrates* — it never optimizes a learned reward, so there is no
reward to hack and it works at n=2 labeled sets. The `validated=False` auditory
probes get wide priors and are *explored* automatically; as they accumulate
correct fires their posteriors sharpen and they enter contention — the harness
**learns the dominance table** instead of us hand-authoring it.

Determinism: updates are pure Beta arithmetic. Thompson sampling takes an
explicit ``numpy.random.Generator`` so runs are reproducible (seed in, same
draws out) — respecting the harness's append-only, replayable ethos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from alignment.agentic.actions import REGISTRY, ActionSpec
from alignment.agentic.events import Event


@dataclass(frozen=True)
class Beta:
    """Beta(α, β) belief over one probe's precision. mean = α/(α+β)."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        """Total evidence mass (pseudo-counts + observed) — the confidence."""
        return self.alpha + self.beta

    def update(self, correct: bool, weight: float = 1.0) -> Beta:
        return (
            Beta(self.alpha + weight, self.beta)
            if correct
            else Beta(self.alpha, self.beta + weight)
        )

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta))


def _prior(spec: ActionSpec) -> Beta:
    """Seed a Beta prior from the registry precision.

    Validated probes get a CONFIDENT prior (strong pseudo-counts) so a few
    contrary observations don't unseat proven channels; unvalidated probes get a
    WEAK prior centered on their provisional guess, so the bandit explores them
    and lets evidence dominate quickly.
    """
    strength = 20.0 if spec.validated else 4.0
    p = min(max(spec.precision, 1e-3), 1 - 1e-3)
    return Beta(alpha=p * strength, beta=(1 - p) * strength)


@dataclass
class PrecisionModel:
    """Per-probe Beta posteriors, seeded from the registry, updated from outcomes."""

    posteriors: dict[str, Beta] = field(default_factory=dict)

    @classmethod
    def seeded(cls, registry: dict[str, ActionSpec] | None = None) -> PrecisionModel:
        reg = registry or REGISTRY
        return cls({name: _prior(spec) for name, spec in reg.items()})

    def precision(self, name: str) -> float:
        """Posterior-mean precision — the calibrated value the ladder should use."""
        b = self.posteriors.get(name)
        return b.mean if b else REGISTRY[name].precision if name in REGISTRY else 0.5

    def confidence(self, name: str) -> float:
        b = self.posteriors.get(name)
        return b.n if b else 0.0

    def observe(self, name: str, correct: bool, weight: float = 1.0) -> None:
        if name not in self.posteriors:
            self.posteriors[name] = _prior(
                REGISTRY.get(name, ActionSpec(name, 0.5, 0.5, ()))
            )
        self.posteriors[name] = self.posteriors[name].update(correct, weight)

    def sample(self, name: str, rng: np.random.Generator) -> float:
        b = self.posteriors.get(name)
        return b.sample(rng) if b else 0.5

    # ---- selection: Thompson-sampled info-per-cost ranking -----------------

    def rank(
        self,
        candidates: tuple[str, ...],
        rng: np.random.Generator,
        *,
        cost_floor: float = 0.05,
    ) -> tuple[str, ...]:
        """Order ``candidates`` by Thompson-sampled precision per unit cost.

        A learned replacement for the static DOMINANCE plan: exploration comes
        for free from the posterior draw (uncertain probes occasionally sample
        high and get tried), exploitation from proven probes' tight posteriors.
        """

        def score(name: str) -> float:
            cost = max(REGISTRY[name].cost if name in REGISTRY else 1.0, cost_floor)
            return self.sample(name, rng) / cost

        return tuple(sorted(candidates, key=score, reverse=True))

    # ---- learning from the append-only event log ---------------------------

    def fit_from_events(
        self,
        events: tuple[Event, ...],
        correct_fn,
        *,
        weight: float = 1.0,
    ) -> int:
        """Replay observe-events, labeling each via ``correct_fn`` and updating.

        ``correct_fn(span_key, probe, payload) -> bool | None`` returns whether
        the probe's proposal was right (None = unlabelable, skipped). This is the
        harness learning from its own audit trail — the design doc's "measure
        which actions ever led to correct commits, re-tune the cost order
        empirically", made Bayesian and online.
        """
        n = 0
        for ev in events:
            if ev.kind != "observe":
                continue
            probe = ev.payload.get("probe")
            if probe is None or ev.payload.get("set_start_s") is None:
                continue  # abstentions carry no correctness signal
            verdict = correct_fn(ev.span_key, probe, ev.payload)
            if verdict is None:
                continue
            self.observe(probe, bool(verdict), weight)
            n += 1
        return n

    # ---- persistence (small JSON; posteriors ARE the learned state) --------

    def to_json(self) -> str:
        return json.dumps(
            {k: [b.alpha, b.beta] for k, b in sorted(self.posteriors.items())},
            indent=1,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> PrecisionModel:
        raw = json.loads(path.read_text())
        return cls({k: Beta(a, b) for k, (a, b) in raw.items()})

    def report(self) -> str:
        lines = []
        for name in sorted(self.posteriors, key=self.precision, reverse=True):
            b = self.posteriors[name]
            val = (
                ""
                if REGISTRY.get(name, None) and REGISTRY[name].validated
                else " (prov)"
            )
            lines.append(f"  {name:16} precision={b.mean:.3f}  n={b.n:.0f}{val}")
        return "learned precisions:\n" + "\n".join(lines)


def gt_correct_fn(gt_by_recording, id_of, *, tol_s: float = 15.0):
    """Build a ``correct_fn`` that labels a probe proposal by GT distance.

    ``gt_by_recording[rec_id]`` -> list of GT rows with ``set_start_s``;
    ``id_of(span_key)`` -> the span's recording id. A proposal within ``tol_s``
    of any GT row for that recording counts as correct.
    """

    def _fn(span_key, probe, payload):
        rec = id_of(span_key)
        rows = gt_by_recording.get(rec)
        if not rows:
            return None
        pred = payload.get("set_start_s")
        if pred is None:
            return None
        return min(abs(pred - r["set_start_s"]) for r in rows) <= tol_s

    return _fn
