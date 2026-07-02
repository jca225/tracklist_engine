"""Per-span belief: precision-weighted fusion of probe observations.

A span's true placement is hidden; each probe emits an ``Observation`` —
a proposed placement (or an abstain), the probe's native confidence, and the
probe's *calibrated precision* (measured P(correct | probe fired), e.g. lyrics
~0.90, fp-sharp ~0.90 — the observation model of the POMDP). Fusion clusters
proposals on the set_start axis and scores the dominant cluster by its share
of total precision-weighted mass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

CLUSTER_TOL_S = 8.0  # proposals within this agree (matches stem-placement guard)


@dataclass(frozen=True)
class Observation:
    probe: str
    set_start_s: float | None  # None = the probe abstained
    confidence: float  # probe-native, [0, 1]
    precision: float  # calibrated prior for this probe, [0, 1]
    cost: float = 0.0
    ref_start_s: float | None = None
    detail: str = ""

    @property
    def weight(self) -> float:
        return self.precision * self.confidence

    @property
    def abstained(self) -> bool:
        return self.set_start_s is None


@dataclass(frozen=True)
class Cluster:
    set_start_s: float  # weight-averaged center
    weight: float
    share: float  # of total non-abstain mass
    probes: tuple[str, ...]


@dataclass(frozen=True)
class SpanBelief:
    slot_label: str
    recording_id: str
    claimed_stem: str
    observations: tuple[Observation, ...] = ()

    def observe(self, obs: Observation) -> SpanBelief:
        return replace(self, observations=(*self.observations, obs))

    def clusters(self, tol_s: float = CLUSTER_TOL_S) -> tuple[Cluster, ...]:
        """Greedy 1-D clustering of non-abstain proposals, heaviest first."""
        live = [o for o in self.observations if not o.abstained and o.weight > 0]
        total = sum(o.weight for o in live)
        if total <= 0:
            return ()
        remaining = sorted(live, key=lambda o: -o.weight)
        out: list[Cluster] = []
        while remaining:
            seed = remaining[0]
            members = [
                o for o in remaining if abs(o.set_start_s - seed.set_start_s) <= tol_s
            ]
            remaining = [o for o in remaining if o not in members]
            w = sum(o.weight for o in members)
            center = sum(o.set_start_s * o.weight for o in members) / w
            out.append(
                Cluster(
                    set_start_s=center,
                    weight=w,
                    share=w / total,
                    probes=tuple(o.probe for o in members),
                )
            )
        return tuple(sorted(out, key=lambda c: -c.weight))

    def best(self) -> Cluster | None:
        cs = self.clusters()
        return cs[0] if cs else None

    def quality(self) -> float:
        """Belief quality in [0, 1] — drives the permission ladder.

        share-of-mass × best member precision: high only when the dominant
        cluster both wins the vote AND contains a trustworthy probe. All-abstain
        or empty → 0.0 (the escalate signal).
        """
        top = self.best()
        if top is None:
            return 0.0
        best_precision = max(
            o.precision
            for o in self.observations
            if not o.abstained and o.probe in top.probes
        )
        return top.share * best_precision

    def probes_run(self) -> frozenset[str]:
        return frozenset(o.probe for o in self.observations)

    def cost_spent(self) -> float:
        return sum(o.cost for o in self.observations)
