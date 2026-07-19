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


# Probes sharing an evidence source do NOT corroborate each other — only the
# strongest per group counts toward combined trust. Keeps a derivative probe
# agreeing with its parent (e.g. `surprise` snaps to `mert_decode`'s band
# centre) from being double-counted as independent agreement, and groups the
# two audio-content matched-filters (fp, chroma) that can fail together.
INDEPENDENCE_GROUP: dict[str, str] = {
    "cue_prior": "cue",  # scraped tracklist metadata
    "mert_decode": "mert",  # MERT audio decode
    "surprise": "mert",  # snaps to mert's centre — not independent of it
    "lyrics": "lyrics",  # Whisper vocal text
    "stem_hubert": "hubert",  # HuBERT vocal embedding
    "fp": "content",  # landmark fingerprint (audio content)
    "chroma_refine": "content",  # chroma matched-filter (audio content)
}


def independence_satisfied(belief: SpanBelief) -> bool:
    """True when the winning cluster meets the E1 G2 independence bar.

    ≥2 independent evidence groups, or one member with calibrated precision ≥0.9.
    Fiber ambiguity is handled separately (``pseudo_acceptance`` / G3).
    """
    top = belief.best()
    if top is None:
        return False
    members = tuple(
        obs
        for obs in belief.observations
        if not obs.abstained and obs.probe in top.probes
    )
    if not members:
        return False
    groups = {INDEPENDENCE_GROUP.get(obs.probe, obs.probe) for obs in members}
    return len(groups) >= 2 or max(obs.precision for obs in members) >= 0.9


def _combine_trust(members: tuple[Observation, ...]) -> float:
    """Noisy-OR of per-group max precision: independent agreeing probes
    corroborate, correlated ones don't double-count. One probe → its own
    precision (1 - (1-p) = p), so single-probe beliefs are unchanged.

    Interpretation: the members all agree on this cluster's location, so trust
    = P(at least one INDEPENDENT group is correct) = 1 - Π(1 - p_group). Uses
    marginal precision (not the agreement-boosted posterior), so it errs
    conservative — it can only under-credit agreement, never invent it.
    """
    if not members:
        return 0.0
    by_group: dict[str, float] = {}
    for o in members:
        g = INDEPENDENCE_GROUP.get(o.probe, o.probe)
        by_group[g] = max(by_group.get(g, 0.0), o.precision)
    prod = 1.0
    for p in by_group.values():
        prod *= 1.0 - p
    return 1.0 - prod


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

    def quality(self, *, combine: bool = False) -> float:
        """Belief quality in [0, 1] — drives the permission ladder.

        share-of-mass × cluster trust. All-abstain or empty → 0.0 (escalate).

        Trust is the max member precision by default: high only when the
        dominant cluster wins the vote AND contains a trustworthy probe. This
        deliberately ignores corroboration — three mediocre probes agreeing
        score no higher than the best of them, so a span with no single
        ≥auto-bar probe can never auto-commit even when several agree.

        With ``combine=True`` trust is instead ``_combine_trust`` — a noisy-OR
        over INDEPENDENT agreeing probes (correlated ones grouped, only the
        strongest per group counts). A single probe is unchanged (noisy-OR of
        one = itself), so existing single-probe commits are preserved; the only
        new behaviour is that independent agreement can clear a higher bar than
        any lone member. Keeps the escalate/abstain semantics identical.
        """
        top = self.best()
        if top is None:
            return 0.0
        members = tuple(
            o for o in self.observations if not o.abstained and o.probe in top.probes
        )
        trust = (
            _combine_trust(members) if combine else max(o.precision for o in members)
        )
        return top.share * trust

    def probes_run(self) -> frozenset[str]:
        return frozenset(o.probe for o in self.observations)

    def cost_spent(self) -> float:
        return sum(o.cost for o in self.observations)


# ---------------------------------------------------------------------------
# Precedence / temporal masking gate (auditory scene analysis)
# ---------------------------------------------------------------------------
# The ear suppresses a quiet sound sitting in the time-frequency shadow of a
# louder one (masking) or arriving just after a leading sound (precedence). The
# harness analog: an observation placing a quiet overlay *inside* a loud
# region's shadow is less trustworthy — down-weight its precision before it
# votes, so the LLM doesn't commit a vocal-under-drop placement on thin evidence.


def masking_gate(
    obs: Observation,
    loud_spans_s: tuple[tuple[float, float], ...],
    *,
    floor: float = 0.5,
) -> Observation:
    """Down-weight ``obs``'s precision when its placement lands in a loud span.

    ``loud_spans_s`` are (start, end) windows where a louder source dominates
    (e.g. the host bed's drops). An observation whose ``set_start_s`` falls in
    one is masked: its precision is scaled by ``floor`` (≤1). Non-overlay
    observations and placements outside every shadow are returned unchanged.
    """
    if obs.abstained or obs.set_start_s is None:
        return obs
    t = obs.set_start_s
    in_shadow = any(lo <= t <= hi for lo, hi in loud_spans_s)
    if not in_shadow:
        return obs
    return replace(
        obs, precision=obs.precision * floor, detail=(obs.detail + " [masked]").strip()
    )


# ---------------------------------------------------------------------------
# Fiber-instance ambiguity gate (B3 — repeat-class abstain)
# ---------------------------------------------------------------------------
# A decoded ref placement that lands inside a self-repeat class (chorus played
# 2-3x) is content-undecidable: every instance has identical audio, so which one
# played rests on the monotonic/continuity prior, NOT the signal. The repo's
# standing stance (path_decode --lam-back help; the --fibers metric) is to abstain
# on the *instance* while still trusting the *content* identity. This gate is the
# belief-layer expression of that: keep the observation but lower its precision so
# the ladder routes an ambiguous-instance placement to review rather than
# auto-committing a coin-flip. Complements masking_gate (same shape).


def fiber_gate(
    obs: Observation,
    labels,  # np.ndarray fiber labels for the ref, or None to no-op
    label_hz: float,
    *,
    mu=None,  # optional soft membership (compute_fibers_soft) — low mu also distrusts
    floor: float = 0.6,
    min_membership: float = 0.15,
) -> Observation:
    """Down-weight ``obs`` when its ref placement is instance-ambiguous.

    Uses ``ref_fibers.fiber_ambiguity`` on ``obs.ref_start_s``: if the placement
    falls in a fiber with >= 2 instances (``ambiguous``), or its soft membership
    is near zero, scale the precision by ``floor`` (≤1). Observations that
    abstained, carry no ref placement, or land outside any repeat class pass
    through unchanged — so single-instance content still auto-commits."""
    if obs.abstained or obs.ref_start_s is None or labels is None:
        return obs
    from workspaces.alignment_prototype.ref_fibers import fiber_ambiguity

    amb = fiber_ambiguity(labels, label_hz, obs.ref_start_s, mu=mu)
    low_membership = mu is not None and amb["membership"] < min_membership
    if not amb["ambiguous"] and not low_membership:
        return obs
    tag = (
        f" [fiber×{amb['n_instances']} amb]"
        if amb["ambiguous"]
        else f" [fiber μ={amb['membership']:.2f}]"
    )
    return replace(
        obs, precision=obs.precision * floor, detail=(obs.detail + tag).strip()
    )


# ---------------------------------------------------------------------------
# Predictive-coding surprise prior (info-dynamics → ordering-free placement)
# ---------------------------------------------------------------------------
# The brain predicts the next sound and spikes on surprise; a boundary/novelty
# spike marks where a *new* track likely entered. Unlike the cue prior and
# monotonic decode, this needs no tracklist order — it is a pure "something new
# started here" signal, usable to seed or sanity-check any span's placement.


def surprise_prior(
    surprise_curve: tuple[tuple[float, float], ...],
    *,
    precision: float = 0.45,
    min_z: float = 1.0,
) -> tuple[Observation, ...]:
    """Turn a (time_s, surprise) novelty curve into candidate-boundary Observations.

    Peaks in the surprise curve rising ``min_z`` standard deviations above the
    mean become weak, order-independent placement candidates (probe
    ``"surprise"``). Confidence scales with the peak's z-score. These seed the
    belief where boundary novelty says a track entered — cheap, and complementary
    to the identity probes.
    """
    if len(surprise_curve) < 3:
        return ()
    times = [t for t, _ in surprise_curve]
    vals = [v for _, v in surprise_curve]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sd = var**0.5
    if sd <= 0:
        return ()
    out: list[Observation] = []
    for i in range(1, len(vals) - 1):
        z = (vals[i] - mean) / sd
        if z >= min_z and vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            out.append(
                Observation(
                    probe="surprise",
                    set_start_s=times[i],
                    confidence=min(1.0, z / 4.0),
                    precision=precision,
                    detail=f"novelty z={z:.1f}",
                )
            )
    return tuple(out)
