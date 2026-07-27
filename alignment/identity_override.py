"""Gated identity override — Part C decision layer of Phase 1B (spec §2C).

Ties the multi-candidate pool ([candidate_pool.py]) to the blind open-set scorer
([open_set_identity.py]): for each slot, score the pool by stem-routed MERT
chamfer, then apply the fail-closed ``margin_gate`` against the tokenizer claim →
an ``IdentityDecision`` (accept-claim | override | abstain) plus the per-slot
identity provenance that seeds the ``AxisBelief(IDENTITY)`` (Law 21 explainability).

This module is PURE over injected features (mirrors ``open_set_identity`` /
``candidate_pool``): the L3/L22 stem-MERT arrays are produced by the extraction
runner ([extract_stem_mert]) and loaded by the ``infer.py`` seam, which then calls
``resolve_identities`` and rewrites each span's ``recording_id``. No DB/GPU/pi
here, so the decision logic is unit-testable in isolation.

First cut = **MERT chamfer alone** (spec §2B): the raw cosine ranking feeds
``margin_gate`` so ``tau``/``floor`` stay in cosine units. Passing ``extra_rankings``
switches to borda fusion — but then ``tau``/``floor`` are in borda units and must
be retuned; the single-LF path is the deployable default. ``tau``/``floor`` are
never defaulted here: they are tuned LOSO (one set → validate the other, Law 19)
and injected by the caller, so no value can silently leak across both sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alignment.candidate_pool import CandidatePool
from alignment.open_set_identity import (
    CandidateScore,
    Decision,
    IdentityDecision,
    borda_fuse,
    margin_gate,
    rank_by_chamfer,
)


@dataclass(frozen=True)
class IdentityResult:
    """A slot's identity decision + its provenance (the AxisBelief seed)."""

    slot_label: str
    decision: IdentityDecision
    provenance: dict


def _unscorable(pool: CandidatePool, reason: str) -> IdentityResult:
    """No audio evidence for this slot -> do no harm: keep the claim if there is
    one, else abstain. Never fabricate an override from missing features."""
    claim_id = pool.claim.recording_id if pool.claim else None
    decision = IdentityDecision(
        Decision.ACCEPT_CLAIM if claim_id is not None else Decision.ABSTAIN,
        claim_id,
        claim_id,
        None,
        None,
        None,
    )
    return IdentityResult(
        pool.slot_label,
        decision,
        {
            "slot_label": pool.slot_label,
            "pool_stem": pool.pool_stem,
            "pool_size": pool.size,
            "source_size": pool.source_size,
            "claim": claim_id,
            "chosen": claim_id,
            "top1": None,
            "margin": None,
            "top1_score": None,
            "decision": decision.decision,
            "scored": [],
            "reason": reason,
        },
    )


def decide_identity(
    pool: CandidatePool,
    query: np.ndarray | None,
    ref_features: dict[str, np.ndarray],
    *,
    tau: float,
    floor: float,
    span_s: float | None = None,
    extra_rankings: tuple[list[CandidateScore], ...] = (),
) -> IdentityResult:
    """Score ``pool`` against ``query`` and gate an override of the claim.

    ``query`` = (D, nq) stem-routed per-measure MERT for the mix span at the
    identity layer; ``ref_features`` maps recording_id -> (D, nr) for candidates
    (only the pool members need be present). Candidates with no features simply
    go unscored (``rank_by_chamfer`` trails them as None). ``extra_rankings`` are
    additional per-LF rankings to borda-fuse with the chamfer ranking (retune
    ``tau``/``floor`` to borda units if used).
    """
    # Restrict the ref set to this pool's candidates, in pool order.
    pool_refs = {
        c.recording_id: ref_features[c.recording_id]
        for c in pool.candidates
        if c.recording_id in ref_features
    }
    if query is None or query.size == 0 or not pool_refs:
        return _unscorable(
            pool,
            "no query" if (query is None or query.size == 0) else "no ref features",
        )

    chamfer_ranking = rank_by_chamfer(query, pool_refs, span_s=span_s)
    if extra_rankings:
        ranked = borda_fuse([chamfer_ranking, *extra_rankings])
    else:
        ranked = chamfer_ranking

    claim_id = pool.claim.recording_id if pool.claim else None
    decision = margin_gate(ranked, claim_id, tau=tau, floor=floor)

    provenance = {
        "slot_label": pool.slot_label,
        "pool_stem": pool.pool_stem,
        "pool_size": pool.size,
        "source_size": pool.source_size,
        "claim": claim_id,
        "chosen": decision.recording_id,
        "top1": decision.top1_id,
        "margin": decision.margin,
        "top1_score": decision.top1_score,
        "decision": decision.decision,
        "tau": tau,
        "floor": floor,
        # LF votes: every scorable candidate's score, best-first (Law 21).
        "scored": [
            (c.recording_id, float(c.score)) for c in ranked if c.score is not None
        ],
        "fused": bool(extra_rankings),
    }
    return IdentityResult(pool.slot_label, decision, provenance)


def resolve_identities(
    pools: dict[str, CandidatePool],
    queries: dict[str, np.ndarray | None],
    ref_features: dict[str, np.ndarray],
    *,
    tau: float,
    floor: float,
    spans: dict[str, float] | None = None,
) -> dict[str, IdentityResult]:
    """Decide identity for every slot with a pool. ``queries`` maps slot_label ->
    the mix span's stem-routed query (missing/None -> do-no-harm keep-claim);
    ``spans`` maps slot_label -> span seconds (drives conditional top-k)."""
    spans = spans or {}
    out: dict[str, IdentityResult] = {}
    for label, pool in pools.items():
        out[label] = decide_identity(
            pool,
            queries.get(label),
            ref_features,
            tau=tau,
            floor=floor,
            span_s=spans.get(label),
        )
    return out


def summarize(results: dict[str, IdentityResult]) -> dict[str, int]:
    """Decision-count rollup for the run log (accept/override/abstain)."""
    counts = {Decision.ACCEPT_CLAIM: 0, Decision.OVERRIDE: 0, Decision.ABSTAIN: 0}
    for r in results.values():
        counts[r.decision.decision] = counts.get(r.decision.decision, 0) + 1
    return counts
