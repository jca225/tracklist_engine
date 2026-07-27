"""Blind open-set identity — pure scoring core (Phase 1B).

Stem-routed MERT-chamfer identity over a real candidate pool, WITHOUT the
tracklist prior: rank which reference recording a mix span is, purely from audio.
This is the deployable, oracle-free core of the open-set findings
(``open_set_acappella_identity_findings.md``): MERT chamfer + conditional top-k +
borda fusion + a fail-closed margin gate against the tokenizer claim.

Design (mirrors ``evals/instance_separability.py``): everything here is PURE over
injected arrays so it is unit-testable with no audio/GPU/pi. The I/O glue (load
L3/L22 per-measure MERT for the stem-routed query + ref pool) and the wiring into
``infer.py``'s candidate pool live elsewhere and are added after RT1 (they touch
the live inference path). No new sensor: this consumes the existing MERT channel
at tuned layers (L3 vocal / L22 instrumental) — see the module CLAUDE.md sensor
freeze.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Conditional top-k: long spans accumulate other layered vocals, dragging the
# chamfer mean; average only the best query bars for spans past this length.
_LONG_SPAN_S = 60.0
_TOPK_FRAC = 0.25


def _l2norm(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def per_query_max_cos(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """For each query frame, its max cosine to any ref frame.

    ``query`` = (D, nq), ``ref`` = (D, nr). Returns (nq,). Empty query or ref -> [].
    Cosine via L2-normalized dot; the max over ref frames is the chamfer's inner
    ``max_ref``.
    """
    if query.size == 0 or ref.size == 0:
        return np.empty(0, dtype=np.float64)
    q = _l2norm(query.astype(np.float64), axis=0)
    r = _l2norm(ref.astype(np.float64), axis=0)
    sims = q.T @ r  # (nq, nr)
    return sims.max(axis=1)


def chamfer_score(
    query: np.ndarray,
    ref: np.ndarray,
    *,
    span_s: float | None = None,
    long_span_s: float = _LONG_SPAN_S,
    topk_frac: float = _TOPK_FRAC,
) -> float | None:
    """``mean_q max_ref cos`` with conditional top-k over long spans.

    For spans longer than ``long_span_s`` (when ``span_s`` is given), aggregate the
    per-query-frame max-cos with the mean of its top ``topk_frac`` fraction instead
    of the full mean — the long-clip contamination fix (73->82% on >60 s acappellas).
    Returns ``None`` when unscorable (empty query/ref), never a fake 0.0.
    """
    pq = per_query_max_cos(query, ref)
    if pq.size == 0:
        return None
    if span_s is not None and span_s > long_span_s:
        k = max(1, int(round(pq.size * topk_frac)))
        pq = np.sort(pq)[-k:]
    return float(pq.mean())


@dataclass(frozen=True)
class CandidateScore:
    recording_id: str
    score: float | None  # None = LF could not score this candidate (abstains on it)


def rank_by_chamfer(
    query: np.ndarray,
    ref_pool: dict[str, np.ndarray],
    *,
    span_s: float | None = None,
) -> list[CandidateScore]:
    """Score every candidate ref against the query; sorted best-first.

    ``ref_pool`` maps recording_id -> (D, nr) per-measure MERT for that ref (at the
    stem-routed layer). Unscorable candidates sort last (score None).
    """
    scored = [
        CandidateScore(rid, chamfer_score(query, ref, span_s=span_s))
        for rid, ref in ref_pool.items()
    ]
    scored.sort(key=lambda c: (c.score is not None, c.score or 0.0), reverse=True)
    return scored


def borda_fuse(rankings: list[list[CandidateScore]]) -> list[CandidateScore]:
    """Borda-count fusion over per-LF rankings (findings: borda(MERT,fp) ~ ceiling).

    Each LF contributes points = (n_scored - rank) to a candidate; unscored
    candidates in an LF get 0 from it. Only the two STRONG LFs should be passed —
    weak LFs (chroma/dtw on acappellas) confidently vote wrong and drag the fuse
    (findings: adding them drops 56-75%). Returns fused ranking, best-first, with
    the borda points in ``.score``.
    """
    points: dict[str, float] = {}
    for ranking in rankings:
        scored = [c for c in ranking if c.score is not None]
        n = len(scored)
        for rank, c in enumerate(scored):
            points[c.recording_id] = points.get(c.recording_id, 0.0) + (n - rank)
    fused = [CandidateScore(rid, pts) for rid, pts in points.items()]
    fused.sort(key=lambda c: c.score or 0.0, reverse=True)
    return fused


class Decision:
    ACCEPT_CLAIM = "accept_claim"  # perception agrees with / doesn't beat the claim
    OVERRIDE = "override"  # perception confidently disagrees -> replace the claim
    ABSTAIN = "abstain"  # nothing confident, claim not defensible -> NULL, persisted


@dataclass(frozen=True)
class IdentityDecision:
    decision: str
    recording_id: str | None  # chosen id (claim, override target, or None=abstain)
    claim_id: str | None
    top1_id: str | None
    margin: float | None  # top1 - claim score (fused units); None if unscorable
    top1_score: float | None


def margin_gate(
    ranked: list[CandidateScore],
    claim_id: str | None,
    *,
    tau: float,
    floor: float,
) -> IdentityDecision:
    """Fail-closed override: replace the claim ONLY when perception is both
    confident and disagrees.

    - override iff top1 != claim AND top1.score >= ``floor`` AND
      (top1.score - claim.score) >= ``tau``.
    - else accept the claim when it is present in the pool (even if not top1 — the
      "do no harm" default; blind MERT is only 68% on BB12).
    - else abstain (claim absent from pool and nothing clears the floor).

    ``tau`` / ``floor`` are tuned on ONE set and validated on the OTHER (LOSO),
    never fit on both (Law 19).
    """
    scored = [c for c in ranked if c.score is not None]
    if not scored:
        return IdentityDecision(
            Decision.ABSTAIN if claim_id is None else Decision.ACCEPT_CLAIM,
            claim_id,
            claim_id,
            None,
            None,
            None,
        )
    top1 = scored[0]
    claim_score = next((c.score for c in scored if c.recording_id == claim_id), None)

    # Override only on a *real* margin: either there is no claim to protect
    # (claimless slot -> confident fill) or the claim was itself scorable, so the
    # gap is meaningful. A non-None claim we simply could not score is NOT
    # evidence the claim is wrong (blind MERT is 68% on BB12) -> do no harm, fall
    # through to accept-claim rather than override on an infinite margin.
    claim_comparable = claim_id is None or claim_score is not None
    if (
        claim_comparable
        and top1.recording_id != claim_id
        and top1.score is not None
        and top1.score >= floor
    ):
        margin = top1.score - (claim_score if claim_score is not None else -np.inf)
        if margin >= tau:
            return IdentityDecision(
                Decision.OVERRIDE,
                top1.recording_id,
                claim_id,
                top1.recording_id,
                float(margin) if np.isfinite(margin) else None,
                top1.score,
            )

    if claim_id is not None:
        margin = (
            (top1.score - claim_score)
            if (claim_score is not None and top1.score is not None)
            else None
        )
        return IdentityDecision(
            Decision.ACCEPT_CLAIM,
            claim_id,
            claim_id,
            top1.recording_id,
            margin,
            top1.score,
        )

    # No claim to fall back on and top1 didn't clear the floor -> abstain.
    return IdentityDecision(
        Decision.ABSTAIN, None, None, top1.recording_id, None, top1.score
    )
