"""Phase 3.6: the deterministic driver — the end-to-end, no-LLM aligner.

Given a set of probes and a candidate pool, for each candidate it runs every probe,
fuses their evidence with merge(), and returns the most-confident non-abstaining
candidate (or abstains if none clears the bar). This is one of the interchangeable
drivers the contract enables: an LLM-agent driver and the trained model are the
others — same probes, same merge, same AlignmentResult.

Reference path resolution (recording_id -> audio file) is injected, so the driver
is pure orchestration and fully testable offline; wiring it to the canonical store
is the integration point, not driver logic.
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..records import SlotCandidate
from .contract import AlignmentResult, CandidatePool, MixContext, Probe, RefContext
from .merge import merge


class DeterministicDriver:
    """Probes x candidates -> per-candidate merge -> best decision."""

    def __init__(
        self,
        probes: Iterable[Probe],
        *,
        resolve_ref: Callable[[SlotCandidate], RefContext | None],
        offset_tol_s: float = 2.0,
        min_confidence: float = 0.0,
        agreement_bonus: float = 0.1,
    ) -> None:
        self.probes = tuple(probes)
        self._resolve_ref = resolve_ref
        self._offset_tol_s = offset_tol_s
        self._min_confidence = min_confidence
        self._agreement_bonus = agreement_bonus

    def align(self, mix: MixContext, candidates: CandidatePool) -> AlignmentResult:
        """Decide the best placement for the mix span over the candidate pool.

        Abstains (rather than forcing a pick) when no candidate's fused evidence
        clears the confidence bar — the open-set-friendly default.
        """
        decided: list[AlignmentResult] = []
        for cand in candidates:
            ref = self._resolve_ref(cand)
            if ref is None:
                continue
            results = tuple(p(mix, ref, candidates) for p in self.probes)
            merged = merge(
                results,
                offset_tol_s=self._offset_tol_s,
                min_confidence=self._min_confidence,
                agreement_bonus=self._agreement_bonus,
            )
            if not merged.abstain:
                decided.append(merged)
        if not decided:
            return AlignmentResult.abstained(source="driver")
        return max(decided, key=lambda r: r.confidence)
