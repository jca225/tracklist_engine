"""Provenance repository (§2/§4) — records runs, lineage edges, observations.

Wraps one execution as an immutable ``Run`` and links its input/output artifacts
+ derivation edges, so every derived artifact traces back to its sources (Law 1).
``closure`` walks the derivation graph to the roots and raises on a cycle
(Law 2). Observations are recorded here too — including abstained/malformed ones,
so a source that could not be parsed leaves a persisted diagnostic (Law 6/7).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from typing import Mapping, Optional, Sequence

from .decisions import decide, entropy
from .types import (
    Artifact,
    AxisBelief,
    Claim,
    Decision,
    DecisionRule,
    Evidence,
    EvidenceDirection,
    Json,
    Observation,
    ObservationStatus,
    Run,
    RunStatus,
    SubjectRef,
)
from .types import Axis as _Axis


class ProvenanceCycle(Exception):
    """A derivation edge would make the lineage graph cyclic (Law 2)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProvenanceRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # --- runs --------------------------------------------------------------
    def begin_run(
        self,
        *,
        process: str,
        process_version: str,
        code_commit: str,
        params_hash: str,
        random_seed: int | None = None,
    ) -> Run:
        run = Run(
            run_id=str(uuid.uuid4()),
            process=process,
            process_version=process_version,
            code_commit=code_commit,
            params_hash=params_hash,
            status=RunStatus.RUNNING,
            started_at=_now(),
            random_seed=random_seed,
        )
        self._conn.execute(
            "INSERT INTO run(run_id, process, process_version, code_commit, "
            "params_hash, status, started_at, random_seed) VALUES (?,?,?,?,?,?,?,?)",
            (
                run.run_id,
                process,
                process_version,
                code_commit,
                params_hash,
                run.status.value,
                run.started_at.isoformat(),
                random_seed,
            ),
        )
        self._conn.commit()
        return run

    def add_input(
        self, run: Run, artifact: Artifact, role: str, ordinal: int = 0
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run_input(run_id, content_sha256, role, ordinal) "
            "VALUES (?,?,?,?)",
            (run.run_id, artifact.content_sha256, role, ordinal),
        )
        self._conn.commit()

    def add_output(
        self, run: Run, artifact: Artifact, role: str, ordinal: int = 0
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run_output(run_id, content_sha256, role, ordinal) "
            "VALUES (?,?,?,?)",
            (run.run_id, artifact.content_sha256, role, ordinal),
        )
        self._conn.commit()

    def derive(
        self, child: Artifact, parent: Artifact, run: Run, relation: str
    ) -> None:
        """Record child←parent. Rejects an edge that would create a cycle (Law 2)."""
        if child.content_sha256 == parent.content_sha256:
            raise ProvenanceCycle("artifact cannot derive from itself")
        # child is an ancestor of parent -> adding parent→child closes a loop.
        if child.content_sha256 in self._ancestors(parent.content_sha256):
            raise ProvenanceCycle(
                f"{parent.content_sha256[:8]}→{child.content_sha256[:8]} is cyclic"
            )
        self._conn.execute(
            "INSERT OR IGNORE INTO derivation(child_sha256, parent_sha256, run_id, "
            "relation) VALUES (?,?,?,?)",
            (child.content_sha256, parent.content_sha256, run.run_id, relation),
        )
        self._conn.commit()

    def succeed(self, run: Run) -> None:
        self._finish(run, RunStatus.SUCCEEDED, None)

    def fail(self, run: Run, error: Json) -> None:
        self._finish(run, RunStatus.FAILED, error)

    def quarantine(self, run: Run, diagnostics: Json) -> None:
        self._finish(run, RunStatus.QUARANTINED, diagnostics)

    def _finish(self, run: Run, status: RunStatus, error: Json | None) -> None:
        self._conn.execute(
            "UPDATE run SET status=?, finished_at=?, error_json=? WHERE run_id=?",
            (
                status.value,
                _now().isoformat(),
                json.dumps(error) if error else None,
                run.run_id,
            ),
        )
        self._conn.commit()

    # --- observations ------------------------------------------------------
    def record_observation(
        self,
        *,
        subject: SubjectRef,
        predicate: str,
        value: object,
        source: Artifact,
        run: Run,
        status: ObservationStatus = ObservationStatus.OBSERVED,
        source_confidence: float | None = None,
        diagnostic_code: str | None = None,
    ) -> Observation:
        obs = Observation(
            observation_id=str(uuid.uuid4()),
            subject=subject,
            predicate=predicate,
            value=value,
            source_sha256=source.content_sha256,
            producer_run_id=run.run_id,
            observed_at=_now(),
            status=status,
            source_confidence=source_confidence,
            diagnostic_code=diagnostic_code,
        )
        self._conn.execute(
            "INSERT INTO observation(observation_id, subject_type, subject_id, "
            "predicate, value_json, source_sha256, producer_run_id, observed_at, "
            "status, source_confidence, diagnostic_code) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                obs.observation_id,
                subject.subject_type,
                subject.subject_id,
                predicate,
                json.dumps(value),
                source.content_sha256,
                run.run_id,
                obs.observed_at.isoformat(),
                status.value,
                source_confidence,
                diagnostic_code,
            ),
        )
        self._conn.commit()
        return obs

    # --- claims / evidence / beliefs (§5/§9) -------------------------------
    def record_claim(
        self,
        *,
        axis: _Axis,
        subject: SubjectRef,
        predicate: str,
        candidate: Optional[SubjectRef],
        run: Run,
        context: Json | None = None,
    ) -> Claim:
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            axis=axis,
            subject=subject,
            predicate=predicate,
            candidate=candidate,
            context=context or {},
            created_by_run_id=run.run_id,
        )
        self._conn.execute(
            "INSERT INTO claim(claim_id, axis, subject_type, subject_id, predicate, "
            "candidate_type, candidate_id, context_json, created_by_run_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                claim.claim_id,
                axis.value,
                subject.subject_type,
                subject.subject_id,
                predicate,
                candidate.subject_type if candidate else None,
                candidate.subject_id if candidate else None,
                json.dumps(claim.context),
                run.run_id,
            ),
        )
        self._conn.commit()
        return claim

    def record_evidence(
        self,
        *,
        claim: Claim,
        source_family: str,
        direction: EvidenceDirection,
        run: Run,
        native_score: Json | None = None,
        uncertainty: Json | None = None,
    ) -> Evidence:
        ev = Evidence(
            evidence_id=str(uuid.uuid4()),
            claim_id=claim.claim_id,
            axis=claim.axis,
            source_family=source_family,
            producer_run_id=run.run_id,
            direction=direction,
            native_score=native_score or {},
            uncertainty=uncertainty or {},
        )
        self._conn.execute(
            "INSERT INTO evidence(evidence_id, claim_id, axis, source_family, "
            "producer_run_id, direction, native_score_json, uncertainty_json) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                ev.evidence_id,
                claim.claim_id,
                claim.axis.value,
                source_family,
                run.run_id,
                direction.value,
                json.dumps(ev.native_score),
                json.dumps(ev.uncertainty),
            ),
        )
        self._conn.commit()
        return ev

    def record_belief(
        self,
        *,
        subject: SubjectRef,
        axis: _Axis,
        posterior: Mapping[str, float],
        rule: DecisionRule,
        run: Run,
        contributing_evidence_ids: Sequence[str] = (),
        supersedes_belief_id: Optional[str] = None,
    ) -> AxisBelief:
        """Decide the posterior via ``rule`` and persist the belief.

        The decision (accept / abstain) is computed here from the pure ``decide``
        so it is never hand-set inconsistently with the rule.
        """
        h = entropy(posterior)
        decision, chosen = decide(posterior, rule)
        belief = AxisBelief(
            belief_id=str(uuid.uuid4()),
            subject=subject,
            axis=axis,
            inference_run_id=run.run_id,
            posterior=dict(posterior),
            entropy=h,
            decision=decision,
            chosen=chosen,
            contributing_evidence_ids=tuple(contributing_evidence_ids),
            supersedes_belief_id=supersedes_belief_id,
        )
        self._conn.execute(
            "INSERT INTO belief(belief_id, subject_type, subject_id, axis, "
            "inference_run_id, posterior_json, entropy, decision, chosen, "
            "contributing_evidence_ids_json, supersedes_belief_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                belief.belief_id,
                subject.subject_type,
                subject.subject_id,
                axis.value,
                run.run_id,
                json.dumps(belief.posterior),
                h,
                decision.value,
                chosen,
                json.dumps(list(contributing_evidence_ids)),
                supersedes_belief_id,
            ),
        )
        self._conn.commit()
        return belief

    # --- graph queries -----------------------------------------------------
    def _ancestors(self, sha: str) -> set[str]:
        """All artifacts reachable from ``sha`` by following parent edges."""
        seen: set[str] = set()
        stack = [sha]
        while stack:
            cur = stack.pop()
            rows = self._conn.execute(
                "SELECT parent_sha256 FROM derivation WHERE child_sha256=?", (cur,)
            ).fetchall()
            for r in rows:
                p = r["parent_sha256"]
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def closure(self, sha: str) -> set[str]:
        """Transitive lineage roots + intermediates of an artifact (Law 1/21)."""
        return self._ancestors(sha)
