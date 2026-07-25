"""Executable system laws (§16) — the checkable subset, as real predicates.

Brick 5: ``assert_system_laws`` made runnable. Each of the 21 laws in §16 of
docs/engine/dj_engine_pseudocode.md becomes a predicate over a
``core.provenance`` SQLite connection, returning a structured per-law verdict
(PASS / VIOLATED / NOT_APPLICABLE) with the offending rows named. This turns
the hand-authored table in docs/engine/law_audit.md into a gate you can run.

Scope honesty (do not overclaim):

- This measures a **substrate provenance DB** — what the Brick 1–4 writers
  persisted — NOT the shipped aligner pipeline. The hand-audited verdicts in
  docs/engine/law_audit.md grade the shipped pipeline and still stand; a green
  law here means "this substrate DB obeys the law", nothing more.
- Laws about entities the substrate does not hold yet (rounds/pseudo-labels
  §12, evaluation/promotion §11) report ``NOT_APPLICABLE`` with the reason.
  They are never faked green. Human labels (§7, laws 8/9) ARE held since
  Brick 6; model/process provenance (§8, laws 13/14) since Brick 7 —
  checkable once assertions / fitted models exist.
- A data-dependent law with zero applicable rows is ``NOT_APPLICABLE``, not
  PASS — an empty database proves nothing.
- Laws 4 and 5 are namespace/shape heuristics (documented on the functions),
  catching the known bug classes, not proving their absence.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional

from .store import ArtifactStore
from .types import ArtifactKind, Axis, Decision, ObservationStatus

_MAX_OFFENDERS = 5

# The 1001TL sided-row placeholder namespace ("Rvmor gap") — a source locator,
# never a canonical recording id (Law 4; see provenance_export._PLACEHOLDER_PREFIX).
_PLACEHOLDER_PREFIXES: tuple[str, ...] = ("tlp",)

# File-ish suffixes that mark an identity value as a path/filename (Law 5).
_PATHLIKE_SUFFIXES: tuple[str, ...] = (
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".als",
    ".json",
    ".npz",
    ".npy",
    ".pt",
    ".pkl",
)

_NON_OBSERVED = (
    ObservationStatus.EXPLICIT_NULL.value,
    ObservationStatus.ABSTAINED.value,
    ObservationStatus.MALFORMED.value,
)


class LawVerdict(StrEnum):
    PASS = "pass"
    VIOLATED = "violated"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class LawResult:
    """One law's verdict over one substrate DB."""

    number: int
    name: str
    verdict: LawVerdict
    reason: str
    checked: int = 0  # rows the predicate examined (0 -> NOT_APPLICABLE)
    offenders: tuple[str, ...] = ()  # up to _MAX_OFFENDERS sample violations


def _finish(
    number: int,
    name: str,
    checked: int,
    offenders: list[str],
    pass_reason: str,
    na_reason: str = "no applicable rows in this substrate DB",
) -> LawResult:
    if checked == 0:
        return LawResult(number, name, LawVerdict.NOT_APPLICABLE, na_reason)
    if offenders:
        return LawResult(
            number,
            name,
            LawVerdict.VIOLATED,
            f"{len(offenders)} violation(s) in {checked} row(s) checked",
            checked,
            tuple(offenders[:_MAX_OFFENDERS]),
        )
    return LawResult(number, name, LawVerdict.PASS, pass_reason, checked)


def _not_applicable(number: int, name: str, reason: str) -> LawResult:
    return LawResult(number, name, LawVerdict.NOT_APPLICABLE, reason)


def _count(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0])


# --- 1 · provenance_is_complete --------------------------------------------
def _law_01(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Every artifact is a source or a run output; every run finished; every
    lineage/observation edge references a run and artifacts that exist."""
    offenders: list[str] = []
    checked = 0

    checked += _count(conn, "SELECT COUNT(*) FROM artifact")
    for r in conn.execute(
        "SELECT content_sha256 FROM artifact a "
        "WHERE a.source_uri IS NULL AND a.source_system IS NULL AND NOT EXISTS "
        "(SELECT 1 FROM run_output ro WHERE ro.content_sha256=a.content_sha256)"
    ).fetchall():
        offenders.append(
            f"artifact {r['content_sha256'][:12]} has no source and no producing run"
        )

    checked += _count(conn, "SELECT COUNT(*) FROM run")
    for r in conn.execute(
        "SELECT run_id, status FROM run WHERE status IN ('running','pending')"
    ).fetchall():
        offenders.append(f"run {r['run_id'][:8]} never finished (status={r['status']})")

    checked += _count(conn, "SELECT COUNT(*) FROM observation")
    for r in conn.execute(
        "SELECT observation_id FROM observation o WHERE NOT EXISTS "
        "(SELECT 1 FROM run r WHERE r.run_id=o.producer_run_id)"
    ).fetchall():
        offenders.append(f"observation {r['observation_id'][:8]} cites a missing run")

    checked += _count(conn, "SELECT COUNT(*) FROM derivation")
    for r in conn.execute(
        "SELECT child_sha256 FROM derivation d WHERE "
        "NOT EXISTS (SELECT 1 FROM run r WHERE r.run_id=d.run_id) OR "
        "NOT EXISTS (SELECT 1 FROM artifact a WHERE a.content_sha256=d.child_sha256) OR "
        "NOT EXISTS (SELECT 1 FROM artifact a WHERE a.content_sha256=d.parent_sha256)"
    ).fetchall():
        offenders.append(
            f"derivation of {r['child_sha256'][:12]} references a missing run/artifact"
        )

    return _finish(
        1,
        "provenance_is_complete",
        checked,
        offenders,
        "every artifact traces to a source or producing run; all runs finished",
        na_reason="empty substrate (no artifacts/runs)",
    )


# --- 2 · provenance_is_acyclic ----------------------------------------------
def _law_02(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Depth-first search over the derivation graph; any back-edge is a cycle."""
    parents: dict[str, list[str]] = {}
    n_edges = 0
    for r in conn.execute("SELECT child_sha256, parent_sha256 FROM derivation"):
        parents.setdefault(r["child_sha256"], []).append(r["parent_sha256"])
        n_edges += 1

    offenders: list[str] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {}
    for start in list(parents):
        if color.get(start, WHITE) != WHITE:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, done = stack.pop()
            if done:
                color[node] = BLACK
                continue
            if color.get(node, WHITE) == BLACK:
                continue
            color[node] = GRAY
            stack.append((node, True))
            for p in parents.get(node, []):
                c = color.get(p, WHITE)
                if c == GRAY:
                    offenders.append(f"cycle through {p[:12]} <-> {node[:12]}")
                elif c == WHITE:
                    stack.append((p, False))

    return _finish(
        2,
        "provenance_is_acyclic",
        n_edges,
        offenders,
        f"derivation graph acyclic ({n_edges} edge(s))",
        na_reason="no derivation edges yet",
    )


# --- 3 · all_artifacts_verify ------------------------------------------------
def _law_03(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Re-hash every stored object; bytes must match their content_sha256."""
    if store is None:
        return _not_applicable(
            3, "all_artifacts_verify", "no object store supplied (conn-only check)"
        )
    offenders: list[str] = []
    checked = 0
    for r in conn.execute("SELECT content_sha256 FROM artifact").fetchall():
        checked += 1
        sha = r["content_sha256"]
        if not store.verify(sha):
            offenders.append(f"artifact {sha[:12]} missing or hash mismatch")
    return _finish(
        3,
        "all_artifacts_verify",
        checked,
        offenders,
        "all stored objects re-hash to their content_sha256",
        na_reason="no artifacts stored",
    )


# --- 4 · no_source_key_is_canonical_recording_id -----------------------------
def _law_04(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """No claim candidate / belief choice is a source-key placeholder (tlp*).

    Heuristic: checks the known 1001TL placeholder namespace — the exact leak
    class of the shipped ``COALESCE(recording_id, track_id)`` (law_audit §4).
    """
    offenders: list[str] = []
    checked = 0
    checked += _count(conn, "SELECT COUNT(*) FROM claim WHERE candidate_id IS NOT NULL")
    checked += _count(conn, "SELECT COUNT(*) FROM belief WHERE chosen IS NOT NULL")
    for prefix in _PLACEHOLDER_PREFIXES:
        for r in conn.execute(
            "SELECT claim_id, candidate_id FROM claim WHERE candidate_id LIKE ?",
            (prefix + "%",),
        ).fetchall():
            offenders.append(
                f"claim {r['claim_id'][:8]} candidate is placeholder "
                f"{r['candidate_id']!r}"
            )
        for r in conn.execute(
            "SELECT belief_id, chosen FROM belief WHERE chosen LIKE ?",
            (prefix + "%",),
        ).fetchall():
            offenders.append(
                f"belief {r['belief_id'][:8]} chose placeholder {r['chosen']!r}"
            )
    return _finish(
        4,
        "no_source_key_is_canonical_recording_id",
        checked,
        offenders,
        "no claim/belief carries a tlp* source-key as identity",
        na_reason="no identity candidates/choices recorded",
    )


def _is_pathlike(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    return value.lower().endswith(_PATHLIKE_SUFFIXES)


# --- 5 · no_path_was_used_as_identity ----------------------------------------
def _law_05(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """No claim candidate / belief choice looks like a filesystem path.

    Heuristic: path separators or file-ish suffixes. Catches the
    identity-by-filename bug class, does not prove its absence.
    """
    offenders: list[str] = []
    checked = 0
    for table, id_col, val_col in (
        ("claim", "claim_id", "candidate_id"),
        ("belief", "belief_id", "chosen"),
    ):
        for r in conn.execute(
            f"SELECT {id_col} AS id, {val_col} AS val FROM {table} "  # noqa: S608
            f"WHERE {val_col} IS NOT NULL"
        ).fetchall():
            checked += 1
            if _is_pathlike(str(r["val"])):
                offenders.append(
                    f"{table} {r['id'][:8]} uses path-like identity {r['val']!r}"
                )
    return _finish(
        5,
        "no_path_was_used_as_identity",
        checked,
        offenders,
        "no claim/belief uses a path-like value as identity",
        na_reason="no identity candidates/choices recorded",
    )


# --- 6 · every_unknown_parser_row_has_diagnostic -----------------------------
def _law_06(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Every abstained/malformed observation carries a diagnostic_code."""
    offenders: list[str] = []
    checked = 0
    for r in conn.execute(
        "SELECT observation_id, predicate, status, diagnostic_code FROM observation "
        "WHERE status IN (?,?)",
        (ObservationStatus.ABSTAINED.value, ObservationStatus.MALFORMED.value),
    ).fetchall():
        checked += 1
        if r["diagnostic_code"] is None:
            offenders.append(
                f"observation {r['observation_id'][:8]} <{r['predicate']}> "
                f"{r['status']} with no diagnostic_code"
            )
    return _finish(
        6,
        "every_unknown_parser_row_has_diagnostic",
        checked,
        offenders,
        "every abstained/malformed observation names its diagnostic",
        na_reason="no abstained/malformed observations recorded",
    )


# --- 7 · all_abstentions_are_persisted ---------------------------------------
def _law_07(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """The DB-checkable half: persisted abstentions are honest — an UNRESOLVED
    belief never smuggles a chosen value, an ACCEPTED one always names it.
    (Whether every upstream abstention *reached* the DB cannot be proven from
    the DB alone — that half stays with the writers.)"""
    offenders: list[str] = []
    checked = _count(conn, "SELECT COUNT(*) FROM belief")
    for r in conn.execute(
        "SELECT belief_id, decision, chosen FROM belief WHERE "
        "(decision=? AND chosen IS NOT NULL) OR (decision=? AND chosen IS NULL)",
        (Decision.UNRESOLVED.value, Decision.ACCEPTED.value),
    ).fetchall():
        offenders.append(
            f"belief {r['belief_id'][:8]} decision={r['decision']} "
            f"chosen={r['chosen']!r}"
        )
    return _finish(
        7,
        "all_abstentions_are_persisted",
        checked,
        offenders,
        "abstaining beliefs persist with no smuggled choice",
        na_reason="no beliefs recorded",
    )


# --- 8 · human_history_is_append_only ----------------------------------------
def _law_08(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Human-label history is append-only: every ``supersedes_assertion_id``
    resolves to a row that STILL EXISTS (a dangling pointer means the old row
    was deleted/overwritten — the exact §7 violation), supersession forms no
    cycle, and every assertion's bundle + producing run exist. The append-only
    property is structural — it can only be broken by mutating rows, which
    these referential checks surface."""
    checked = _count(conn, "SELECT COUNT(*) FROM human_label_assertion")
    offenders: list[str] = []

    for r in conn.execute(
        "SELECT assertion_id, supersedes_assertion_id FROM human_label_assertion a "
        "WHERE a.supersedes_assertion_id IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM human_label_assertion o "
        " WHERE o.assertion_id=a.supersedes_assertion_id)"
    ).fetchall():
        offenders.append(
            f"assertion {r['assertion_id'][:8]} supersedes "
            f"{str(r['supersedes_assertion_id'])[:8]} which no longer exists "
            "(history was rewritten)"
        )

    # Supersession must be acyclic (an assertion can never — transitively —
    # supersede itself).
    supersedes: dict[str, str] = {
        r["assertion_id"]: r["supersedes_assertion_id"]
        for r in conn.execute(
            "SELECT assertion_id, supersedes_assertion_id FROM human_label_assertion "
            "WHERE supersedes_assertion_id IS NOT NULL"
        )
    }
    resolved: set[str] = set()
    for start in supersedes:
        if start in resolved:
            continue
        path: list[str] = []
        on_path: set[str] = set()
        node: Optional[str] = start
        while node is not None and node in supersedes and node not in resolved:
            if node in on_path:
                offenders.append(f"supersession cycle through {node[:8]}")
                break
            on_path.add(node)
            path.append(node)
            node = supersedes[node]
        resolved.update(path)

    for r in conn.execute(
        "SELECT assertion_id FROM human_label_assertion a WHERE NOT EXISTS "
        "(SELECT 1 FROM human_label_bundle b WHERE b.bundle_id=a.bundle_id)"
    ).fetchall():
        offenders.append(f"assertion {r['assertion_id'][:8]} cites a missing bundle")
    for r in conn.execute(
        "SELECT assertion_id FROM human_label_assertion a WHERE NOT EXISTS "
        "(SELECT 1 FROM run r WHERE r.run_id=a.produced_run_id)"
    ).fetchall():
        offenders.append(f"assertion {r['assertion_id'][:8]} cites a missing run")

    return _finish(
        8,
        "human_history_is_append_only",
        checked,
        offenders,
        "every supersession resolves to a live row; no cycle; bundles/runs intact",
        na_reason="no human-label assertions recorded (§7 adapter not run)",
    )


# --- 9 · human_uncertainty_is_field_specific ---------------------------------
def _law_09(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Every human assertion carries a non-empty uncertainty model (a dict with
    a ``kind``), and no bundle applies ONE identical model uniformly across
    multiple distinct fields — that is the global-scalar-confidence smell the
    law forbids."""
    offenders: list[str] = []
    checked = 0
    # (bundle_id -> field set / distinct-model set) for the uniformity check.
    bundle_fields: dict[str, set[str]] = {}
    bundle_models: dict[str, set[str]] = {}
    for r in conn.execute(
        "SELECT assertion_id, bundle_id, field, uncertainty_model_json "
        "FROM human_label_assertion"
    ).fetchall():
        checked += 1
        raw = r["uncertainty_model_json"]
        model: object = None
        try:
            model = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            model = None
        if not isinstance(model, dict) or not str(model.get("kind", "")):
            offenders.append(
                f"assertion {r['assertion_id'][:8]} <{r['field']}> has no "
                "usable uncertainty_model (missing/empty/kind-less)"
            )
            continue
        bundle_fields.setdefault(r["bundle_id"], set()).add(r["field"])
        bundle_models.setdefault(r["bundle_id"], set()).add(
            json.dumps(model, sort_keys=True)
        )
    for bundle_id, fields in bundle_fields.items():
        if len(fields) >= 2 and len(bundle_models.get(bundle_id, set())) == 1:
            offenders.append(
                f"bundle {bundle_id[:8]} applies ONE uncertainty model uniformly "
                f"across {len(fields)} distinct fields (global scalar smell)"
            )
    return _finish(
        9,
        "human_uncertainty_is_field_specific",
        checked,
        offenders,
        "every assertion carries its own field-specific uncertainty model",
        na_reason="no human-label assertions recorded (§7 adapter not run)",
    )


# --- 10 · identity_placement_structure_are_separate --------------------------
def _law_10(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Axis values are valid and never cross: every evidence row carries the
    same axis as its claim (no cross-axis fusion at the evidence level)."""
    valid = {a.value for a in Axis}
    offenders: list[str] = []
    checked = 0
    for table, id_col in (
        ("claim", "claim_id"),
        ("evidence", "evidence_id"),
        ("belief", "belief_id"),
    ):
        for r in conn.execute(
            f"SELECT {id_col} AS id, axis FROM {table}"  # noqa: S608
        ).fetchall():
            checked += 1
            if r["axis"] not in valid:
                offenders.append(
                    f"{table} {r['id'][:8]} has invalid axis {r['axis']!r}"
                )
    for r in conn.execute(
        "SELECT e.evidence_id, e.axis AS e_axis, c.axis AS c_axis "
        "FROM evidence e JOIN claim c ON c.claim_id=e.claim_id WHERE e.axis != c.axis"
    ).fetchall():
        offenders.append(
            f"evidence {r['evidence_id'][:8]} axis={r['e_axis']} fused onto "
            f"claim axis={r['c_axis']}"
        )
    return _finish(
        10,
        "identity_placement_structure_are_separate",
        checked,
        offenders,
        "all axis values valid; no evidence crosses its claim's axis",
        na_reason="no claims/evidence/beliefs recorded",
    )


def _numeric_coord_paths(value: object, prefix: str = "") -> list[str]:
    """Coordinate-like keys (``*_s``) holding a number, anywhere in a payload."""
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k)
            path = f"{prefix}.{key}" if prefix else key
            if (
                key.endswith("_s")
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            ):
                found.append(path)
            if isinstance(v, (dict, list)):
                found.extend(_numeric_coord_paths(v, path))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(_numeric_coord_paths(v, f"{prefix}[{i}]"))
    return found


# --- 11 · no_unknown_coordinate_equals_zero ----------------------------------
def _law_11(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """The substrate reading: an unknown is JSON null, never a number. An
    abstained/explicit-null/malformed observation must not embed a numeric
    coordinate (``*_s`` key) — 0.0 included."""
    offenders: list[str] = []
    checked = 0
    for r in conn.execute(
        "SELECT observation_id, predicate, status, value_json FROM observation "
        "WHERE status IN (?,?,?)",
        _NON_OBSERVED,
    ).fetchall():
        checked += 1
        value = json.loads(r["value_json"]) if r["value_json"] else None
        for path in _numeric_coord_paths(value):
            offenders.append(
                f"observation {r['observation_id'][:8]} <{r['predicate']}> "
                f"{r['status']} carries numeric coordinate {path}"
            )
    return _finish(
        11,
        "no_unknown_coordinate_equals_zero",
        checked,
        offenders,
        "no abstained/null observation smuggles a numeric coordinate",
        na_reason="no abstained/null observations recorded",
    )


# --- 12 · decoder_consumes_posteriors_not_raw_margins ------------------------
def _law_12(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    n_flagged = _count(
        conn,
        "SELECT COUNT(*) FROM observation "
        "WHERE diagnostic_code='uncalibrated_priority_decode'",
    )
    return _not_applicable(
        12,
        "decoder_consumes_posteriors_not_raw_margins",
        "decoder not represented in the substrate; its debt is visible as "
        f"{n_flagged} 'uncalibrated_priority_decode' observation(s) (P2 closes)",
    )


# --- 13 · every_model_has_training_snapshot ----------------------------------
def _law_13(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Every fitted model references a training snapshot that EXISTS. A null or
    dangling ``training_snapshot_id`` is a model whose training set cannot be
    reproduced — the exact §8 violation."""
    offenders: list[str] = []
    checked = 0
    for r in conn.execute(
        "SELECT fitted_model_id, training_snapshot_id FROM fitted_model"
    ).fetchall():
        checked += 1
        snap_id = r["training_snapshot_id"]
        if not snap_id:
            offenders.append(
                f"fitted_model {r['fitted_model_id'][:12]} has no training snapshot"
            )
            continue
        if (
            conn.execute(
                "SELECT 1 FROM training_snapshot WHERE training_snapshot_id=?",
                (snap_id,),
            ).fetchone()
            is None
        ):
            offenders.append(
                f"fitted_model {r['fitted_model_id'][:12]} cites missing "
                f"snapshot {str(snap_id)[:12]}"
            )
    return _finish(
        13,
        "every_model_has_training_snapshot",
        checked,
        offenders,
        "every fitted model references a live frozen training snapshot",
        na_reason="no fitted_model rows recorded (§8 writers not run)",
    )


_BUILTIN_KINDS: tuple[str, ...] = tuple(k.value for k in ArtifactKind)


# --- 14 · every_model_has_code_and_environment -------------------------------
def _law_14(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Two halves. (a) Every fitted model's process spec exists, carries a
    non-empty code commit, and references a parameter set + environment spec
    that exist. (b) Every artifact of a REGISTERED feature kind (a kind outside
    the built-in ArtifactKind enum, i.e. created via ``@artifact_type``) is the
    output of a run whose process has a ProcessSpec — a registered feature with
    no versioned producer is the shipped-pipeline bug this law exists for."""
    offenders: list[str] = []
    checked = 0

    def _spec_offense(spec_id: str, what: str) -> str:
        return f"process_spec {spec_id[:12]} {what}"

    for r in conn.execute(
        "SELECT fitted_model_id, process_spec_id FROM fitted_model"
    ).fetchall():
        checked += 1
        spec = conn.execute(
            "SELECT process_spec_id, code_commit, parameter_set_id, "
            "environment_spec_id FROM process_spec WHERE process_spec_id=?",
            (r["process_spec_id"],),
        ).fetchone()
        if spec is None:
            offenders.append(
                f"fitted_model {r['fitted_model_id'][:12]} cites missing "
                f"process_spec {str(r['process_spec_id'])[:12]}"
            )
            continue
        if not str(spec["code_commit"] or "").strip():
            offenders.append(
                _spec_offense(spec["process_spec_id"], "has no code_commit")
            )
        if (
            conn.execute(
                "SELECT 1 FROM parameter_set WHERE parameter_set_id=?",
                (spec["parameter_set_id"],),
            ).fetchone()
            is None
        ):
            offenders.append(
                _spec_offense(spec["process_spec_id"], "cites missing parameter_set")
            )
        if (
            conn.execute(
                "SELECT 1 FROM environment_spec WHERE environment_spec_id=?",
                (spec["environment_spec_id"],),
            ).fetchone()
            is None
        ):
            offenders.append(
                _spec_offense(spec["process_spec_id"], "cites missing environment_spec")
            )

    placeholders = ",".join("?" for _ in _BUILTIN_KINDS)
    for r in conn.execute(
        f"SELECT content_sha256, kind FROM artifact "  # noqa: S608
        f"WHERE kind NOT IN ({placeholders})",
        _BUILTIN_KINDS,
    ).fetchall():
        checked += 1
        produced = conn.execute(
            "SELECT 1 FROM run_output ro JOIN run r ON r.run_id=ro.run_id "
            "WHERE ro.content_sha256=? AND r.process_spec_id IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM process_spec ps "
            "            WHERE ps.process_spec_id=r.process_spec_id)",
            (r["content_sha256"],),
        ).fetchone()
        if produced is None:
            offenders.append(
                f"feature artifact {r['content_sha256'][:12]} "
                f"[{r['kind']}] has no ProcessSpec-versioned producing run"
            )
    return _finish(
        14,
        "every_model_has_code_and_environment",
        checked,
        offenders,
        "models + registered feature artifacts all trace to versioned "
        "code/params/environment",
        na_reason="no fitted models or registered-kind feature artifacts "
        "recorded (§8 writers not run)",
    )


# --- 21 · every_published_value_is_explainable -------------------------------
def _law_21(conn: sqlite3.Connection, store: Optional[ArtifactStore]) -> LawResult:
    """Every belief resolves its run + cited evidence (and each evidence its
    claim); every observation resolves its source artifact — i.e. explain()
    can answer for every persisted value."""
    offenders: list[str] = []
    checked = 0
    for r in conn.execute(
        "SELECT belief_id, inference_run_id, contributing_evidence_ids_json FROM belief"
    ).fetchall():
        checked += 1
        run = conn.execute(
            "SELECT 1 FROM run WHERE run_id=?", (r["inference_run_id"],)
        ).fetchone()
        if run is None:
            offenders.append(f"belief {r['belief_id'][:8]} cites a missing run")
        for ev_id in json.loads(r["contributing_evidence_ids_json"] or "[]"):
            ev = conn.execute(
                "SELECT claim_id FROM evidence WHERE evidence_id=?", (ev_id,)
            ).fetchone()
            if ev is None:
                offenders.append(
                    f"belief {r['belief_id'][:8]} cites missing evidence "
                    f"{str(ev_id)[:8]}"
                )
            elif (
                conn.execute(
                    "SELECT 1 FROM claim WHERE claim_id=?", (ev["claim_id"],)
                ).fetchone()
                is None
            ):
                offenders.append(f"evidence {str(ev_id)[:8]} cites missing claim")
    for r in conn.execute(
        "SELECT observation_id FROM observation o WHERE NOT EXISTS "
        "(SELECT 1 FROM artifact a WHERE a.content_sha256=o.source_sha256)"
    ).fetchall():
        offenders.append(
            f"observation {r['observation_id'][:8]} cites a missing source artifact"
        )
        checked += 1
    checked += _count(conn, "SELECT COUNT(*) FROM observation")
    return _finish(
        21,
        "every_published_value_is_explainable",
        checked,
        offenders,
        "every belief/observation resolves its full evidence/source chain",
        na_reason="no beliefs/observations recorded",
    )


# --- laws over entities the substrate does not hold yet ----------------------
_UNBUILT: dict[int, tuple[str, str]] = {
    15: (
        "no_round_consumes_its_own_outputs",
        "no co-training round entities in the substrate (§12 unbuilt, P3)",
    ),
    16: (
        "pseudo_label_graph_is_acyclic",
        "no pseudo-label entities in the substrate (§12 unbuilt, P3)",
    ),
    17: (
        "rejected_rounds_are_reproducible",
        "no round entities in the substrate (§12 unbuilt, P3)",
    ),
    18: (
        "evaluation_is_per_set_and_per_axis",
        "evaluation results not represented in the substrate (scorer-side; "
        "shipped-pipeline audit: PASS)",
    ),
    19: (
        "not_claims_corpus_calibration_from_two_sets",
        "decision rules are not persisted in the substrate DB; the claim "
        "discipline lives in docs (shipped-pipeline audit: PASS)",
    ),
    20: (
        "reconstruction_does_not_certify_identity",
        "reconstruction features not represented in the substrate "
        "(shipped-pipeline audit: PASS)",
    ),
}

_CHECKERS: dict[
    int, Callable[[sqlite3.Connection, Optional[ArtifactStore]], LawResult]
] = {
    1: _law_01,
    2: _law_02,
    3: _law_03,
    4: _law_04,
    5: _law_05,
    6: _law_06,
    7: _law_07,
    8: _law_08,
    9: _law_09,
    10: _law_10,
    11: _law_11,
    12: _law_12,
    13: _law_13,
    14: _law_14,
    21: _law_21,
}


def check_laws(
    conn: sqlite3.Connection, store: Optional[ArtifactStore] = None
) -> list[LawResult]:
    """Run every §16 law against a substrate DB; results in law order 1–21.

    ``conn`` must come from :func:`core.provenance.connect` (row factory set).
    ``store`` enables law 3 (byte re-verification); without it law 3 is
    NOT_APPLICABLE.
    """
    results: list[LawResult] = []
    for number in range(1, 22):
        checker = _CHECKERS.get(number)
        if checker is not None:
            results.append(checker(conn, store))
        else:
            name, reason = _UNBUILT[number]
            results.append(_not_applicable(number, name, reason))
    return results


# --- text rendering ----------------------------------------------------------
_VERDICT_LABEL = {
    LawVerdict.PASS: "PASS",
    LawVerdict.VIOLATED: "VIOLATED",
    LawVerdict.NOT_APPLICABLE: "N/A",
}


def format_law_results(results: list[LawResult], *, show_offenders: bool = True) -> str:
    lines = [
        "§16 executable law check — substrate DB scope only",
        "(shipped-pipeline verdicts live in docs/engine/law_audit.md and are",
        " NOT graded here)",
        "",
    ]
    for r in results:
        label = _VERDICT_LABEL[r.verdict]
        n = f" [{r.checked} checked]" if r.checked else ""
        lines.append(f"{r.number:>2} {label:<8} {r.name:<45} {r.reason}{n}")
        if show_offenders:
            for off in r.offenders:
                lines.append(f"       ! {off}")
    tally = {v: sum(1 for r in results if r.verdict is v) for v in LawVerdict}
    lines.append("")
    lines.append(
        f"tally: {tally[LawVerdict.PASS]} PASS / "
        f"{tally[LawVerdict.VIOLATED]} VIOLATED / "
        f"{tally[LawVerdict.NOT_APPLICABLE]} N/A"
    )
    return "\n".join(lines)


def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    from .store import connect

    ap = argparse.ArgumentParser(
        description="Run the §16 system laws against a provenance substrate DB."
    )
    ap.add_argument("--db", required=True, help="provenance DB root dir")
    ap.add_argument(
        "--no-offenders", action="store_true", help="suppress sample offending rows"
    )
    args = ap.parse_args()

    root = Path(args.db)
    if not (root / "provenance.db").exists():
        print(f"no provenance.db under {root}", file=sys.stderr)
        return 2
    conn = connect(root)
    store = ArtifactStore(root, conn)
    results = check_laws(conn, store)
    print(format_law_results(results, show_offenders=not args.no_offenders))
    return 1 if any(r.verdict is LawVerdict.VIOLATED for r in results) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
