"""Reconcile crash residue in a provenance store (enable-blocker #1).

The writers commit in several steps, so a process killed mid-sequence (worker
restart, spot preemption) can leave two kinds of permanent Law-1 offenders in an
append-only store:

- a run stuck ``status='running'`` — a running run cannot survive the process
  that owned it, so any ``running`` run found later is dead; and
- an **orphan artifact** — ``put_bytes`` committed the row but the crash beat its
  ``add_output`` edge, so the artifact traces to no producing run.

Reconcile is the repair tool: run it at analysis-worker startup (or via
``python -m core.provenance.reconcile --db <root> [--prune]``). Marking stale runs
failed is always safe and on by default; pruning orphan artifacts is opt-in
(``--prune``) because it deletes rows — safe only *because* those artifacts were
never a completed output of any run.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ReconcileReport:
    stale_runs_failed: int
    orphan_artifacts: tuple[str, ...]
    pruned_artifacts: int = 0

    @property
    def clean(self) -> bool:
        return self.stale_runs_failed == 0 and not self.orphan_artifacts


def _orphan_shas(conn: sqlite3.Connection) -> list[str]:
    # An artifact that is neither a declared source nor any run's output, and is
    # referenced by nothing (so it is crash residue, not a live parent/source).
    rows = conn.execute(
        "SELECT a.content_sha256 FROM artifact a "
        "WHERE a.source_uri IS NULL AND a.source_system IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM run_output ro WHERE ro.content_sha256=a.content_sha256) "
        "AND NOT EXISTS (SELECT 1 FROM run_input ri WHERE ri.content_sha256=a.content_sha256) "
        "AND NOT EXISTS (SELECT 1 FROM derivation d WHERE d.child_sha256=a.content_sha256 OR d.parent_sha256=a.content_sha256) "
        "AND NOT EXISTS (SELECT 1 FROM observation o WHERE o.source_sha256=a.content_sha256)"
    ).fetchall()
    return [r["content_sha256"] for r in rows]


def reconcile_store(
    conn: sqlite3.Connection, *, prune_orphans: bool = False
) -> ReconcileReport:
    """Mark stale ``running`` runs failed; report (and optionally prune) orphan
    artifacts. Idempotent — a second run over a clean store changes nothing."""
    now = datetime.now(timezone.utc).isoformat()

    stale = [
        r["run_id"]
        for r in conn.execute(
            "SELECT run_id FROM run WHERE status='running'"
        ).fetchall()
    ]
    for rid in stale:
        conn.execute(
            "UPDATE run SET status='failed', finished_at=?, error_json=? WHERE run_id=?",
            (now, json.dumps({"reason": "reconciled_stale_run"}), rid),
        )

    orphans = _orphan_shas(conn)
    pruned = 0
    if prune_orphans:
        for sha in orphans:
            conn.execute("DELETE FROM artifact WHERE content_sha256=?", (sha,))
            pruned += 1
    conn.commit()
    return ReconcileReport(
        stale_runs_failed=len(stale),
        orphan_artifacts=tuple(orphans),
        pruned_artifacts=pruned,
    )


def main() -> int:
    import argparse
    from pathlib import Path

    from .store import connect

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="provenance store root")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="also DELETE orphan artifact rows (crash residue). Off by default.",
    )
    args = ap.parse_args()

    conn = connect(Path(args.db))
    report = reconcile_store(conn, prune_orphans=args.prune)
    print(f"reconcile — {args.db}")
    print(f"  stale 'running' runs → failed : {report.stale_runs_failed}")
    print(
        f"  orphan artifacts              : {len(report.orphan_artifacts)}"
        + (
            f" (pruned {report.pruned_artifacts})"
            if args.prune
            else " (report-only; --prune to delete)"
        )
    )
    print("  store is clean." if report.clean else "  residue found (above).")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
