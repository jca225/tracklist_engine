"""Behavioural tests for scripts/pi_autopull.sh (the cluster auto-sync timer).

The script keeps a pi's checkout in sync with origin/main. These tests build a
throwaway bare origin + a working clone (the stand-in "pi") and drive the real
script against it via subprocess, so no live cluster is touched.

Key behaviour under test: when the incoming range touches a schema/migration
file, the script must HOLD (not fast-forward) and print MIGRATION PENDING, so a
human runs the coordinated migrate+deploy instead of the code silently landing
on the canonical box ahead of its schema (see issue #73 / Crush Phase B GATE).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "pi_autopull.sh"

_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}


def _git(args, cwd):
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **_ENV},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"git {args} failed: {r.stderr}"
    return r.stdout.strip()


def _head(cwd) -> str:
    return _git(["rev-parse", "HEAD"], cwd)


def _push_change(seed: Path, rel: str, content: str, msg: str) -> None:
    p = seed / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _git(["add", "-A"], seed)
    _git(["commit", "-m", msg], seed)
    _git(["push", "-q", "origin", "main"], seed)


def _run_autopull(work: Path):
    return subprocess.run(
        ["bash", str(work / "scripts" / "pi_autopull.sh")],
        cwd=str(work),
        env={
            **os.environ,
            **_ENV,
            "TRACKLIST_REPO": str(work),
            "TRACKLIST_BRANCH": "main",
        },
        capture_output=True,
        text=True,
    )


@pytest.fixture
def pi_repo(tmp_path):
    """Bare origin + working clone seeded with the real script + placeholder tree."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True
    )
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True)

    (seed / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, seed / "scripts" / "pi_autopull.sh")
    os.chmod(seed / "scripts" / "pi_autopull.sh", 0o755)
    (seed / "requirements.txt").write_text("# reqs\n")
    (seed / "app.py").write_text("x = 1\n")
    (seed / "web_crawler" / "database").mkdir(parents=True)
    (seed / "web_crawler" / "database" / "schema.sql").write_text("-- schema v1\n")
    (seed / "scripts" / "migrations").mkdir()
    (seed / "scripts" / "migrations" / "migrate_base.sql").write_text("-- base\n")
    _git(["add", "-A"], seed)
    _git(["commit", "-m", "init"], seed)
    _git(["push", "-q", "origin", "main"], seed)

    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    return work, seed


def test_migration_change_holds_and_does_not_pull(pi_repo):
    work, seed = pi_repo
    before = _head(work)
    _push_change(
        seed,
        "scripts/migrations/migrate_content_history.sql",
        "-- new\n",
        "feat: migration",
    )

    r = _run_autopull(work)

    assert "MIGRATION PENDING" in r.stdout, r.stdout + r.stderr
    assert _head(work) == before, "must NOT fast-forward when a migration is pending"


def test_schema_change_holds_and_does_not_pull(pi_repo):
    work, seed = pi_repo
    before = _head(work)
    _push_change(
        seed, "web_crawler/database/schema.sql", "-- schema v2\n", "feat: schema bump"
    )

    r = _run_autopull(work)

    assert "MIGRATION PENDING" in r.stdout, r.stdout + r.stderr
    assert _head(work) == before, "schema.sql change must also hold the sync"


def test_code_only_change_still_syncs(pi_repo):
    """Guard must be scoped: a plain code change (no migration) fast-forwards."""
    work, seed = pi_repo
    _push_change(seed, "app.py", "x = 2\n", "feat: code change")
    remote = _git(["rev-parse", "origin/main"], seed)

    r = _run_autopull(work)

    assert "MIGRATION PENDING" not in r.stdout, r.stdout
    assert _head(work) == remote, "a non-migration change must sync normally"


def test_dirty_tracked_tree_skips_without_pulling(pi_repo):
    """No-clobber: uncommitted tracked change refuses the pull entirely."""
    work, seed = pi_repo
    before = _head(work)
    _push_change(seed, "app.py", "x = 3\n", "feat: upstream change")
    (work / "app.py").write_text("local wip\n")  # dirty the working tree

    r = _run_autopull(work)

    assert "SKIP" in r.stdout, r.stdout
    assert _head(work) == before, "must not pull over uncommitted tracked WIP"
