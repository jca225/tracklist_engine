"""Tests for scripts/entropy_audit.py — the AST-based bug-class detector that
fences the classes from the 2026-07-16 audit so they can't regress.

The fences target the *precise* bug class: subprocess calls that talk to the
network / DB (ssh / rsync / sqlite3) missing a timeout (B1 hang) or, when
text-mode, missing an explicit encoding (A2 mojibake). Benign local subprocess
calls must NOT count.
"""

from __future__ import annotations

from scripts.entropy_audit import _iter_py_files, scan_source


def _n(src: str, key: str) -> int:
    return scan_source(src)[key]


# --- net subprocess without timeout (B1) --------------------------------------
def test_ssh_call_without_timeout_counts():
    src = 'import subprocess\nsubprocess.run(["ssh", host, cmd], check=True)\n'
    assert _n(src, "net_subprocess_no_timeout") == 1


def test_ssh_call_with_timeout_does_not_count():
    src = 'import subprocess\nsubprocess.run(["ssh", host, cmd], timeout=30)\n'
    assert _n(src, "net_subprocess_no_timeout") == 0


def test_rsync_multiline_call_with_timeout_ok():
    src = (
        "import subprocess\n"
        "subprocess.check_call(\n"
        '    ["rsync", "-a", src, dst],\n'
        "    timeout=RSYNC_TIMEOUT,\n"
        ")\n"
    )
    assert _n(src, "net_subprocess_no_timeout") == 0


def test_local_subprocess_without_timeout_is_ignored():
    # not a network/db call -> not our fence
    src = 'import subprocess\nsubprocess.run(["ls", "-l"])\n'
    assert _n(src, "net_subprocess_no_timeout") == 0


def test_sqlite3_call_without_timeout_counts():
    src = 'import subprocess\nsubprocess.run(["sqlite3", db], input=sql)\n'
    assert _n(src, "net_subprocess_no_timeout") == 1


# --- text-mode net subprocess without encoding (A2) ---------------------------
def test_text_ssh_without_encoding_counts():
    src = 'import subprocess\nsubprocess.run(["ssh", h], text=True, timeout=5)\n'
    assert _n(src, "net_subprocess_no_encoding") == 1


def test_text_ssh_with_encoding_ok():
    src = (
        "import subprocess\n"
        'subprocess.run(["ssh", h], text=True, encoding="utf-8", timeout=5)\n'
    )
    assert _n(src, "net_subprocess_no_encoding") == 0


def test_text_false_is_not_flagged():
    src = 'import subprocess\nsubprocess.run(["ssh", h], text=False, timeout=5)\n'
    assert _n(src, "net_subprocess_no_encoding") == 0


# --- bare except (silent-failure class) ---------------------------------------
def test_bare_except_counts():
    src = "try:\n    x()\nexcept:\n    pass\n"
    assert _n(src, "bare_except") == 1


def test_typed_except_does_not_count():
    src = "try:\n    x()\nexcept ValueError:\n    pass\n"
    assert _n(src, "bare_except") == 0


# --- robustness ---------------------------------------------------------------
def test_syntax_error_source_yields_zeroes_not_crash():
    counts = scan_source("def (:\n")  # unparseable
    assert counts["net_subprocess_no_timeout"] == 0
    assert counts["bare_except"] == 0


def test_method_named_run_is_not_subprocess():
    # pool.run(...) must not be mistaken for subprocess.run
    src = 'pool.run(["ssh", h])\n'
    assert _n(src, "net_subprocess_no_timeout") == 0


def test_skip_parts_are_relative_to_root(tmp_path):
    # Regression: a repo checked out UNDER a skipped path (e.g. a git worktree in
    # .claude/worktrees/) must still scan its own tree. The skip must apply to
    # parts relative to the scan root, not the absolute path.
    root = tmp_path / ".claude" / "worktrees" / "wt"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "a.py").write_text("x = 1\n")
    (root / "venvs").mkdir()
    (root / "venvs" / "skipme.py").write_text("y = 2\n")
    found = {p.name for p in _iter_py_files(root)}
    assert "a.py" in found  # not skipped despite .claude in the absolute path
    assert "skipme.py" not in found  # venvs still skipped (relative)
