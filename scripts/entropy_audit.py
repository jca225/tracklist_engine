"""Entropy / bug-class audit — the deterministic layer of the weekly-audit process.

AST-based detector that fences the classes from the 2026-07-16 ingest/pull/encoding
audit so they cannot regress after being fixed. The line-based ratchet in
`guardrails.py` can't see these — they are about a *missing keyword argument* on a
multi-line call — so this uses `ast`.

Fenced classes (all target the PRECISE bug, to avoid false positives on benign
local subprocess calls):

- `net_subprocess_no_timeout` — a `subprocess.run/check_call/check_output/Popen/call`
  whose args mention ssh / rsync / sqlite3, with no `timeout=` (bug B1: an
  unbounded network/DB call hangs the loop).
- `net_subprocess_no_encoding` — the same net calls in text mode
  (`text=True`/`universal_newlines=True`) with no explicit `encoding=` (bug A2:
  latin-1 mojibake at the Mac/pi boundary).
- `bare_except` — a bare `except:` (silent-failure class; also swallows
  SystemExit/KeyboardInterrupt).

The net-subprocess fences are scoped to the pipeline modules that actually talk to
pi-storage (`scripts/`, `ingest/`, `analysis/`); `bare_except` is repo-wide.

Modes:
  entropy_audit.py --snapshot   print current counts (JSON + human)
  entropy_audit.py --check      compare against the baseline; nonzero on regression
  entropy_audit.py --bump       rewrite the baseline to current counts

`guardrails.py` calls `check()` so this rides the existing `make check` / pre-commit
/ CI gate with no extra wiring. The baseline lives in `scripts/entropy_ratchet.json`.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "entropy_ratchet.json"

# Precise bug class → the classes fenced. Keys are stable (baseline JSON + guardrails).
BUG_CLASSES = (
    "net_subprocess_no_timeout",
    "net_subprocess_no_encoding",
    "bare_except",
)

_NET_TOKENS = ("ssh", "rsync", "sqlite3")
_SUBPROCESS_FUNCS = frozenset({"run", "check_call", "check_output", "Popen", "call"})
# net-subprocess fences apply only where we actually shell out to pi-storage.
_NET_SCOPE_ROOTS = ("scripts", "ingest", "analysis")
_SKIP_PARTS = frozenset(
    {"venvs", "cue-detr", "__pycache__", ".git", "node_modules", ".claude"}
)


def _is_subprocess_call(node: ast.Call) -> bool:
    """True for `subprocess.<func>(...)` — the attribute form the codebase uses.
    Deliberately does NOT match a bare `run(...)` or `pool.run(...)`, so unrelated
    `.run` methods aren't mistaken for subprocess."""
    f = node.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr in _SUBPROCESS_FUNCS
        and isinstance(f.value, ast.Name)
        and f.value.id == "subprocess"
    )


def _mentions_net(node: ast.Call) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            low = sub.value.lower()
            if any(tok in low for tok in _NET_TOKENS):
                return True
    return False


def _kwargs(node: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}


def _is_true(expr: ast.expr | None) -> bool:
    return isinstance(expr, ast.Constant) and expr.value is True


def scan_source(text: str, *, net_scoped: bool = True) -> Counter:
    """Count bug-class occurrences in one source string. `net_scoped=False` still
    counts net-subprocess classes (used when the caller already scoped by path)."""
    counts: Counter = Counter({k: 0 for k in BUG_CLASSES})
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return counts  # unparseable → contribute nothing, never crash the gate
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            counts["bare_except"] += 1
        elif isinstance(node, ast.Call) and _is_subprocess_call(node):
            if not _mentions_net(node):
                continue
            kw = _kwargs(node)
            if "timeout" not in kw:
                counts["net_subprocess_no_timeout"] += 1
            text_mode = _is_true(kw.get("text")) or _is_true(
                kw.get("universal_newlines")
            )
            if text_mode and "encoding" not in kw:
                counts["net_subprocess_no_encoding"] += 1
    return counts


def _iter_py_files(root: Path):
    # Skip on parts RELATIVE to the scan root, not absolute — otherwise a repo
    # checked out under a skipped path (e.g. a git worktree living in
    # .claude/worktrees/) would skip its own entire tree.
    root = root.resolve()
    for path in sorted(root.rglob("*.py")):
        if _SKIP_PARTS & set(path.relative_to(root).parts):
            continue
        yield path


def _in_net_scope(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return any(rel == r or rel.startswith(r + "/") for r in _NET_SCOPE_ROOTS)


def scan_repo(root: Path = REPO_ROOT) -> Counter:
    """Repo-wide counts: net-subprocess classes only under _NET_SCOPE_ROOTS,
    bare_except everywhere."""
    totals: Counter = Counter({k: 0 for k in BUG_CLASSES})
    for path in _iter_py_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        local = scan_source(text)
        totals["bare_except"] += local["bare_except"]
        if _in_net_scope(path):
            totals["net_subprocess_no_timeout"] += local["net_subprocess_no_timeout"]
            totals["net_subprocess_no_encoding"] += local["net_subprocess_no_encoding"]
    return totals


def _load_baseline() -> dict[str, int] | None:
    if not BASELINE_PATH.is_file():
        return None
    return json.loads(BASELINE_PATH.read_text())


def check(root: Path = REPO_ROOT) -> list[str]:
    """Return a list of regression messages (empty = clean). Called by guardrails."""
    baseline = _load_baseline()
    if baseline is None:
        return []  # disarmed until a baseline is committed
    counts = scan_repo(root)
    msgs: list[str] = []
    for key in BUG_CLASSES:
        allowed = baseline.get(key)
        if allowed is None:
            continue
        if counts[key] > allowed:
            msgs.append(
                f"entropy_audit:{key} {counts[key]} > baseline {allowed} — a new "
                f"instance of this bug class was introduced; fix it, or if truly "
                f"intentional raise the baseline in {BASELINE_PATH.name} with justification"
            )
    return msgs


def _print_report(counts: Counter, baseline: dict[str, int] | None) -> None:
    print(json.dumps(dict(counts), indent=2))
    print("\nbug-class fences:")
    for key in BUG_CLASSES:
        base = "—" if baseline is None else baseline.get(key, "—")
        print(f"  {key:32s} {counts[key]:4d}  (baseline {base})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="entropy / bug-class audit")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--snapshot", action="store_true", help="print current counts")
    g.add_argument(
        "--check", action="store_true", help="fail on regression vs baseline"
    )
    g.add_argument(
        "--bump", action="store_true", help="rewrite baseline to current counts"
    )
    args = p.parse_args(argv)

    counts = scan_repo()
    if args.bump:
        BASELINE_PATH.write_text(json.dumps(dict(counts), indent=2) + "\n")
        print(f"entropy_audit: baseline written to {BASELINE_PATH.name}")
        _print_report(counts, dict(counts))
        return 0
    if args.check:
        msgs = check()
        if msgs:
            for m in msgs:
                print(m, file=sys.stderr)
            return 1
        print("entropy_audit: OK")
        return 0
    # --snapshot (explicit) or no flag at all → report current counts
    if args.snapshot or not (args.check or args.bump):
        _print_report(counts, _load_baseline())
    return 0


if __name__ == "__main__":
    sys.exit(main())
