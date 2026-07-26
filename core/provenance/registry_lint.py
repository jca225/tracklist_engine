"""Registry version-collision fence (fold-in C) — static, merge-safe.

The metaprogramming registry ([registry.py]) declares artifact kinds
(``@artifact_type``) and producers (``@transformation``) into a module singleton
at import. Two branches can each add a declaration with the SAME identity —
``@transformation("parse-session", "3", …)`` — and the runtime guard only trips
when both happen to be imported in one process (and even then only compares the
``fn``, not the version). A merge lands a silent duplicate migration number.

This is the append-only fence the runtime can't see: a pure AST scan of the
source tree that fails when the same declaration identity is declared in more
than one place. Identity is ``kind`` for artifact types and ``(name, version)``
for transformations (the "namespace, name, version" of §2 versioning). Adding a
NEW version of a producer is fine (append-only); re-declaring an existing one is
the collision. stdlib + ``ast`` only, so it rides ``make check`` / CI with no
imports and no provenance DB.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_DECORATORS = {"artifact_type", "transformation"}


@dataclass(frozen=True)
class Declaration:
    file: str
    line: int
    decorator: str  # "artifact_type" | "transformation"
    identity: str  # the collision key ("kind" or "name@version")


def _str_arg(node: ast.expr | None) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _decl_from_call(call: ast.Call, path: str) -> Declaration | None:
    """Extract a Declaration from an ``@artifact_type(...)``/``@transformation(...)``
    decorator call, or None if it isn't one / can't be read statically."""
    func = call.func
    name = (
        func.id
        if isinstance(func, ast.Name)
        else (func.attr if isinstance(func, ast.Attribute) else None)
    )
    if name not in _DECORATORS:
        return None
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    args = call.args
    if name == "artifact_type":
        # artifact_type(kind, media_type, version=1) — identity is kind.
        kind = _str_arg(args[0] if args else None) or _str_arg(kw.get("kind"))
        if kind is None:
            return None
        return Declaration(path, call.lineno, name, f"kind:{kind}")
    # transformation(name, version, ...) — identity is name@version.
    tname = _str_arg(args[0] if len(args) >= 1 else None) or _str_arg(kw.get("name"))
    version = _str_arg(args[1] if len(args) >= 2 else None) or _str_arg(
        kw.get("version")
    )
    if tname is None or version is None:
        return None
    return Declaration(path, call.lineno, name, f"txn:{tname}@{version}")


def scan_source(text: str, path: str) -> list[Declaration]:
    """All registry declarations in one source file (decorator calls only)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[Declaration] = []
    for node in ast.walk(tree):
        # decorators live on FunctionDef/ClassDef; also catch bare calls.
        decos = getattr(node, "decorator_list", [])
        for d in decos:
            if isinstance(d, ast.Call):
                decl = _decl_from_call(d, path)
                if decl is not None:
                    out.append(decl)
    return out


def scan_tree(
    root: Path,
    *,
    skip: tuple[str, ...] = (
        "/tests/",
        "/attic/",
        "/.git/",
        "/.claude/",  # agent worktrees are checkouts of THIS repo, not dup decls
        "/venvs/",
        "/.venv/",
    ),
) -> list[Declaration]:
    decls: list[Declaration] = []
    for py in sorted(root.rglob("*.py")):
        s = str(py)
        if any(part in s for part in skip):
            continue
        decls.extend(scan_source(py.read_text(encoding="utf-8", errors="ignore"), s))
    return decls


def find_collisions(decls: list[Declaration]) -> dict[str, list[Declaration]]:
    """Identities declared in more than one place — the merge collisions.

    A single identity declared once is fine; the same identity in 2+ locations is
    the duplicate a merge would land. (Two decorators on one line at the same
    location are deduped — that's one declaration, not a collision.)"""
    by_identity: dict[str, list[Declaration]] = {}
    for d in decls:
        by_identity.setdefault(d.identity, []).append(d)
    return {
        ident: locs
        for ident, locs in by_identity.items()
        if len({(x.file, x.line) for x in locs}) > 1
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI glue
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "roots", nargs="*", default=None, help="dirs to scan (default: repo)"
    )
    args = p.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    roots = [Path(r) for r in args.roots] if args.roots else [repo]
    decls: list[Declaration] = []
    for r in roots:
        decls.extend(scan_tree(r))
    collisions = find_collisions(decls)
    if not collisions:
        print(f"registry-lint: OK ({len(decls)} declarations, no collisions)")
        return 0
    for ident, locs in sorted(collisions.items()):
        print(f"registry-lint: DUPLICATE {ident} declared in {len(locs)} places:")
        for d in locs:
            print(f"    {d.file}:{d.line}")
    print(
        "registry-lint: registry declarations must be append-only — a repeated "
        "(name, version)/kind is a merge collision. Bump the version or rename."
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
