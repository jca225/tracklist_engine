#!/usr/bin/env python3
"""Docs garbage collector — mark & sweep stale docs by *reachability*.

The problem: dated snapshots (handoffs, bearings, reviews, corrections) pile up
in ``docs/`` and no one can tell which are current. This is a GC problem: the
``/align-checkpoint`` discipline is the *mutator* (stops allocating new dated
handoffs); this is the *collector* (sweeps what's already dead).

**Liveness = reachability, not age.** A doc is LIVE if it is transitively linked
from a *root* (any ``CLAUDE.md``, the canonical alignment docs, or a skill).
That correctly keeps a dated handoff that CLAUDE.md still cites as an active
caveat (e.g. the reconcile "do not re-run --apply" warning) while collecting
dated handoffs nothing points at anymore.

Classification of each top-level ``docs/*.md``:
  - **LIVE**         — reachable from a root. Keep.
  - **COLLECTABLE**  — dated-snapshot pattern AND unreachable. Sweep → docs/archive/.
  - **ORPHAN**       — unreachable but NOT dated (could be a real standalone doc
                       that just isn't linked yet). Reported, never auto-swept.

Usage::

    python scripts/docs_gc.py            # classify + report (dry run)
    python scripts/docs_gc.py --apply    # move COLLECTABLE → docs/archive/, fix refs

Run from repo root. Archives (git mv), never deletes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"
# Memory is also an orientation root: a design doc that memory points at is LIVE,
# even if no CLAUDE.md links it. Its `../../docs/X.md` links are cross-tree and
# don't filesystem-resolve, so reachability matches those by basename (below).
MEMORY_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-johnnycabrahams-Desktop-tracklist-engine"
    / "memory"
)

# A doc is a dated snapshot if its name matches any of these (superseded-by-time
# artifacts: session handoffs, dated reviews/bearings/corrections/suspects).
DATED_RE = re.compile(
    r"_\d{8}\.md$|handoff|_review_|bearings|corrections|_suspects_|failure_tables",
    re.IGNORECASE,
)

_MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _canonical_roots() -> list[Path]:
    """Files whose links define 'live' — CLAUDE.md, canonical align docs, skills."""
    roots: list[Path] = []
    for p in REPO_ROOT.rglob("CLAUDE.md"):
        if any(part in {"venvs", ".claude", "__pycache__"} for part in p.parts):
            continue
        roots.append(p)
    roots += [p for p in REPO_ROOT.glob(".claude/skills/*/SKILL.md")]
    if MEMORY_DIR.is_dir():
        roots += sorted(MEMORY_DIR.glob("*.md"))
    for name in (
        "alignment_state_of_record.md",
        "alignment_status.md",
        "alignment_recharacterization.md",
        "alignment_objective.md",
        "architecture_north_star.md",
    ):
        p = DOCS_DIR / name
        if p.is_file():
            roots.append(p)
    return roots


def _md_targets(path: Path) -> list[Path]:
    """Absolute paths of local .md links in `path` (skip URLs/anchors)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[Path] = []
    for m in _MD_LINK_RE.finditer(text):
        target = m.group(1).strip().split("#", 1)[0].strip()
        if not target or "://" in target or not target.endswith(".md"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            # cross-tree link (e.g. memory's ../../docs/X.md) — match by basename
            by_name = DOCS_DIR / Path(target).name
            if by_name.is_file():
                resolved = by_name
        out.append(resolved)
    return out


def reachable_docs() -> set[Path]:
    """Transitive closure of .md links from the roots that land in docs/."""
    seen: set[Path] = set()
    queue = list(_canonical_roots())
    while queue:
        cur = queue.pop()
        for tgt in _md_targets(cur):
            if tgt in seen or not tgt.is_file():
                continue
            seen.add(tgt)
            queue.append(tgt)  # follow links transitively, even out of docs/
    return {p for p in seen if p.parent == DOCS_DIR}


def classify() -> dict[str, list[Path]]:
    reachable = reachable_docs()
    live: list[Path] = []
    collectable: list[Path] = []
    orphan: list[Path] = []
    for p in sorted(DOCS_DIR.glob("*.md")):
        if p in reachable:
            live.append(p)
        elif DATED_RE.search(p.name):
            collectable.append(p)
        else:
            orphan.append(p)
    return {"live": live, "collectable": collectable, "orphan": orphan}


def _rewrite_refs(basename: str) -> None:
    """Repoint inbound links from `docs/<basename>` to `docs/archive/<basename>`."""
    for md in REPO_ROOT.rglob("*.md"):
        if any(part in {"venvs", ".claude", "__pycache__"} for part in md.parts):
            continue
        if md.parent == ARCHIVE_DIR:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        new = text.replace(f"docs/{basename}", f"docs/archive/{basename}")
        if md.parent == DOCS_DIR:  # within docs/, links are bare
            new = new.replace(f"]({basename}", f"](archive/{basename}")
        if new != text:
            md.write_text(new, encoding="utf-8")
            print(f"  ref-fixed {md.relative_to(REPO_ROOT)}")


def apply_sweep(collectable: list[Path]) -> None:
    if not collectable:
        print("nothing to collect.")
        return
    ARCHIVE_DIR.mkdir(exist_ok=True)
    for p in collectable:
        dest = ARCHIVE_DIR / p.name
        subprocess.run(["git", "mv", str(p), str(dest)], cwd=REPO_ROOT, check=True)
        print(f"archived {p.name}")
        _rewrite_refs(p.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true", help="move COLLECTABLE → docs/archive/"
    )
    args = ap.parse_args()

    groups = classify()
    print(
        f"docs/: {len(groups['live'])} live · "
        f"{len(groups['collectable'])} collectable · "
        f"{len(groups['orphan'])} orphan (unlinked, undated)"
    )
    if groups["collectable"]:
        print("\nCOLLECTABLE (dated + unreachable → docs/archive/):")
        for p in groups["collectable"]:
            print(f"  {p.name}")
    if groups["orphan"]:
        print("\nORPHAN (unlinked + undated — review by hand, NOT auto-swept):")
        for p in groups["orphan"]:
            print(f"  {p.name}")

    if args.apply:
        print("\n--apply: sweeping COLLECTABLE …")
        apply_sweep(groups["collectable"])
    elif groups["collectable"]:
        print("\n(dry run — rerun with --apply to archive the collectable set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
