#!/usr/bin/env python3
"""Mechanical guardrails — stale names, legacy columns, wrong repo depth.

Run from repo root::

    python scripts/guardrails.py

Exit 0 if clean, 1 with actionable errors on stdout.
Used by git pre-commit, Cursor afterFileEdit hook, and GitHub Actions.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Vendored third-party trees (own LICENSE, upstream-authored) are not ours to
# police — cue-detr (DETR cue model), msst_webui (MSST-WebUI separation trainer).
# .claude holds agent worktrees (full repo copies) — scanning them double-counts
# every ratchet baseline.
SKIP_DIR_NAMES = frozenset(
    {"venvs", "cue-detr", "msst_webui", "data", "__pycache__", ".git", ".claude"}
)

# Intentional uses of legacy strings (DB source labels, etc.)
AUDIO_PIPELINE_ALLOW_SUBSTR = frozenset({"audio_pipeline_v1"})

VARIANT_TAG_ALLOW_FILES = frozenset(
    {
        REPO_ROOT / "core" / "identity.py",
        REPO_ROOT / "scripts" / "guardrails.py",
    }
)

STALE_MODULE_SKIP_FILES = frozenset({REPO_ROOT / "scripts" / "guardrails.py"})

DOC_STALE_PIPELINE_RE = re.compile(
    r"audio_pipeline/(?:analysis|adapters|main)",
)

DOCS_DIR = REPO_ROOT / "docs"

ADAPTER_PARENTS_DIRS = (
    REPO_ROOT / "analysis" / "adapters",
    REPO_ROOT / "ingest" / "adapters",
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    check: str
    detail: str

    def format(self) -> str:
        rel = self.path.relative_to(REPO_ROOT)
        return f"{rel}:{self.line}: [{self.check}] {self.detail}"


def _iter_py_files() -> list[Path]:
    out: list[Path] = []
    root = REPO_ROOT.resolve()
    for path in root.rglob("*.py"):
        # Skip on parts RELATIVE to the repo root, not absolute — otherwise a
        # checkout living under a skipped dir (e.g. a git worktree in
        # .claude/worktrees/) skips its own entire tree and the gate scans
        # nothing. From the main repo, .claude/worktrees copies are still skipped.
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        out.append(path)
    return sorted(out)


def _line_has_allowlisted_substr(line: str, allow: frozenset[str]) -> bool:
    return any(token in line for token in allow)


def _check_stale_audio_pipeline(path: Path, text: str) -> list[Violation]:
    if path in STALE_MODULE_SKIP_FILES:
        return []
    patterns = (
        (re.compile(r"\bimport\s+audio_pipeline\b"), "stale import audio_pipeline"),
        (re.compile(r"\bfrom\s+audio_pipeline\b"), "stale from audio_pipeline"),
        (re.compile(r"audio_pipeline/"), "stale path audio_pipeline/"),
        (re.compile(r"audio_pipeline\."), "stale module ref audio_pipeline."),
        (re.compile(r"-m\s+audio_pipeline\b"), "stale -m audio_pipeline entrypoint"),
    )
    violations: list[Violation] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _line_has_allowlisted_substr(line, AUDIO_PIPELINE_ALLOW_SUBSTR):
            continue
        for pat, detail in patterns:
            if pat.search(line):
                violations.append(Violation(path, lineno, "stale_module", detail))
                break
    return violations


def _check_stale_data_analysis(path: Path, text: str) -> list[Violation]:
    if path in STALE_MODULE_SKIP_FILES:
        return []
    patterns = (
        (re.compile(r"\bimport\s+data_analysis\b"), "stale import data_analysis"),
        (re.compile(r"\bfrom\s+data_analysis\b"), "stale from data_analysis"),
        (re.compile(r"data_analysis/"), "stale path data_analysis/"),
    )
    violations: list[Violation] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat, detail in patterns:
            if pat.search(line):
                violations.append(Violation(path, lineno, "stale_module", detail))
                break
    return violations


def _check_variant_tag(path: Path, text: str) -> list[Violation]:
    if path in VARIANT_TAG_ALLOW_FILES:
        return []
    if path.suffix == ".py" and path.parent.name == "scripts":
        if path.name.startswith("migrate_") or path.suffix == ".sql":
            return []
    pat = re.compile(r"\bvariant_tag\b")
    violations: list[Violation] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if pat.search(line):
            violations.append(
                Violation(
                    path,
                    lineno,
                    "legacy_column",
                    "variant_tag — use stem (identity axis)",
                )
            )
    return violations


def _check_stale_audio_pipeline_docs(path: Path, text: str) -> list[Violation]:
    """Flag markdown that still links to pre-split audio_pipeline/ code paths."""
    violations: list[Violation] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if DOC_STALE_PIPELINE_RE.search(line):
            violations.append(
                Violation(
                    path,
                    lineno,
                    "stale_doc_path",
                    "audio_pipeline/ code path in doc — use ingest/ or analysis/",
                )
            )
    return violations


# The .als codec's core (docs/als_interpreter_plan.md §5). Internal package,
# but the layering is enforced: core modules carry generic-Ableton knowledge
# ONLY — no project-side imports. identity.py / tags.py are the project-facing
# half; __init__.py is the facade composing both.
ALS_CORE_DIR = REPO_ROOT / "labeling" / "als"
ALS_CORE_FILES = frozenset(
    {
        "cst.py",
        "models.py",
        "read.py",
        "semantics.py",
        "validate.py",
        "write.py",
        "roundtrip.py",
    }
)
ALS_CORE_FORBIDDEN = (
    (
        re.compile(r"(?:from|import)\s+labeling\.als\.(?:identity|tags)\b"),
        "codec-core als module imports the private half (identity/tags)",
    ),
    (
        re.compile(
            r"(?:from|import)\s+(?:core|web_crawler|ingest|analysis|tokenizer|eda|personalization|workspaces)\b"
        ),
        "codec-core als module imports project-side code",
    ),
    (
        re.compile(r"from\s+labeling\s+import|from\s+labeling\.(?!als\b)"),
        "codec-core als module imports labeling outside the codec",
    ),
)


def _check_als_core_boundary(path: Path, text: str) -> list[Violation]:
    if path.parent != ALS_CORE_DIR or path.name not in ALS_CORE_FILES:
        return []
    violations: list[Violation] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat, detail in ALS_CORE_FORBIDDEN:
            if pat.search(line):
                violations.append(Violation(path, lineno, "als_core_boundary", detail))
                break
    return violations


def _check_adapter_parents(path: Path, text: str) -> list[Violation]:
    in_adapter_dir = any(path == d or d in path.parents for d in ADAPTER_PARENTS_DIRS)
    if not in_adapter_dir:
        return []
    pat = re.compile(r"parents\[3\]")
    violations: list[Violation] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if pat.search(line):
            violations.append(
                Violation(
                    path,
                    lineno,
                    "repo_depth",
                    "parents[3] in adapter — use parents[2] for repo root",
                )
            )
    return violations


# A declared-but-unread argparse flag is a silent lie: you pass it, nothing
# happens (the `--fiber-k` / `--tol-s` class). Detect statically, but stay
# conservative — this check blocks commits, so it must never false-positive.
# Files that reach flags dynamically are skipped entirely.
_DYNAMIC_ARGS_ACCESS = re.compile(
    r"\bvars\s*\(|\bgetattr\s*\(|\basdict\s*\(|namespace\s*="
)


def _argparse_dest(call: ast.Call) -> str | None:
    """The ``dest`` an ``add_argument`` call binds, or None if uninferable.

    Explicit ``dest=`` wins; otherwise derive from the first ``--long`` option.
    Returns None for positional args (always read by name) and short-only opts.
    """
    for kw in call.keywords:
        if (
            kw.arg == "dest"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    for arg in call.args:
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            continue
        opt = arg.value
        if opt.startswith("--"):
            return opt[2:].replace("-", "_")
        if opt.startswith("-"):
            continue  # short option — keep looking for a long one
        return None  # positional argument
    return None


def _check_dead_argparse_flags(path: Path, text: str) -> list[Violation]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []  # a broken file is CI/pytest's problem, not this check's
    if _DYNAMIC_ARGS_ACCESS.search(text):
        return []  # dynamic flag access — can't statically prove one unused
    declared: dict[str, int] = {}  # dest -> first add_argument lineno
    accessed: set[str] = set()  # every `<obj>.<attr>` name in the file
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            dest = _argparse_dest(node)
            if dest is not None:
                declared.setdefault(dest, node.lineno)
        elif isinstance(node, ast.Attribute):
            accessed.add(node.attr)
    violations: list[Violation] = []
    for dest, lineno in sorted(declared.items(), key=lambda kv: kv[1]):
        if dest not in accessed:
            violations.append(
                Violation(
                    path,
                    lineno,
                    "dead_flag",
                    f"--{dest.replace('_', '-')} declared but args.{dest} never read",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Ratchet checks — entropy counters that may only go DOWN.
#
# Each detector greps for a known failure-class pattern
# (docs/entropy_reduction_plan.md, workstream F). The committed baseline in
# guardrails_ratchet.json is the maximum allowed count; introducing a new
# occurrence fails `make check`, removing occurrences prompts a baseline
# lowering. Attic (closed experiments) and tests are exempt.

RATCHET_BASELINE_PATH = REPO_ROOT / "scripts" / "guardrails_ratchet.json"

_KNOWN_SET_IDS = r"(?:1fsnxchk|2nvzlh2k|w1mgcjt|pwgrrb1|1rfb0yl9)"

RATCHET_PATTERNS: dict[str, re.Pattern[str]] = {
    # a hardcoded set id used as a default / constant (the --gt=BB12 class)
    "set_id_default": re.compile(rf'(?:default\s*=\s*|=\s*)"{_KNOWN_SET_IDS}"'),
    # raw manifest.json parsing outside core/contracts (schema-drift class)
    "raw_manifest_read": re.compile(r"manifest\.json"),
    # relative-depth path anchoring (path-resolution class)
    "parents_depth": re.compile(r"parents\[[0-9]\]"),
}

# contracts is the sanctioned home for artifact reads — its loaders naming
# the artifact files is the FIX for the class, not an instance of it
_RATCHET_SKIP_PARTS = frozenset({"attic", "tests", "contracts"})

# Scoped ratchets: (roots, pattern) counted ONLY under the named roots.
# kernel_flags = CLI surface of the kernel path (W1, kernel_data_engine_plan):
# a flag that has been default-on for two weeks becomes code and its flag
# dies, so this number only goes down.
_KERNEL_ROOTS = (
    "workspaces/alignment_prototype/infer.py",
    "workspaces/alignment_prototype/joint_ref_decode.py",
    "workspaces/alignment_prototype/drivers",
)
SCOPED_RATCHETS: dict[str, tuple[tuple[str, ...], re.Pattern[str]]] = {
    "kernel_flags": (_KERNEL_ROOTS, re.compile(r"add_argument\(")),
}


def _under_roots(path: Path, roots: tuple[str, ...]) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return any(rel == r or rel.startswith(r + "/") for r in roots)


def _ratchet_counts() -> dict[str, int]:
    counts = {name: 0 for name in RATCHET_PATTERNS}
    counts.update({name: 0 for name in SCOPED_RATCHETS})
    for path in _iter_py_files():
        if _RATCHET_SKIP_PARTS & set(path.relative_to(REPO_ROOT.resolve()).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = text.splitlines()
        for name, pat in RATCHET_PATTERNS.items():
            counts[name] += sum(1 for line in lines if pat.search(line))
        for name, (roots, pat) in SCOPED_RATCHETS.items():
            if _under_roots(path, roots):
                counts[name] += sum(1 for line in lines if pat.search(line))
    return counts


def _check_ratchets() -> list[Violation]:
    import json

    if not RATCHET_BASELINE_PATH.is_file():
        return []  # no baseline committed yet — ratchet disarmed
    baseline: dict[str, int] = json.loads(RATCHET_BASELINE_PATH.read_text())
    violations: list[Violation] = []
    for name, count in _ratchet_counts().items():
        allowed = baseline.get(name)
        if allowed is None:
            continue
        if count > allowed:
            violations.append(
                Violation(
                    RATCHET_BASELINE_PATH,
                    0,
                    f"ratchet:{name}",
                    f"{count} occurrences > baseline {allowed} — a new instance of "
                    f"this failure class was introduced (pattern "
                    f"{RATCHET_PATTERNS[name].pattern!r}); fix it or, if truly "
                    "intentional, raise the baseline with justification",
                )
            )
        elif count < allowed:
            # progress! not a violation — nudge to lock it in
            print(
                f"guardrails: ratchet {name} improved {allowed} -> {count}; "
                f"lower it in {RATCHET_BASELINE_PATH.name}"
            )
    return violations


def run_checks() -> list[Violation]:
    violations: list[Violation] = []
    for path in _iter_py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(
                Violation(path, 0, "read_error", str(exc)),
            )
            continue
        violations.extend(_check_stale_audio_pipeline(path, text))
        violations.extend(_check_stale_data_analysis(path, text))
        violations.extend(_check_variant_tag(path, text))
        violations.extend(_check_adapter_parents(path, text))
        violations.extend(_check_als_core_boundary(path, text))
        violations.extend(_check_dead_argparse_flags(path, text))
    if DOCS_DIR.is_dir():
        for path in sorted(DOCS_DIR.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                violations.append(Violation(path, 0, "read_error", str(exc)))
                continue
            violations.extend(_check_stale_audio_pipeline_docs(path, text))
    violations.extend(_check_ratchets())
    # AST-based bug-class fences (subprocess-no-timeout / -no-encoding, bare
    # except) live in entropy_audit.py — the line-based ratchet above can't see a
    # missing kwarg on a multi-line call. Ride the same gate.
    try:  # runs both as `python scripts/guardrails.py` and as `scripts.guardrails`
        from scripts.entropy_audit import BASELINE_PATH as _EA_BASELINE
        from scripts.entropy_audit import check as _entropy_check
    except ImportError:
        from entropy_audit import BASELINE_PATH as _EA_BASELINE
        from entropy_audit import check as _entropy_check

    for msg in _entropy_check():
        violations.append(Violation(_EA_BASELINE, 0, "entropy_audit", msg))
    return violations


def main() -> int:
    violations = run_checks()
    if not violations:
        print("guardrails: OK")
        return 0
    print(f"guardrails: {len(violations)} violation(s)", file=sys.stderr)
    for v in violations:
        print(v.format(), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
