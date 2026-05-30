#!/usr/bin/env python3
"""Mechanical guardrails — stale names, legacy columns, wrong repo depth.

Run from repo root::

    python scripts/guardrails.py

Exit 0 if clean, 1 with actionable errors on stdout.
Used by git pre-commit, Cursor afterFileEdit hook, and GitHub Actions.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = frozenset({"venvs", "cue-detr", "data", "__pycache__", ".git"})

# Intentional uses of legacy strings (DB source labels, etc.)
AUDIO_PIPELINE_ALLOW_SUBSTR = frozenset({"audio_pipeline_v1"})

VARIANT_TAG_ALLOW_FILES = frozenset(
    {
        REPO_ROOT / "core" / "identity.py",
        REPO_ROOT / "scripts" / "guardrails.py",
    }
)

STALE_MODULE_SKIP_FILES = frozenset({REPO_ROOT / "scripts" / "guardrails.py"})

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
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
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


def _check_adapter_parents(path: Path, text: str) -> list[Violation]:
    in_adapter_dir = any(
        path == d or d in path.parents for d in ADAPTER_PARENTS_DIRS
    )
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
