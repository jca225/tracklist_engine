"""Real `Scorer` impls + their pure stdout parsers.

The framework treats the canonical scorers as a FROZEN referee — it shells out
to them and parses their console output; it never reimplements or varies them
(framework spec §3). The parsing is pure and unit-tested
(tests/test_scorers.py); the subprocess orchestration is integration (needs
pulled sets) and is exercised by the CLI, not CI.

- compose grain → `score_timeline_vs_gt` (per-axis, strict + fiber-aware)
- isolate grain → `trajectory/train.py` evaluate() lines (strict trajectory_acc)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping

_REPO = Path(__file__).resolve().parent.parent.parent.parent

# Mirrors drivers/race.py `_METRICS` (kept local to avoid importing the heavy
# drivers package just for regexes). If race.py's format changes, update both.
_TIMELINE_PATTERNS: dict[str, re.Pattern] = {
    "identity": re.compile(r"identity:\s+\d+/\d+\s+\((\d+)%\)"),
    "place_median_s": re.compile(r"set placement \|pred-gt\|:\s+median=([\d.]+)s"),
    "ref_median_s": re.compile(r"ref offset \|pred-gt\|.*?median=([\d.]+)s"),
    "traj_headline": re.compile(r"HEADLINE multiseg\+loop n=\s*\d+ traj-acc=(\d+)%"),
    "traj_acappella": re.compile(r"stem\s+acappella\s+n=\s*\d+ traj-acc=(\d+)%"),
}

# `[<tag>] traj-acc ALL 0.421 | HEADLINE multiseg+loop 0.375`
_TRAJ_LINE = re.compile(
    r"\[([^\]]+)\]\s+traj-acc ALL\s+([\d.]+)\s+\|\s+HEADLINE multiseg\+loop\s+([\d.]+)"
)


def parse_timeline_metrics(stdout: str) -> dict[str, float]:
    """Extract the score_timeline_vs_gt headline numbers. A field that does not
    appear is ABSENT from the dict (never silently 0 — a missing metric is a
    real signal, not a zero)."""
    out: dict[str, float] = {}
    for key, pat in _TIMELINE_PATTERNS.items():
        m = pat.search(stdout)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_trajectory_metrics(stdout: str) -> dict[str, dict[str, float]]:
    """Map each evaluate() tag -> {'all':, 'headline':} (fractions in [0,1])."""
    out: dict[str, dict[str, float]] = {}
    for tag, allv, head in _TRAJ_LINE.findall(stdout):
        out[tag.strip()] = {"all": float(allv), "headline": float(head)}
    return out


def timeline_metrics_to_referee(metrics: Mapping[str, float]) -> dict[str, float]:
    """Adapt parsed timeline metrics to the runner's referee contract: `strict`
    is the multiseg+loop headline as a fraction in [0,1] (the runner's
    learned-nothing check reads `strict`)."""
    out = dict(metrics)
    if "traj_headline" in metrics:
        out["strict"] = metrics["traj_headline"] / 100.0
    return out


# --------------------------------------------------------------------------- #
# subprocess Scorer impls (integration)                                        #
# --------------------------------------------------------------------------- #


def _run(argv: list[str]) -> str:
    proc = subprocess.run(argv, cwd=str(_REPO), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"scorer failed: {' '.join(argv)}\n{out}")
    return out


class TimelineScorer:
    """Compose-grain referee: scores a written PredictedTimeline through
    `score_timeline_vs_gt` (strict + fiber-aware) exactly as `make race` does."""

    def score(self, artifact: object, ctx: object) -> Mapping[str, float]:
        timeline = Path(artifact)  # backend returns the timeline path
        set_id = getattr(ctx, "set_id")
        base = [
            sys.executable,
            "-m",
            "alignment.score_timeline_vs_gt",
            "--set-id",
            set_id,
            "--timeline",
            str(timeline),
        ]
        strict = timeline_metrics_to_referee(parse_timeline_metrics(_run(base)))
        fiber = parse_timeline_metrics(_run(base + ["--fibers"]))
        if "traj_headline" in fiber:
            strict["fiber"] = fiber["traj_headline"] / 100.0
        return strict


__all__ = [
    "parse_timeline_metrics",
    "parse_trajectory_metrics",
    "timeline_metrics_to_referee",
    "TimelineScorer",
]
