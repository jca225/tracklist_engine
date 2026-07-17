"""Shared timeline-provenance utilities.

Lifted verbatim from ``experiments/bb_baselines.py`` (~lines 247-288).
``bb_baselines`` re-imports from here so its existing call sites and tests
keep working without modification.

Provenance functions are deliberately *not* pure (``_git_sha`` shells out,
``driver_provenance`` hits the filesystem) but this is a lift-and-shift — no
behaviour change.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-level constants shared with bb_baselines that are needed to resolve
# driver-timeline paths.  bb_baselines imports these constants from the parent
# repo, so we replicate the same derivation here to stay self-contained.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent  # …/alignment_prototype/../../..

# Our drivers, read from existing timelines: name -> out/ filename template
# (must stay in sync with bb_baselines.DRIVERS — if that dict changes, update
# here too; keep them identical so driver_provenance stays correct).
DRIVERS: dict[str, str] = {
    "classical": "{sid}_classical_timeline.json",
    "agentic": "{sid}_agentic_timeline.json",
}

_OUT_DIR = _REPO / "workspaces" / "alignment_prototype" / "out"


# ------------------------------------------------ provenance / cohort guard
# The 2026-07-17 scare: the harness silently scored ``out/`` driver timelines
# from DIFFERENT dates (classical 07-11 vs agentic 07-14) as if they were one
# run, and compared them to a superseded race-board snapshot.  Same scorer,
# mismatched artifacts.  These functions make that impossible: every driver
# timeline's mtime is captured, printed, and gated — a within-set cohort that
# spans too much time fails LOUDLY instead of masquerading as a clean
# comparison.  (Baselines are recomputed from audio each run, so they carry no
# staleness risk.)
def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def driver_provenance(set_id: str, methods: list[str]) -> dict[str, dict]:
    """Per DRIVER method: the timeline file's path, existence, and mtime."""
    prov: dict[str, dict] = {}
    for name in methods:
        if name not in DRIVERS:
            continue
        tl = _OUT_DIR / DRIVERS[name].format(sid=set_id)
        if tl.is_file():
            mt = tl.stat().st_mtime
            prov[name] = {
                "path": DRIVERS[name].format(sid=set_id),
                "mtime_epoch": mt,
                "mtime": datetime.fromtimestamp(mt, timezone.utc).isoformat(
                    timespec="minutes"
                ),
                "exists": True,
            }
        else:
            prov[name] = {"path": str(tl), "exists": False}
    return prov


def cohort_spread_s(prov: dict[str, dict]) -> float | None:
    """Max mtime spread (seconds) among a set's EXISTING driver timelines, or None
    if fewer than two exist.  Driver timelines compared WITHIN one set must come
    from one coherent run; a large spread means mismatched vintages (the bug)."""
    mts = [p["mtime_epoch"] for p in prov.values() if p.get("exists")]
    return (max(mts) - min(mts)) if len(mts) >= 2 else None
