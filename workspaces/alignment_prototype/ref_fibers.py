"""Import shim — fibers moved to the ``fibers/`` package (2026-07-09).

Kept so the 18 existing importers keep working (the ``als_io`` precedent).
New code should import ``workspaces.alignment_prototype.fibers`` directly;
migrate callers opportunistically, then delete this shim.
"""

from __future__ import annotations

from workspaces.alignment_prototype.fibers.detect import *  # noqa: F401,F403
from workspaces.alignment_prototype.fibers.detect import (  # noqa: F401
    _avg_linkage,
    _diag_sim,
    _long_repeats,
)
