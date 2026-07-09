"""Import shim — moved to ``fibers/harmony.py`` (2026-07-09).

New code should import ``workspaces.alignment_prototype.fibers`` directly;
migrate callers opportunistically, then delete this shim.
"""

from __future__ import annotations

from workspaces.alignment_prototype.fibers.harmony import *  # noqa: F401,F403
