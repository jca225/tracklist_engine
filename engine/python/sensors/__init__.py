"""Typed sensor adapters for the Rust DJ kernel.

Sensors speak JSON validated against repo-root ``schemas/``. They do not own
identity or provenance — Rust does. Heavy MIR / yt-dlp / Ableton parse stay
here (or wrap the legacy monolith) and deposit results as content-addressed
artifacts on the kernel side (see ``docs/storage.md``).

Experimental Propose plugs (OD, agentic) live under ``sensors.plugs``.
"""

from sensors.adapters import AbletonStub, AnalyzeStub, DownloadStub
from sensors.schemas import load_schema, validate_payload

__all__ = [
    "AbletonStub",
    "AnalyzeStub",
    "DownloadStub",
    "load_schema",
    "validate_payload",
]
