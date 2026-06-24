"""Turn opaque yt-dlp failures into named, remedied errors.

The two infra failures that recur on the cluster — YouTube bot-detection and a
missing JS runtime — used to surface as raw tracebacks, and the fix lived only
in the ``audio-pipeline-debug`` skill (a human runbook). This module classifies
a yt-dlp stderr/error string into a known :class:`FailureMode` and returns the
exact remediation, so a failed download says *what to do* instead of dumping a
stacktrace. It also offers a proactive :func:`check_environment` so a run can
refuse to start when node is absent.

Detection is signature-based on the error text; remedies reference the skill for
the full procedure. Adding a new mode = one entry in ``_SIGNATURES`` + ``_REMEDY``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    BOT_DETECTION = "bot_detection"
    NO_JS_RUNTIME = "no_js_runtime"
    PO_TOKEN_403 = "po_token_403"
    FORMAT_UNAVAILABLE = "format_unavailable"


# Lowercased substrings that identify each mode. First match wins, in this order
# (more-specific signatures first so e.g. the JS-runtime hint isn't shadowed).
_SIGNATURES: tuple[tuple[FailureMode, tuple[str, ...]], ...] = (
    (
        FailureMode.BOT_DETECTION,
        ("sign in to confirm", "confirm you're not a bot", "confirm you are not a bot"),
    ),
    (
        FailureMode.NO_JS_RUNTIME,
        ("no supported javascript runtime", "no supported js runtime"),
    ),
    (FailureMode.PO_TOKEN_403, ("http error 403", "403 forbidden")),
    (
        FailureMode.FORMAT_UNAVAILABLE,
        (
            "only images are available",
            "requested format is not available",
            "only images",
        ),
    ),
)

_REMEDY: dict[FailureMode, str] = {
    FailureMode.BOT_DETECTION: (
        "YouTube bot-detection: refresh the yt-dlp cookies "
        "(audio-pipeline-debug skill, step 2 — re-export Safari cookies and scp "
        "to pi-storage ~/.config/yt-dlp/cookies.txt)."
    ),
    FailureMode.NO_JS_RUNTIME: (
        "Missing JS runtime: install node and point yt-dlp at it "
        "(audio-pipeline-debug skill, step 1 — js_runtimes=node + remote-components "
        "ejs:github). On the Mac use ingest.ytdlp_profile."
    ),
    FailureMode.PO_TOKEN_403: (
        "HTTP 403 on the media stream (PO-token gate): use "
        "player_client=web_safari with Safari cookies — set DownloadConfig.mac_profile "
        "or call ingest.ytdlp_profile.mac_ytdlp_base (Mac only)."
    ),
    FailureMode.FORMAT_UNAVAILABLE: (
        "Only image formats returned: the n-challenge was not solved — ensure the "
        "JS runtime is configured (see NO_JS_RUNTIME remedy)."
    ),
}


@dataclass(frozen=True)
class EnvIssue:
    ok: bool
    detail: str


def classify_failure(error_text: str | None) -> FailureMode | None:
    """Map a yt-dlp stderr/error string to a known failure mode, or None."""
    if not error_text:
        return None
    low = error_text.lower()
    for mode, needles in _SIGNATURES:
        if any(n in low for n in needles):
            return mode
    return None


def remedy_for(mode: FailureMode) -> str:
    """Human-actionable remediation for a failure mode."""
    return _REMEDY[mode]


def annotate(error_text: str | None) -> str | None:
    """Append the remedy to a yt-dlp error string when it matches a known mode.

    Returns the original text unchanged (or None) when nothing matches, so it is
    safe to wrap every download error detail with this.
    """
    if not error_text:
        return error_text
    mode = classify_failure(error_text)
    if mode is None:
        return error_text
    return f"{error_text}\n  ↳ {mode.value}: {remedy_for(mode)}"


def check_environment(*, require_node: bool = True) -> EnvIssue:
    """Proactive preflight: confirm a JS runtime is present before a run.

    Cheap enough to call at the top of a download entrypoint. Cookie freshness
    can't be checked without a live request, so it's left to runtime classification.
    """
    if require_node and not (shutil.which("node") or shutil.which("nodejs")):
        return EnvIssue(
            ok=False,
            detail=remedy_for(FailureMode.NO_JS_RUNTIME),
        )
    return EnvIssue(ok=True, detail="ok")
