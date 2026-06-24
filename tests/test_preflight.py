"""Preflight classifier: the two recurring infra failures must be named, and
unknown errors must pass through untouched."""

from __future__ import annotations

from ingest.preflight import (
    FailureMode,
    annotate,
    check_environment,
    classify_failure,
    remedy_for,
)


def test_classifies_bot_detection():
    err = "ERROR: [youtube] abc: Sign in to confirm you're not a bot"
    assert classify_failure(err) is FailureMode.BOT_DETECTION


def test_classifies_no_js_runtime():
    err = "ERROR: No supported JavaScript runtime could be found"
    assert classify_failure(err) is FailureMode.NO_JS_RUNTIME


def test_classifies_403():
    assert (
        classify_failure("unable to download: HTTP Error 403: Forbidden")
        is FailureMode.PO_TOKEN_403
    )


def test_classifies_only_images():
    assert classify_failure(
        "Requested format is not available; only images are available"
    ) in (
        FailureMode.FORMAT_UNAVAILABLE,
        FailureMode.NO_JS_RUNTIME,
    )


def test_unknown_error_is_none_and_passes_through():
    err = "ERROR: HTTP Error 500: Internal Server Error"
    assert classify_failure(err) is None
    assert annotate(err) == err  # unchanged


def test_annotate_appends_remedy_for_known():
    err = "Sign in to confirm you're not a bot"
    out = annotate(err)
    assert out is not None and err in out and "cookies" in out.lower()


def test_annotate_handles_none_and_empty():
    assert annotate(None) is None
    assert annotate("") == ""


def test_every_mode_has_a_remedy():
    for mode in FailureMode:
        assert remedy_for(mode)


def test_check_environment_returns_envissue():
    issue = check_environment(require_node=False)
    assert issue.ok and issue.detail == "ok"
