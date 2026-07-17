"""Tests for scripts/loop_hardening.py — the shared driver-loop hardening
primitives (bug classes C1 escalation / C2 transient-backoff / C3 honest tally,
plus the B2 sha256 helper).
"""

from __future__ import annotations

import logging

from scripts.loop_hardening import (
    MAX_TRANSIENT_ATTEMPTS,
    Counts,
    exit_status,
    register_transient,
    sha256_file,
    transient_backoff_seconds,
)

log = logging.getLogger("test")


def _code(*a, **k):
    return exit_status(*a, **k)[0]


# --- C1: exit_status ----------------------------------------------------------
def test_clean_run_is_success():
    code, level, msg = exit_status(n_done=100, n_failed=0)
    assert code == 0
    assert level == logging.INFO
    assert "analyzed 100" in msg


def test_all_failed_is_error_nonzero():
    code, level, msg = exit_status(n_done=0, n_failed=50)
    assert code == 1
    assert level == logging.ERROR
    assert "ALL work failed" in msg


def test_empty_queue_is_clean():
    code, level, _ = exit_status(n_done=0, n_failed=0)
    assert code == 0
    assert level == logging.INFO


def test_majority_failure_escalates():
    assert _code(40, 60) == 1  # 60% > 50% default


def test_minority_failure_below_limit_is_success():
    assert _code(90, 10) == 0


def test_ratio_limit_is_configurable():
    assert _code(80, 20) == 0
    assert _code(80, 20, fail_ratio_limit=0.1) == 1


def test_persist_failures_escalate_even_with_successes():
    code, level, msg = exit_status(n_done=100, n_failed=0, n_persist_failed=3)
    assert code == 1
    assert level == logging.ERROR
    assert "persist" in msg.lower()


# --- C3: Counts ---------------------------------------------------------------
def test_counts_start_zero_and_increment():
    c = Counts()
    assert c.snapshot() == (0, 0, 0)
    c.done_inc()
    c.done_inc()
    c.failed_inc()
    c.persist_failed_inc()
    assert c.snapshot() == (2, 1, 1)


# --- C2: register_transient ---------------------------------------------------
def test_transient_retries_then_quarantines():
    counts = Counts()
    failed: set[int] = set()
    attempts: dict[int, int] = {}
    # First MAX-1 attempts do NOT quarantine (return False), leave tid pickable.
    for i in range(1, MAX_TRANSIENT_ATTEMPTS):
        quarantined = register_transient(
            7, "rsync timeout", attempts, failed, counts, log
        )
        assert quarantined is False
        assert 7 not in failed
    # The MAX-th attempt quarantines and counts one final failure.
    quarantined = register_transient(7, "rsync timeout", attempts, failed, counts, log)
    assert quarantined is True
    assert 7 in failed
    assert counts.snapshot() == (0, 1, 0)


def test_backoff_is_bounded_and_monotonic():
    assert transient_backoff_seconds(1) == 2
    assert transient_backoff_seconds(2) == 4
    assert transient_backoff_seconds(100) == 30  # capped


# --- B2: sha256_file ----------------------------------------------------------
def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "audio.bin"
    data = b"tracklist-engine" * 5000
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
