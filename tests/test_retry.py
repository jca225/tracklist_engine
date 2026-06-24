"""core.retry: retries only genuine-transient Errs, with exponential backoff."""

from __future__ import annotations

from core.result import Err, Ok
from core.retry import retry


def _counter(results):
    """Return a zero-arg fn that yields the given results in order."""
    seq = iter(results)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return next(seq)

    return fn, calls


def test_returns_ok_without_retrying():
    fn, calls = _counter([Ok(7)])
    sleeps: list[float] = []
    assert retry(fn, attempts=3, sleep=sleeps.append) == Ok(7)
    assert calls["n"] == 1 and sleeps == []


def test_retries_until_ok():
    fn, calls = _counter([Err("net"), Err("net"), Ok(42)])
    sleeps: list[float] = []
    assert retry(fn, attempts=3, base_delay_s=0.5, sleep=sleeps.append) == Ok(42)
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]  # exponential backoff between the 3 attempts


def test_exhausts_and_returns_last_err():
    fn, calls = _counter([Err("a"), Err("b"), Err("c")])
    sleeps: list[float] = []
    assert retry(fn, attempts=3, sleep=sleeps.append) == Err("c")
    assert calls["n"] == 3 and len(sleeps) == 2


def test_does_not_retry_when_retry_on_false():
    # A deterministic failure (e.g. bot-detection) must not be retried.
    fn, calls = _counter([Err("fatal"), Ok(1)])
    sleeps: list[float] = []
    out = retry(fn, attempts=5, retry_on=lambda e: e != "fatal", sleep=sleeps.append)
    assert out == Err("fatal")
    assert calls["n"] == 1 and sleeps == []


def test_retry_on_selective():
    # Retries the transient one, stops at the first non-retryable.
    fn, calls = _counter([Err("transient"), Err("fatal"), Ok(1)])
    sleeps: list[float] = []
    out = retry(
        fn, attempts=5, retry_on=lambda e: e == "transient", sleep=sleeps.append
    )
    assert out == Err("fatal")
    assert calls["n"] == 2 and sleeps == [0.5]


def test_attempts_clamped_to_one():
    fn, calls = _counter([Err("x")])
    assert retry(fn, attempts=0, sleep=lambda _d: None) == Err("x")
    assert calls["n"] == 1
