"""Bug-audit C3: `fetch_measure_times` must not crash the whole backfill loop
on a NULL/absent `measure_times_json`.

`ssh_pi(sql)` returns "" when the column is NULL or the row is absent.
`json.loads("")` raises `json.JSONDecodeError`, which the loops' original
`except subprocess.CalledProcessError` did not catch — so the first such row
killed the entire backfill and abandoned all remaining work. The fix raises a
caught `ValueError` (JSONDecodeError is itself a ValueError subclass) so the
loop quarantines that one item and continues.
"""

from __future__ import annotations

import pytest

from scripts import mert_backfill_loop, set_mert_backfill_loop


@pytest.mark.parametrize(
    "mod, arg",
    [(mert_backfill_loop, 101), (set_mert_backfill_loop, 5)],
)
@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_fetch_measure_times_raises_valueerror_on_empty(monkeypatch, mod, arg, empty):
    monkeypatch.setattr(mod, "ssh_pi", lambda _sql: empty)
    with pytest.raises(ValueError):
        mod.fetch_measure_times(arg)


@pytest.mark.parametrize(
    "mod, arg",
    [(mert_backfill_loop, 101), (set_mert_backfill_loop, 5)],
)
def test_fetch_measure_times_parses_valid_json(monkeypatch, mod, arg):
    monkeypatch.setattr(mod, "ssh_pi", lambda _sql: "[0.0, 1.5, 3.0]")
    assert mod.fetch_measure_times(arg) == (0.0, 1.5, 3.0)
