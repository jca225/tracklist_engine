"""Cross-artifact join guard.

Any function that combines two set-scoped artifacts must prove they describe
the SAME set. On 2026-07-09 the scorer silently joined BB11 predictions to
BB12 ground truth (a hardcoded --gt default) and produced a plausible-looking
0%-identity scorecard that cost two sessions a day. This guard makes that a
loud error at the join point.
"""

from __future__ import annotations

from core.contracts.ids import SetId


class SetMismatchError(ValueError):
    """Two artifacts claiming different set_ids were combined."""


def join_guard(*set_ids: SetId | str, context: str = "") -> SetId:
    """Assert all given set_ids are equal and non-empty; return the one id.

    Call at every point where two set-scoped artifacts meet::

        sid = join_guard(timeline.set_id, gt.set_id, context="scoring")
    """
    ids = {str(s) for s in set_ids}
    if not ids or "" in ids or None in ids:
        raise SetMismatchError(
            f"join_guard{f' ({context})' if context else ''}: "
            f"missing/empty set_id among {set_ids!r}"
        )
    if len(ids) != 1:
        raise SetMismatchError(
            f"join_guard{f' ({context})' if context else ''}: "
            f"artifacts describe different sets: {sorted(ids)!r}"
        )
    return SetId(ids.pop())
