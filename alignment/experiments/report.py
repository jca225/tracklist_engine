"""Paper tables from the long store. CIs are span-level bootstrap ONLY (n=2 sets
forbids a set-level CI). Every trajectory table carries the fiber − strict gap."""

from __future__ import annotations

import numpy as np


def _clean(values) -> np.ndarray:
    return np.array([v for v in values if v is not None], dtype=float)


def mean_ci(values, *, seed: int = 0, n: int = 1000) -> tuple[float, float, float]:
    v = _clean(values)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)]
    return (
        float(v.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def paired_delta_ci(
    a, b, *, seed: int = 0, n: int = 1000
) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) − mean(b) over a shared span index."""
    av, bv = _clean(a), _clean(b)
    m = min(av.size, bv.size)
    av, bv = av[:m], bv[:m]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        boots.append(av[idx].mean() - bv[idx].mean())
    return (
        float(av.mean() - bv.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.0f}%"


def headline_table(rows: list[dict]) -> str:
    """Per set × stem: strict, fiber-aware, and the gap (the which-instance
    residual). Uses only baseline classical/looptrace rows."""
    base = [
        r
        for r in rows
        if r["driver"] == "classical" and r.get("decoder") == "looptrace"
    ]
    sets = sorted({r["set_id"] for r in base})
    stems = ("acappella", "regular", "instrumental")
    out = [
        "| set | stem | strict | fiber-aware | gap (fiber−strict) |",
        "|---|---|---|---|---|",
    ]
    for sid in sets:
        for stem in stems:
            sub = [r for r in base if r["set_id"] == sid and r["stem"] == stem]
            if not sub:
                continue
            sm, _, _ = mean_ci([r["strict"] for r in sub])
            fm, _, _ = mean_ci([r["fiber"] for r in sub])
            gap = "—" if (sm != sm or fm != fm) else f"+{100 * (fm - sm):.0f}pp"
            out.append(f"| {sid} | {stem} | {_pct(sm)} | {_pct(fm)} | {gap} |")
    return "\n".join(out)


def ablation_table(
    rows: list[dict], field: str, left, right, *, metric: str = "strict"
) -> str:
    """One-toggle ablation: mean(left) vs mean(right) on `field`, with a paired
    span-bootstrap CI on the delta."""
    a = [r[metric] for r in rows if r.get(field) == left]
    b = [r[metric] for r in rows if r.get(field) == right]
    am, _, _ = mean_ci(a)
    bm, _, _ = mean_ci(b)
    d, lo, hi = paired_delta_ci(a, b)
    return (
        f"| {field}={left} vs {right} ({metric}) | {_pct(am)} | {_pct(bm)} "
        f"| {100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
    )
