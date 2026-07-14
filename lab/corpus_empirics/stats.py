"""Shared correlation / regression helpers for corpus-empirics scripts."""
from __future__ import annotations

from typing import Sequence


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def partial_pearson(xs, ys, zs):
    """Pearson(x, y) controlling for z."""

    def resid(vs, ctrl):
        mc = sum(ctrl) / len(ctrl)
        mv = sum(vs) / len(vs)
        num = sum((c - mc) * (v - mv) for c, v in zip(ctrl, vs))
        den = sum((c - mc) ** 2 for c in ctrl)
        b = num / den if den else 0.0
        a = mv - b * mc
        return [v - (a + b * c) for c, v in zip(ctrl, vs)]

    return pearson(resid(xs, zs), resid(ys, zs))


def fit_ols(y, X):
    """OLS with intercept. Returns (coefficients, r_squared, residuals) or (None, None, None)."""
    n, k = len(y), len(X)
    cols = [[1.0] * n] + X
    XtX = [
        [sum(cols[i][r] * cols[j][r] for r in range(n)) for j in range(k + 1)]
        for i in range(k + 1)
    ]
    Xty = [sum(cols[i][r] * y[r] for r in range(n)) for i in range(k + 1)]
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    m = k + 1
    for i in range(m):
        p = max(range(i, m), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        d = A[i][i]
        if abs(d) < 1e-12:
            return None, None, None
        for j in range(i, m + 1):
            A[i][j] /= d
        for r in range(m):
            if r == i:
                continue
            f = A[r][i]
            for j in range(i, m + 1):
                A[r][j] -= f * A[i][j]
    b = [A[i][m] for i in range(m)]
    yhat = [b[0] + sum(b[j + 1] * X[j][r] for j in range(k)) for r in range(n)]
    sse = sum((y[r] - yhat[r]) ** 2 for r in range(n))
    my = sum(y) / n
    sst = sum((y[r] - my) ** 2 for r in range(n))
    return b, 1 - sse / sst, [y[r] - yhat[r] for r in range(n)]
