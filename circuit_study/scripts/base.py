"""Two-dimensional dominated hypervolume used by the analysis scripts (excerpt of the optimization code)."""

from __future__ import annotations

import numpy as np

from problem import HV_REF_POINT


def pareto_mask(pae: np.ndarray, psat: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points for 2-D maximization."""
    n = len(pae)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        # i is dominated if some j is >= in both objectives and > in one.
        dominated = (
            (pae >= pae[i]) & (psat >= psat[i]) & ((pae > pae[i]) | (psat > psat[i]))
        )
        if dominated.any():
            nd[i] = False
    return nd


def dominated_hypervolume(
    pae: np.ndarray,
    psat: np.ndarray,
    ref: tuple[float, float] = HV_REF_POINT,
) -> float:
    """2-D dominated hypervolume for maximization w.r.t. reference ``ref``.

    Args:
        pae: First-objective values (PAE %).
        psat: Second-objective values (Psat dBm).
        ref: Reference point (dominated by all counted points).

    Returns:
        Dominated area; 0.0 if no point strictly dominates ``ref``.
    """
    x = np.asarray(pae, dtype=float) - ref[0]
    y = np.asarray(psat, dtype=float) - ref[1]
    keep = (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if x.size == 0:
        return 0.0

    nd = pareto_mask(x, y)
    x, y = x[nd], y[nd]

    order = np.argsort(-x)  # x descending
    xs, ys = x[order], y[order]
    xs_next = np.append(xs[1:], 0.0)
    return float(np.sum(ys * (xs - xs_next)))
