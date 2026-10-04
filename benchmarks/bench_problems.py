"""Constrained multi-objective benchmark problems, implemented without pymoo.

The worker machines have numpy, scipy, scikit-learn and BoTorch but not pymoo, so the six problems used by the
benchmark study are implemented here directly. Each implementation mirrors the pymoo 0.6.2 definition and is
checked against it by ``validate_against_pymoo.py`` on a machine that has pymoo.

Convention: objectives are minimized and constraints are satisfied when g(x) <= 0, as in pymoo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------- MW helpers (pymoo MW base class)
def _la1(A: float, B: float, C: float, D: float, theta: np.ndarray) -> np.ndarray:
    return A * np.power(np.sin(B * np.pi * np.power(theta, C)), D)


def _la2(A: float, B: float, C: float, D: float, theta: np.ndarray) -> np.ndarray:
    return A * np.power(np.sin(B * np.power(theta, C)), D)


def _g1(X: np.ndarray, n_obj: int = 2) -> np.ndarray:
    d = X.shape[1]
    n = d - n_obj
    z = np.power(X[:, n_obj - 1:], n)
    i = np.arange(n_obj - 1, d)
    exp = 1 - np.exp(-10.0 * (z - 0.5 - i / (2 * d)) * (z - 0.5 - i / (2 * d)))
    return 1 + exp.sum(axis=1)


def _g2(X: np.ndarray, n_obj: int = 2) -> np.ndarray:
    d = X.shape[1]
    i = np.arange(n_obj - 1, d)
    z = 1 - np.exp(-10.0 * (X[:, n_obj - 1:] - i / d) * (X[:, n_obj - 1:] - i / d))
    contrib = (0.1 / d) * z * z + 1.5 - 1.5 * np.cos(2 * np.pi * z)
    return 1 + contrib.sum(axis=1)


def _g3(X: np.ndarray, n_obj: int = 2) -> np.ndarray:
    contrib = 2.0 * np.power(X[:, n_obj - 1:] + (X[:, n_obj - 2:-1] - 0.5) * (X[:, n_obj - 2:-1] - 0.5) - 1.0, 2.0)
    return 1 + contrib.sum(axis=1)


# --------------------------------------------------------------------------- problems
def _mw1(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = _g1(X)
    f0 = X[:, 0]
    f1 = g * (1 - 0.85 * f0 / g)
    g0 = f0 + f1 - 1 - _la1(0.5, 2.0, 1.0, 8.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
    return np.column_stack([f0, f1]), g0.reshape(-1, 1)


def _mw2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = _g2(X)
    f0 = X[:, 0]
    f1 = g * (1 - f0 / g)
    g0 = f0 + f1 - 1 - _la1(0.5, 3.0, 1.0, 8.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
    return np.column_stack([f0, f1]), g0.reshape(-1, 1)


def _mw3(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = _g3(X)
    f0 = X[:, 0]
    f1 = g * (1 - f0 / g)
    t = np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0
    g0 = f0 + f1 - 1.05 - _la1(0.45, 0.75, 1.0, 6.0, t)
    g1 = 0.85 - f0 - f1 + _la1(0.3, 0.75, 1.0, 2.0, t)
    return np.column_stack([f0, f1]), np.column_stack([g0, g1])


def _mw7(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = _g3(X)
    f0 = g * X[:, 0]
    f1 = g * np.sqrt(1 - np.power(f0 / g, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        atan = np.arctan(f1 / f0)
    g0 = f0 ** 2 + f1 ** 2 - np.power(1.2 + np.abs(_la2(0.4, 4.0, 1.0, 16.0, atan)), 2.0)
    g1 = np.power(1.15 - _la2(0.2, 4.0, 1.0, 8.0, atan), 2.0) - f0 ** 2 - f1 ** 2
    return np.column_stack([f0, f1]), np.column_stack([g0, g1])


def _mw11(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = _g3(X)
    f0 = g * X[:, 0]
    f1 = g * np.sqrt(2.0 - np.power(f0 / g, 2.0))
    g0 = -1.0 * (3.0 - f0 * f0 - f1) * (3.0 - 2.0 * f0 * f0 - f1)
    g1 = (3.0 - 0.625 * f0 * f0 - f1) * (3.0 - 7.0 * f0 * f0 - f1)
    g2 = -1.0 * (1.62 - 0.18 * f0 * f0 - f1) * (1.125 - 0.125 * f0 * f0 - f1)
    g3 = (2.07 - 0.23 * f0 * f0 - f1) * (0.63 - 0.07 * f0 * f0 - f1)
    return np.column_stack([f0, f1]), np.column_stack([g0, g1, g2, g3])


def _tnk(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0, f1 = X[:, 0], X[:, 1]
    g0 = -(np.square(X[:, 0]) + np.square(X[:, 1]) - 1.0 - 0.1 * np.cos(16.0 * np.arctan(X[:, 0] / X[:, 1])))
    g1 = 2 * (np.square(X[:, 0] - 0.5) + np.square(X[:, 1] - 0.5)) - 1
    return np.column_stack([f0, f1]), np.column_stack([g0, g1])


def _osy(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f0 = -(25 * (X[:, 0] - 2) ** 2 + (X[:, 1] - 2) ** 2 + (X[:, 2] - 1) ** 2 + (X[:, 3] - 4) ** 2 + (X[:, 4] - 1) ** 2)
    f1 = np.sum(np.square(X), axis=1)
    g = np.column_stack([
        (X[:, 0] + X[:, 1] - 2.0) / 2.0,
        (6.0 - X[:, 0] - X[:, 1]) / 6.0,
        (2.0 - X[:, 1] + X[:, 0]) / 2.0,
        (2.0 - X[:, 0] + 3.0 * X[:, 1]) / 2.0,
        (4.0 - (X[:, 2] - 3.0) ** 2 - X[:, 3]) / 4.0,
        ((X[:, 4] - 3.0) ** 2 + X[:, 5] - 4.0) / 4.0,
    ])
    return np.column_stack([f0, f1]), -g


@dataclass(frozen=True)
class Problem:
    """A constrained multi-objective problem: minimize F, feasible where every G <= 0."""

    name: str
    pymoo_name: str
    fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    xl: np.ndarray
    xu: np.ndarray
    n_obj: int
    n_constr: int
    hv_ref: tuple[float, ...]          # reference point for hypervolume, fixed in advance

    @property
    def n_var(self) -> int:
        return len(self.xl)

    def evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        return self.fn(X)


def _unit(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(n), np.ones(n)


# Dimensions chosen so that the feasible fraction of a uniform sample is sparse but reachable (0.7-5%),
# the regime of the circuit study (2%); measured on 200,000 samples per problem.
# --------------------------------------------------------------------------- C-DTLZ (pymoo many/cdtlz.py)
def _cdtlz_obj(X: np.ndarray, n_obj: int, alpha: float) -> np.ndarray:
    """DTLZ2/DTLZ4 objectives; alpha=1 gives DTLZ2, alpha=100 the biased DTLZ4."""
    X_, X_M = X[:, : n_obj - 1], X[:, n_obj - 1:]
    g = np.sum(np.square(X_M - 0.5), axis=1)
    f = []
    for i in range(n_obj):
        v = 1.0 + g
        v = v * np.prod(np.cos(np.power(X_[:, : X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
        if i > 0:
            v = v * np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)
        f.append(v)
    return np.column_stack(f)


def _constraint_c2(F: np.ndarray, r: float) -> np.ndarray:
    n_obj = F.shape[1]
    v1 = np.full(F.shape[0], np.inf)
    for i in range(n_obj):
        t = (F[:, i] - 1.0) ** 2 + (np.sum(F ** 2, axis=1) - F[:, i] ** 2) - r ** 2
        v1 = np.minimum(t, v1)
    a = 1.0 / np.sqrt(n_obj)
    v2 = np.sum((F - a) ** 2, axis=1) - r ** 2
    return np.minimum(v1, v2)


def _constraint_c3_spherical(F: np.ndarray) -> np.ndarray:
    n_obj = F.shape[1]
    return np.column_stack([1.0 - F[:, i] ** 2 / 4.0 - (np.sum(F ** 2, axis=1) - F[:, i] ** 2)
                            for i in range(n_obj)])


def _c2dtlz2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = _cdtlz_obj(X, 2, 1.0)
    return F, _constraint_c2(F, 0.2).reshape(-1, 1)


def _c3dtlz4(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = _cdtlz_obj(X, 2, 100.0)
    return F, _constraint_c3_spherical(F)


PROBLEMS: dict[str, Problem] = {
    "MW2": Problem("MW2", "mw2", _mw2, *_unit(5), 2, 1, (1.2, 1.2)),
    "MW3": Problem("MW3", "mw3", _mw3, *_unit(5), 2, 2, (1.2, 1.2)),
    "MW7": Problem("MW7", "mw7", _mw7, *_unit(5), 2, 2, (1.5, 1.5)),
    "MW11": Problem("MW11", "mw11", _mw11, np.zeros(5), np.full(5, np.sqrt(2.0)), 2, 4, (2.2, 2.2)),
    "TNK": Problem("TNK", "tnk", _tnk, np.array([0.0, 1e-30]), np.array([np.pi, np.pi]), 2, 2, (1.2, 1.2)),
    "OSY": Problem("OSY", "osy", _osy, np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]), np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0]), 2, 6, (0.0, 386.0)),
    "C2DTLZ2": Problem("C2DTLZ2", "c2dtlz2", _c2dtlz2, *_unit(7), 2, 1, (1.4, 1.4)),
    "C3DTLZ4": Problem("C3DTLZ4", "c3dtlz4", _c3dtlz4, *_unit(7), 2, 2, (2.6, 2.4)),
}
