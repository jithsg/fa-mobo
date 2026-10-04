"""T3a: pre-registered multi-criteria metrics of each run's returned design set, and paired tests.

Run from the scripts folder. Writes T3a_per_run.csv and T3a_tests.csv to outputs/ (a copy of T3a_per_run.csv is in data/precomputed).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ARMS, DESIGN_VARS, SEEDS, feasible_mask, load_hist  # noqa: E402

T3 = HERE.parent / "outputs"
K_CAP, K_REF, PSAT_REF, OP_MIN, OP_MARGIN = 20.0, 5.0, 18.0, 15.5, 15.8
K_LEVELS = (10, 12, 15, 20)
EPS = 0.05
MAIN = "fa15qc"
OTHERS = ["fa15q", "qnc", "qnm", "qn"]
PRIMARY = ["P1_hv3_K", "P2_hv3_slack", "P3_mhv_K12", "P4_frontier_mean", "P5_eps"]
LOWER_BETTER = {"P5_eps"}
VIEWED = {f"S{i}" for i in range(12, 22)}


# ----------------------------------------------------------------------------- geometry
def nondominated(P: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (maximisation)."""
    n = len(P)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)
        keep[i] = not dom.any()
    return keep


def hv2(P: np.ndarray, ref: tuple[float, float]) -> float:
    Q = P - np.asarray(ref)
    Q = Q[np.all(Q > 0, axis=1)]
    if len(Q) == 0:
        return 0.0
    Q = Q[nondominated(Q)]
    Q = Q[np.argsort(-Q[:, 0])]
    xs_next = np.append(Q[1:, 0], 0.0)
    return float(np.sum(Q[:, 1] * (Q[:, 0] - xs_next)))


def hv3(P: np.ndarray, ref: tuple[float, float, float]) -> float:
    """Exact 3-D HV by slicing along the third objective."""
    Q = P - np.asarray(ref)
    Q = Q[np.all(Q > 0, axis=1)]
    if len(Q) == 0:
        return 0.0
    zs = np.sort(np.unique(Q[:, 2]))[::-1]
    total = 0.0
    for i, z in enumerate(zs):
        z_next = zs[i + 1] if i + 1 < len(zs) else 0.0
        total += (z - z_next) * hv2(Q[Q[:, 2] >= z, :2], (0.0, 0.0))
    return total


def eps_to_front(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per reference point r: min over a of max_j (r_j - a_j) (additive, maximisation)."""
    return np.array([np.min(np.max(r - A, axis=1)) for r in R])


# ----------------------------------------------------------------------------- per-run metrics
def feasible_set(arm: str, seed: int, budget: int = 181) -> pd.DataFrame:
    h = load_hist(arm, seed).iloc[:budget].drop_duplicates(subset=DESIGN_VARS)
    f = h[feasible_mask(h)].copy()
    f["Kc"] = f["minKf"].clip(upper=K_CAP)
    f["slack"] = f["OP1dB_dBm"] - OP_MIN
    return f


def run_metrics(f: pd.DataFrame, front_n: np.ndarray, pae_lo: float, pae_hi: float) -> dict:
    pae, psat, K = f["PAE_percent"].to_numpy(), f["Psat_dBm"].to_numpy(), f["minKf"].to_numpy()
    m = {
        "P1_hv3_K": hv3(np.c_[pae, psat, f["Kc"]], (0.0, PSAT_REF, K_REF)),
        "P2_hv3_slack": hv3(np.c_[pae, psat, f["slack"]], (0.0, PSAT_REF, 0.0)),
        "S1_hv2_pae_K": hv2(np.c_[pae, f["Kc"]], (0.0, K_REF)),
    }
    best = {k: float(pae[K >= k].max()) if (K >= k).any() else 0.0 for k in K_LEVELS}
    m["P4_frontier_mean"] = float(np.mean(list(best.values())))
    for i, k in enumerate(K_LEVELS):
        m[f"S{5 + i}_bestpae_K{k}"] = best[k]
    marg = f["OP1dB_dBm"].to_numpy() >= OP_MARGIN
    for k in K_LEVELS:
        sel = marg & (K >= k)
        v = hv2(np.c_[pae[sel], psat[sel]], (0.0, PSAT_REF))
        m["P3_mhv_K12" if k == 12 else f"S{2 + [10, 15, 20].index(k)}_mhv_K{k}"] = v
    m["S9_nd_pae_K"] = int(nondominated(np.c_[pae, K]).sum())
    m["S10_nd_pae_psat_K"] = int(nondominated(np.c_[pae, psat, K]).sum())
    A = np.c_[(pae - pae_lo) / (pae_hi - pae_lo), (f["Kc"].to_numpy() - K_REF) / (K_CAP - K_REF)]
    e = eps_to_front(A, front_n)
    m["P5_eps"], m["S11_coverage"] = float(e.max()), float(np.mean(e <= EPS))
    for i, (p, k) in enumerate(itertools.product((30, 35), (10, 12, 15))):
        m[f"S{12 + i}_n_pae{p}_K{k}"] = int(((pae >= p) & (K >= k)).sum())
    for i, k in enumerate(K_LEVELS):
        m[f"S{18 + i}_n_op158_K{k}"] = int((marg & (K >= k)).sum())
    return m


def pooled_front(sets: dict) -> tuple[np.ndarray, float, float]:
    allf = pd.concat(sets.values())
    lo, hi = allf["PAE_percent"].min(), allf["PAE_percent"].max()
    P = np.c_[(allf["PAE_percent"] - lo) / (hi - lo), (allf["Kc"] - K_REF) / (K_CAP - K_REF)]
    P = np.unique(P, axis=0)
    return P[nondominated(P)], lo, hi


def per_run_table(budget: int) -> pd.DataFrame:
    sets = {(a, s): feasible_set(a, s, budget) for a in ARMS for s in SEEDS}
    front, lo, hi = pooled_front(sets)
    print(f"budget {budget}: pooled front {len(front)} pts, PAE range {lo:.2f}-{hi:.2f}")
    rows = [{"arm": a, "seed": s, "budget": budget, "n_feas": len(f), **run_metrics(f, front, lo, hi)}
            for (a, s), f in sets.items()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def exact_wilcoxon(d: np.ndarray) -> float:
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    obs = ranks[d > 0].sum()
    mu = ranks.sum() / 2
    signs = np.array(list(itertools.product((0, 1), repeat=n)), dtype=float)
    stats = signs @ ranks
    return float(np.mean(np.abs(stats - mu) >= abs(obs - mu) - 1e-9))


def boot_ci(d: np.ndarray, rng: np.random.Generator, B: int = 10000) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def tests(tab: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260914)
    metrics = [c for c in tab.columns if c[:1] in "PS" and c[1].isdigit()]
    rows = []
    for met in metrics:
        piv = tab.pivot(index="seed", columns="arm", values=met).loc[SEEDS]
        sign = -1.0 if met in LOWER_BETTER else 1.0
        for other in OTHERS:
            d = sign * (piv[MAIN] - piv[other]).to_numpy(dtype=float)
            lo, hi = boot_ci(d, rng)
            rows.append({"metric": met, "family": "primary" if met in PRIMARY else "secondary",
                         "viewed": met.split("_")[0] in VIEWED, "vs": ARMS[other][0],
                         "fa_mean": piv[MAIN].mean(), "other_mean": piv[other].mean(),
                         "diff_fa_better": d.mean(), "ci_lo": lo, "ci_hi": hi,
                         "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                         "p": exact_wilcoxon(d)})
    out = pd.DataFrame(rows)
    prim = out["family"] == "primary"
    out.loc[prim, "p_holm_primary"] = holm(out.loc[prim, "p"].to_numpy())
    out["p_holm_all"] = holm(out["p"].to_numpy())
    return out


def sanity_checks() -> None:
    rng = np.random.default_rng(0)
    P = rng.random((12, 3))
    mc = rng.random((400000, 3))
    dom = np.zeros(len(mc), bool)
    for p in P:
        dom |= np.all(mc <= p, axis=1)
    print(f"hv3 check: exact {hv3(P, (0, 0, 0)):.4f} vs MC {dom.mean():.4f}")
    from base import dominated_hypervolume  # noqa: E402
    h = load_hist("fa15qc", 101); f = h[feasible_mask(h)]
    a = dominated_hypervolume(f["PAE_percent"].to_numpy(), f["Psat_dBm"].to_numpy())
    b = hv2(f[["PAE_percent", "Psat_dBm"]].to_numpy(), (0, 0))
    print(f"hv2 check vs base: {a:.4f} vs {b:.4f}")
    print(f"wilcoxon floor n=10: {exact_wilcoxon(np.arange(1, 11, dtype=float)):.5f}")


def main() -> None:
    sanity_checks()
    tab = pd.concat([per_run_table(181), per_run_table(141)], ignore_index=True)
    tab.to_csv(T3 / "T3a_per_run.csv", index=False)
    res = tests(tab[tab["budget"] == 181])
    res.to_csv(T3 / "T3a_tests.csv", index=False)
    sens = tests(tab[tab["budget"] == 141])
    sens[sens["family"] == "primary"].to_csv(T3 / "T3a_tests_141.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    cols = ["metric", "vs", "fa_mean", "other_mean", "diff_fa_better", "ci_lo", "ci_hi", "wins", "ties", "p", "p_holm_primary", "p_holm_all"]
    print(res[cols].round(4).to_string(index=False))
    print("\n--- budget 141, primary ---")
    print(sens.loc[sens["family"] == "primary", cols[:-2]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
