"""Faithful implementation of the paper's Algorithm 1 (feasibility augmentation).

Reproduces the exact pipeline used to produce the 45 augmentation candidates:
near-feasible labelling (OP1dB relaxed by 1 dB), three bootstrap RF ensembles
(sizes 5/10/15), risk-adjusted scoring of a 20k LHS candidate pool
(score = p_mean - lambda * p_std), over-selection of the top-25 per ensemble,
union pooling, mean-rank aggregation, and diversity-capped greedy selection
(max 18 per Lx id / Ldc3 id, final N = 45).

Constants match the original run in Liu-5_10_15.ipynb; every knob is exposed
as a parameter so the sensitivity and stability studies can sweep them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUM_COLS = ["M3_width", "CB", "Cs", "Cm"]
CAT_COLS = ["Lx_id", "Ldc3_id"]
KEY_COLS = CAT_COLS + NUM_COLS


@dataclass(frozen=True)
class Algo1Config:
    """All Algorithm 1 hyperparameters (paper defaults)."""

    relax_op1db_db: float = 1.0     # near-feasible OP1dB relaxation (dB)
    lambda_risk: float = 0.5        # risk-adjusted score penalty
    ensemble_sizes: tuple[int, ...] = (5, 10, 15)
    n_candidates: int = 20000       # LHS candidate pool size
    top_k_over: int = 25            # over-selection per ensemble
    max_per_lx: int = 18            # diversity cap per Lx id
    max_per_ldc3: int = 18          # diversity cap per Ldc3 id
    n_select: int = 45              # final shortlist size
    n_estimators: int = 600
    min_samples_leaf: int = 2
    # Tweak 2 (2026-09-13): weight each candidate's risk-adjusted feasibility
    # score by its predicted efficiency, q(x)^pae_weight with q = clip(PAE_hat /
    # PAE_ref, 0, 1) from a random-forest PAE regressor fitted on the LHS set
    # (PAE_ref = best near-feasible LHS PAE). 0 disables it (paper behaviour).
    pae_weight: float = 0.0


PAPER = Algo1Config()


def near_feasible_label(df: pd.DataFrame, relax_op1db_db: float) -> pd.Series:
    """Near-feasible training label: hard constraints with OP1dB relaxed."""
    return (
        (df["OP1dB_dBm"] >= 15.5 - relax_op1db_db)
        & (df["Gain_dB"] >= 20.0)
        & (df["minKf"] >= 5.0)
        & (df["VDS_peak_M3_V"] < 8.7)
        & (df["PDC_total_mW"] <= 350.0)
        & (df["HB_converged_flag"] == 1)
        & (df["is_valid"] == 1)
    ).astype(int)


def strict_feasible(df: pd.DataFrame) -> pd.Series:
    """Strict feasibility (paper Eq. 24) used for all acceptance decisions."""
    return near_feasible_label(df, relax_op1db_db=0.0)


def _make_pipe(cfg: Algo1Config, rs: int) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ])),
        ("rf", RandomForestClassifier(
            n_estimators=cfg.n_estimators, min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced_subsample", random_state=rs, n_jobs=-1)),
    ])


def _proba_feasible(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """P(feasible) that tolerates single-class bootstrap resamples."""
    proba = pipe.predict_proba(X)
    classes = pipe.named_steps["rf"].classes_
    if proba.shape[1] == 1:
        return np.full(len(X), float(classes[0]))
    return proba[:, list(classes).index(1)]


def _lhs_candidates(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """LHS pool for continuous vars within observed bounds + uniform discrete ids."""
    cont: dict[str, np.ndarray] = {}
    for k in NUM_COLS:
        lo, hi = float(df[k].min()), float(df[k].max())
        edges = np.linspace(0.0, 1.0, n + 1)
        u = rng.random(n) * (edges[1:] - edges[:-1]) + edges[:-1]
        rng.shuffle(u)
        cont[k] = lo + u * (hi - lo)
    return pd.DataFrame({
        "Lx_id": rng.choice(sorted(df["Lx_id"].dropna().unique()), n),
        "Ldc3_id": rng.choice(sorted(df["Ldc3_id"].dropna().unique()), n),
        **cont,
    })


def run_algo1(
    lhs_df: pd.DataFrame,
    cfg: Algo1Config = PAPER,
    seed: int = 7,
) -> pd.DataFrame:
    """Run Algorithm 1 and return the selected candidate designs.

    Returns a DataFrame of ``cfg.n_select`` rows with KEY_COLS plus per-ensemble
    scores and the aggregated mean rank.
    """
    rng = np.random.default_rng(seed)
    y = near_feasible_label(lhs_df, cfg.relax_op1db_db)
    X = lhs_df[NUM_COLS + CAT_COLS]

    cand = _lhs_candidates(lhs_df, cfg.n_candidates, rng)
    scores: dict[int, np.ndarray] = {}
    for n_models in cfg.ensemble_sizes:
        probs = []
        for i in range(n_models):
            idx = rng.integers(0, len(X), len(X))
            pipe = _make_pipe(cfg, rs=1000 + i + n_models * 10)
            pipe.fit(X.iloc[idx], y.iloc[idx])
            probs.append(_proba_feasible(pipe, cand[NUM_COLS + CAT_COLS]))
        P = np.vstack(probs)
        scores[n_models] = P.mean(0) - cfg.lambda_risk * P.std(0)

    scored = cand.copy()
    if cfg.pae_weight > 0:
        from sklearn.ensemble import RandomForestRegressor
        ok = lhs_df["HB_converged_flag"] >= 0.5
        reg = Pipeline([
            ("prep", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                ("num", "passthrough", NUM_COLS),
            ])),
            ("rf", RandomForestRegressor(n_estimators=cfg.n_estimators,
                                         min_samples_leaf=cfg.min_samples_leaf,
                                         random_state=4242, n_jobs=-1)),
        ])
        reg.fit(X[ok], lhs_df.loc[ok, "PAE_percent"])
        pae_hat = reg.predict(cand[NUM_COLS + CAT_COLS])
        near = y.astype(bool) & ok
        pae_ref = float(lhs_df.loc[near, "PAE_percent"].max()) if near.any() \
            else float(lhs_df.loc[ok, "PAE_percent"].max())
        q = np.clip(pae_hat / max(pae_ref, 1e-6), 0.0, 1.0) ** cfg.pae_weight
        scores = {k: s * q for k, s in scores.items()}
        scored["pae_hat"] = pae_hat
    for n_models, s in scores.items():
        scored[f"score_{n_models}"] = s

    # Over-select top-K per ensemble, union pool, mean-rank aggregation.
    top_idx: dict[int, set[int]] = {
        k: set(np.argsort(-scores[k])[: cfg.top_k_over]) for k in scores
    }
    union = sorted(set().union(*top_idx.values()))
    pool = scored.iloc[union].copy()
    ranks = np.mean(
        [pd.Series(-scores[k][union]).rank(method="average").to_numpy() for k in scores],
        axis=0,
    )
    pool["mean_rank"] = ranks
    pool = pool.sort_values("mean_rank")

    # Diversity-capped greedy selection.
    lx_count: dict[str, int] = {}
    ldc3_count: dict[str, int] = {}
    chosen: list[int] = []
    for i, row in pool.iterrows():
        if len(chosen) == cfg.n_select:
            break
        lx, ldc3 = row["Lx_id"], row["Ldc3_id"]
        if lx_count.get(lx, 0) >= cfg.max_per_lx or ldc3_count.get(ldc3, 0) >= cfg.max_per_ldc3:
            continue
        chosen.append(i)
        lx_count[lx] = lx_count.get(lx, 0) + 1
        ldc3_count[ldc3] = ldc3_count.get(ldc3, 0) + 1
    if len(chosen) < cfg.n_select:  # fill remaining in rank order (no cap)
        for i in pool.index:
            if len(chosen) == cfg.n_select:
                break
            if i not in chosen:
                chosen.append(i)
    return pool.loc[chosen].reset_index(drop=True)


def train_verifier(lhs_df: pd.DataFrame, cfg: Algo1Config = PAPER, seed: int = 99) -> Pipeline:
    """Independent verifier RF trained on the full dataset (no bootstrap)."""
    pipe = _make_pipe(cfg, rs=seed)
    pipe.fit(lhs_df[NUM_COLS + CAT_COLS], near_feasible_label(lhs_df, cfg.relax_op1db_db))
    return pipe


def verifier_pfeas(verifier: Pipeline, designs: pd.DataFrame) -> np.ndarray:
    """Verifier-predicted near-feasibility probability for candidate designs."""
    return _proba_feasible(verifier, designs[NUM_COLS + CAT_COLS])


def with_params(cfg: Algo1Config = PAPER, **kw) -> Algo1Config:
    """Immutable config override helper."""
    return replace(cfg, **kw)
