"""The four Bayesian arms of the manuscript, generalized to constrained multi-objective benchmark problems.

Arms (as in the paper): FA-MOBO (augmentation + in-loop classifier), Aug-BO (augmentation only),
Clf-BO (classifier only) and qNEHVI (plain constrained qLogNEHVI). Settings follow the manuscript's
settings table: SingleTaskGP per outcome, qLogNEHVI with 64 Sobol samples, q = 3 chosen sequentially,
3 restarts from 128 raw samples, and a random-forest classifier with 300 trees, minimum leaf 2 and
balanced-subsample weights, entering the acquisition as w * log(p_RF + eps) with w = 1, eps = 1e-3.

Two rules of the circuit study are circuit-specific and are replaced by generic equivalents:
  * the OP1dB-relaxed near-feasible label becomes a normalized total constraint violation below a
    predefined quantile of the seed's positive violations;
  * the efficiency (PAE) weight becomes a generic objective-quality weight: an augmented Chebyshev
    scalarization of the normalized objectives, divided by its best value among near-feasible seed points.
Both definitions are fixed across problems.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition import AcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from gpytorch.mlls import SumMarginalLogLikelihood
from scipy.stats import qmc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------------------------------------------- fixed settings (manuscript's Table 7)
Q_BATCH, MC_SAMPLES, N_RESTARTS, RAW_SAMPLES, MAXITER = 3, 64, 3, 128, 100
RF_TREES, RF_LEAF, RF_W, RF_EPS = 300, 2, 1.0, 1e-3
ENSEMBLE_SIZES, ENS_TREES, LAMBDA_RISK = (5, 10, 15), 600, 0.5
N_CANDIDATES, TOP_K_OVER = 20_000, 25
VIOLATION_QUANTILE = 0.25          # near-feasible label: total normalized violation <= this quantile of the seed's positives
CHEB_RHO = 0.05                    # augmented Chebyshev weight on the sum term


@dataclass(frozen=True)
class Budget:
    """Budget scaled by problem dimension, in the proportions of the circuit study (10d seed, 1.5d augmentation, 6.6d BO)."""

    n_seed: int
    n_aug: int
    n_bo: int

    @staticmethod
    def for_dim(d: int) -> "Budget":
        """Proportions of the circuit study (10d seed, 1.5d augmentation, 6.6d optimization), with floors so that
        low-dimensional problems still receive a meaningful number of optimization evaluations."""
        return Budget(n_seed=max(20, 10 * d), n_aug=max(5, round(1.5 * d)), n_bo=max(35, round(6.6 * d)))

    @property
    def total(self) -> int:
        return self.n_seed + self.n_aug + self.n_bo


def lhs(n: int, d: int, seed: int) -> np.ndarray:
    """Latin-hypercube sample in the unit cube."""
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def to_unit(X: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    return (X - xl) / (xu - xl)


def from_unit(U: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    return xl + U * (xu - xl)


def violation(G: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Total constraint violation, each constraint normalized by its scale."""
    return np.maximum(G / scale, 0.0).sum(axis=1)


def chebyshev_quality(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Augmented Chebyshev value of the normalized minimization objectives; larger is better."""
    Z = (F - lo) / np.maximum(hi - lo, 1e-12)
    return -(Z.max(axis=1) + CHEB_RHO * Z.sum(axis=1))


class RFWeightedLogAcq(AcquisitionFunction):
    """log-acquisition + w * log(p_RF(x) + eps), as in the manuscript's Eq. for the classifier-guided acquisition."""

    def __init__(self, base: AcquisitionFunction, rf: RandomForestClassifier, weight: float = RF_W, eps: float = RF_EPS) -> None:
        super().__init__(model=base.model)
        self.base, self.rf, self.weight, self.eps = base, rf, weight, eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        val = self.base(X)
        flat = X.detach().reshape(-1, X.shape[-1]).cpu().numpy()
        proba = self.rf.predict_proba(flat)
        cls = list(self.rf.classes_)
        p = proba[:, cls.index(1)] if 1 in cls else np.zeros(len(flat))
        p = torch.tensor(p, dtype=X.dtype, device=X.device).reshape(X.shape[:-1]).mean(dim=-1)
        return val + self.weight * torch.log(p + self.eps)

    @property
    def X_pending(self):  # noqa: N802
        return self.base.X_pending

    def set_X_pending(self, X_pending=None) -> None:  # noqa: N802
        self.base.set_X_pending(X_pending)


def augmentation_batch(U: np.ndarray, F: np.ndarray, G: np.ndarray, scale: np.ndarray, n_aug: int, seed: int,
                       use_quality: bool = True) -> np.ndarray:
    """Algorithm 1, generalized: rank Latin-hypercube candidates by risk-adjusted feasibility times objective quality."""
    rng = np.random.default_rng(seed)
    d = U.shape[1]
    ok = np.isfinite(F).all(axis=1) & np.isfinite(G).all(axis=1)
    U, F, G = U[ok], F[ok], G[ok]
    v = violation(G, scale)
    pos = v[v > 0]
    tau = float(np.quantile(pos, VIOLATION_QUANTILE)) if len(pos) else 0.0
    y = (v <= tau).astype(int)                                   # near-feasible label
    if y.sum() < 2:                                              # fall back to the most nearly feasible points
        y = np.zeros(len(v), int); y[np.argsort(v)[:max(2, len(v) // 20)]] = 1
    cand = rng.random((N_CANDIDATES, d))
    scores = {}
    for m in ENSEMBLE_SIZES:
        probs = []
        for i in range(m):
            idx = rng.integers(0, len(U), len(U))
            rf = RandomForestClassifier(n_estimators=ENS_TREES, min_samples_leaf=RF_LEAF, class_weight="balanced_subsample",
                                        random_state=1000 + i + 10 * m, n_jobs=-1)
            rf.fit(U[idx], y[idx])
            p = rf.predict_proba(cand)
            probs.append(p[:, list(rf.classes_).index(1)] if 1 in rf.classes_ else np.zeros(len(cand)))
        P = np.vstack(probs)
        scores[m] = P.mean(0) - LAMBDA_RISK * P.std(0)
    if use_quality:
        lo, hi = F.min(axis=0), F.max(axis=0)
        q_obs = chebyshev_quality(F, lo, hi)
        reg = RandomForestRegressor(n_estimators=ENS_TREES, min_samples_leaf=RF_LEAF, random_state=4242, n_jobs=-1)
        reg.fit(U, q_obs)
        q_hat = reg.predict(cand)
        near = y.astype(bool)
        q_ref = float(q_obs[near].max()) if near.any() else float(q_obs.max())
        span = float(np.ptp(q_obs)) or 1.0
        w = np.clip((q_hat - (q_ref - span)) / span, 0.0, 1.0)    # 1 at the best near-feasible quality, 0 a span below it
        scores = {m: s * w for m, s in scores.items()}
    top = {m: set(np.argsort(-s)[:TOP_K_OVER]) for m, s in scores.items()}
    pool = sorted(set().union(*top.values()))
    ranks = np.mean([np.argsort(np.argsort(-scores[m][pool])) for m in scores], axis=0)
    chosen = [pool[i] for i in np.argsort(ranks)[:n_aug]]
    return cand[chosen]


# ---------------------------------------------------------------- Bayesian loop
def _fit(U: np.ndarray, Y: np.ndarray) -> ModelListGP:
    X = torch.tensor(U, dtype=torch.double)
    models = [SingleTaskGP(X, torch.tensor(Y[:, i:i + 1], dtype=torch.double), outcome_transform=Standardize(m=1))
              for i in range(Y.shape[1])]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _ref_point(Y: np.ndarray, n_obj: int) -> torch.Tensor:
    obj = torch.tensor(Y[:, :n_obj], dtype=torch.double)
    feas = torch.tensor((Y[:, n_obj:] <= 0).all(axis=1), dtype=torch.bool)
    pool = obj[feas] if bool(feas.any()) else obj
    return infer_reference_point(pool[is_non_dominated(pool)])


def propose(U: np.ndarray, Y: np.ndarray, n_obj: int, seed: int, use_classifier: bool) -> np.ndarray:
    """One batch of q designs from constrained qLogNEHVI, optionally weighted by the in-loop classifier."""
    d = U.shape[1]
    n_constr = Y.shape[1] - n_obj
    torch.manual_seed(seed)          # the acquisition optimizer draws its restarts from the global generator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _fit(U, Y)
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=_ref_point(Y, n_obj), X_baseline=torch.tensor(U, dtype=torch.double), prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=seed),
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(n_obj))),
            constraints=[(lambda Z, k=k: Z[..., n_obj + k]) for k in range(n_constr)],
        )
        if use_classifier:
            feas = (Y[:, n_obj:] <= 0).all(axis=1).astype(int)
            rf = RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=RF_LEAF, class_weight="balanced_subsample",
                                        random_state=seed, n_jobs=4)
            rf.fit(U, feas)
            acqf = RFWeightedLogAcq(acqf, rf)
        bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()
        cand, _ = optimize_acqf(acqf, bounds=bounds, q=Q_BATCH, num_restarts=N_RESTARTS, raw_samples=RAW_SAMPLES,
                                sequential=True, options={"maxiter": MAXITER, "batch_limit": 5})
    return cand.detach().numpy()


def run_arm(problem, arm: str, seed: int) -> dict:
    """One run of one arm on one problem; returns the evaluation history in evaluation order."""
    d = problem.n_var
    b = Budget.for_dim(d)
    xl, xu = problem.xl, problem.xu
    U = lhs(b.n_seed, d, seed)
    F, G = problem.evaluate(from_unit(U, xl, xu))
    scale = np.maximum(np.abs(np.where(np.isfinite(G), G, 0.0)).max(axis=0), 1e-12)   # per-constraint normalizer from the seed
    phase = ["seed"] * b.n_seed
    use_aug = arm in ("FA-MOBO", "Aug-BO")
    use_clf = arm in ("FA-MOBO", "Clf-BO")
    n_bo = b.n_bo + (0 if use_aug else b.n_aug)                  # arms without augmentation spend that budget in the loop
    if use_aug:
        A = augmentation_batch(U, F, G, scale, b.n_aug, seed)
        Fa, Ga = problem.evaluate(from_unit(A, xl, xu))
        U, F, G = np.vstack([U, A]), np.vstack([F, Fa]), np.vstack([G, Ga])
        phase += ["aug"] * len(A)
    done = 0
    while done < n_bo:
        ok = np.isfinite(F).all(axis=1) & np.isfinite(G).all(axis=1)
        Y = np.column_stack([-F[ok], G[ok]])                      # maximize -F, constraints g <= 0
        cand = propose(U[ok], Y, problem.n_obj, seed + done, use_clf)
        take = min(len(cand), n_bo - done)
        cand = cand[:take]
        Fc, Gc = problem.evaluate(from_unit(cand, xl, xu))
        U, F, G = np.vstack([U, cand]), np.vstack([F, Fc]), np.vstack([G, Gc])
        phase += ["bo"] * take
        done += take
    valid = np.isfinite(F).all(axis=1) & np.isfinite(G).all(axis=1)
    feas = valid & (G <= 0).all(axis=1)                           # an evaluation without finite values is infeasible
    return {"problem": problem.name, "arm": arm, "seed": seed, "n_var": d,
            "budget": {"seed": b.n_seed, "aug": b.n_aug if use_aug else 0, "bo": n_bo, "total": len(F)},
            "F": F.tolist(), "G": G.tolist(), "feasible": feas.tolist(), "valid": valid.tolist(), "phase": phase}
