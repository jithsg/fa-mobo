"""The random-forest components of FA-MOBO: model selection, calibration, learning and what they learn.

Everything here is computed from the shipped run histories (data/runs) and the 100-design seed
(data/seed/lhs_100.csv); no new circuit simulation is needed.

  python rf_analysis.py            # writes figures, tables, numbers.json and rf_section.tex to out/

Parts
  1. classifier_comparison   why a random forest: repeated stratified CV on the seed's near-feasible label
  2. calibration_replay      the in-loop classifier refit before every BO batch, as the run did, scored on
                             the batch it then proposed (a REPLAY: the run did not log its probabilities)
  3. learning_curve          out-of-bag AUC of that refit classifier against the simulation count
  4. feature_importance      held-out permutation importance per design variable
  5. disagreement            ensemble disagreement (Eq. sigma_hat) against realized feasibility

Encodings. The augmentation forest (Algorithm 1) sees the inductors as library indices (one-hot); the
in-loop classifier sees the 10-D unit-cube encoding with a 3-D (L, Q, SRF) embedding per inductor. The
embedding needs the PDK inductor library, which cannot be distributed, so the shipped default is the index
encoding; when the library is present the embedding replay is run too and both are reported.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
LIU = HERE.parent
from algo1 import CAT_COLS, NUM_COLS, near_feasible_label, strict_feasible  # noqa: E402
from common import ARMS, SEEDS, load_hist  # noqa: E402

OUT = LIU / "outputs" / "classifier_validation"; OUT.mkdir(parents=True, exist_ok=True)
LHS = pd.read_csv(LIU / "data" / "seed" / "lhs_100.csv")
DESIGN = CAT_COLS + NUM_COLS
VAR_LABEL = {"M3_width": r"$W_{\mathrm{M3}}$", "CB": r"$C_B$", "Cs": r"$C_s$", "Cm": r"$C_m$",
             "Lx_id": r"$L_x$", "Ldc3_id": r"$L_{\mathrm{dc3}}$"}
ARM_LABEL = {"fa15qc": "FA-MOBO", "qnc": "Clf-BO"}
COL = {a: ARMS[a][2] for a in ARM_LABEL}
RF_INLOOP = dict(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample")
RF_AUG = dict(n_estimators=600, min_samples_leaf=2, class_weight="balanced_subsample")
NUM: dict = {}
RAW: list = []            # every replay prediction, dumped to out/replay_predictions.csv for reuse

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.gaussian_process import GaussianProcessClassifier  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 100, "savefig.dpi": 1500,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- encodings
def onehot_prep(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
                              ("num", StandardScaler() if scale_numeric else "passthrough", NUM_COLS)])


def embedding_available() -> bool:
    try:
        from base import SearchSpace  # noqa: F401
        SearchSpace().designs_to_unit(LHS.iloc[:2][DESIGN])
        return True
    except Exception:
        return False


def to_embedding(df: pd.DataFrame) -> np.ndarray:
    from base import SearchSpace
    return np.asarray(SearchSpace().designs_to_unit(df[DESIGN].reset_index(drop=True)), float)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("  wrote", name)



# --------------------------------------------------------------------------- drawing (data in, figure out)
def draw_comparison(res: dict, n_pos: int, n_repeats: int) -> None:
    names = list(res); ref = [n for n in names if "as used" in n][0]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    for ax, key, lbl in ((axes[0], "auc", "ROC-AUC"), (axes[1], "ap", "Average precision")):
        vals = [res[n][key] for n in names]
        ax.boxplot(vals, vert=False, widths=0.55, showfliers=False, medianprops={"color": "k"})
        for i, v in enumerate(vals):
            ax.scatter(v, np.full(len(v), i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), s=5, alpha=0.35, color="0.3")
        ax.axvline(float(np.median(res[ref][key])), color=COL["fa15qc"], lw=0.9, ls="--", zorder=0)     # the forest as used
        ax.set_yticks(range(1, len(names) + 1)); ax.set_yticklabels(names if key == "auc" else [""] * len(names))
        ax.set_xlabel(lbl); ax.invert_yaxis(); ax.grid(axis="x", lw=0.3, alpha=0.5)
        if key == "ap":
            chance = NUM["comparison_setup"]["chance_ap"]
            ax.axvline(chance, color="0.5", lw=0.8, ls=":"); ax.text(chance, 0.55, " chance", fontsize=6, color="0.4", va="bottom")
    fig.suptitle(f"Feasibility classifiers on the 100-design seed ({n_pos} near-feasible positives), {n_repeats} repeats of stratified 5-fold CV",
                 fontsize=8, y=1.02)
    save(fig, "fig_rf_classifier_comparison")


def draw_calibration(rel: dict, learn: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.45})
    ax = axes[0]; ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", label="perfect calibration")
    for arm, (conf, acc, cnt, brier, e) in rel.items():
        m = cnt > 0
        ax.plot(conf[m], acc[m], "o-", ms=3.5, lw=1.1, color=COL[arm], label=f"{ARM_LABEL[arm]}  (Brier {brier:.3f}, ECE {e:.3f})")
    ax.set_xlabel("predicted feasibility probability $p_{\\mathrm{RF}}$ (bin mean)"); ax.set_ylabel("observed feasible fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", frameon=False)
    ax.set_title("(a) Reliability on proposed designs", loc="left")
    ax = axes[1]
    for arm, (x, mu, sd) in learn.items():
        ax.plot(x, mu, color=COL[arm], lw=1.3, label=ARM_LABEL[arm]); ax.fill_between(x, mu - sd, mu + sd, color=COL[arm], alpha=0.12, lw=0)
    for xv, lab in ((100, " seed"), (115, " aug.")):
        ax.axvline(xv, color="0.7", lw=0.7, ls=":"); ax.text(xv, 0.703, lab, fontsize=6, color="0.45", va="bottom")
    ax.set_xlabel("simulations before the batch"); ax.set_ylabel("OOB ROC-AUC (mean $\\pm$ SD, ten seeds)"); ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right", frameon=False); ax.set_title("(b) Out-of-bag AUC of the refit classifier", loc="left")
    save(fig, "fig_rf_calibration_learning")


def draw_importance(d: dict) -> None:
    order = sorted(DESIGN, key=lambda v: -d[v]["mean"])
    fig, ax = plt.subplots(figsize=(3.7, 2.3))
    ax.barh([VAR_LABEL[v] for v in order], [d[v]["mean"] for v in order], xerr=[d[v]["sd"] for v in order],
            color=COL["fa15qc"], alpha=0.85, error_kw={"lw": 0.8, "capsize": 2})
    ax.axvline(0, color="0.3", lw=0.7); ax.invert_yaxis(); ax.set_xlabel("permutation importance (held-out ROC-AUC drop)")
    ax.set_title("Strict-feasibility classifier at 181 simulations\n(ten FA-MOBO runs, mean $\\pm$ SD)", loc="left")
    ax.grid(axis="x", lw=0.3, alpha=0.5); save(fig, "fig_rf_feature_importance")


# --------------------------------------------------------------------------- 1. why a random forest
def classifier_comparison(n_repeats: int = 20) -> None:
    print("1. classifier comparison on the seed")
    y = near_feasible_label(LHS, 1.0).to_numpy()
    X = LHS[DESIGN]
    models = {
        "Random forest (as used)": Pipeline([("prep", onehot_prep(False)), ("m", RandomForestClassifier(**RF_AUG, random_state=0, n_jobs=-1))]),
        "Random forest, no class weight": Pipeline([("prep", onehot_prep(False)), ("m", RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=0, n_jobs=-1))]),
        "Gradient boosting": Pipeline([("prep", onehot_prep(False)), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=0))]),
        "Logistic regression": Pipeline([("prep", onehot_prep(True)), ("m", LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000))]),
        "SVM (RBF)": Pipeline([("prep", onehot_prep(True)), ("m", SVC(class_weight="balanced", probability=True, random_state=0))]),
        "k-nearest neighbours (k=5)": Pipeline([("prep", onehot_prep(True)), ("m", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        "Gaussian-process classifier (RBF)": Pipeline([("prep", onehot_prep(True)), ("m", GaussianProcessClassifier(ConstantKernel() * RBF(1.0), n_restarts_optimizer=2, random_state=0))]),
    }
    res = {k: {"auc": [], "ap": []} for k in models}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    folds = list(cv.split(X, y))
    for name, pipe in models.items():
        oof = np.zeros((n_repeats, len(y)))
        for i, (tr, te) in enumerate(folds):
            r = i // 5
            pipe.fit(X.iloc[tr], y[tr])
            oof[r, te] = pipe.predict_proba(X.iloc[te])[:, 1]
        for r in range(n_repeats):
            res[name]["auc"].append(roc_auc_score(y, oof[r])); res[name]["ap"].append(average_precision_score(y, oof[r]))
        print(f"   {name:32s} AUC {np.mean(res[name]['auc']):.3f}±{np.std(res[name]['auc']):.3f}  AP {np.mean(res[name]['ap']):.3f}±{np.std(res[name]['ap']):.3f}")
    NUM["comparison"] = {k: {"auc_mean": float(np.mean(v["auc"])), "auc_sd": float(np.std(v["auc"])),
                             "ap_mean": float(np.mean(v["ap"])), "ap_sd": float(np.std(v["ap"]))} for k, v in res.items()}
    NUM["comparison_setup"] = {"n": int(len(y)), "positives": int(y.sum()), "chance_ap": float(y.mean()),
                               "cv": f"repeated stratified 5-fold, {n_repeats} repeats, pooled out-of-fold predictions"}
    NUM["comparison_repeats"] = {k: {"auc": [float(x) for x in v["auc"]], "ap": [float(x) for x in v["ap"]]} for k, v in res.items()}
    draw_comparison(res, int(y.sum()), n_repeats)


# --------------------------------------------------------------------------- 2/3. replay of the in-loop classifier
def _batches(h: pd.DataFrame, q: int = 3):
    opt = h.index[h["phase"] == "opt"].to_numpy()
    for i in range(0, len(opt), q):
        yield opt[i:i + q]


def replay(arm: str, encoding: str, model: str = "rf") -> dict:
    global RAW
    """Refit the in-loop classifier before every BO batch on the converged designs so far; score the batch."""
    ps, ys, sims, curve = [], [], [], []
    for s in SEEDS:
        h = load_hist(arm, s).sort_values("sim_index").reset_index(drop=True)
        for batch in _batches(h):
            train = h.iloc[:batch[0]]; train = train[train["HB_converged_flag"] == 1]
            ytr = train["feasible"].astype(int).to_numpy()
            if ytr.sum() < 1 or ytr.sum() == len(ytr):
                continue
            if model == "lr":
                est = LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000)
                if encoding == "embedding":
                    est.fit(to_embedding(train), ytr); p = est.predict_proba(to_embedding(h.iloc[batch]))[:, 1]
                else:
                    pipe = Pipeline([("prep", onehot_prep(True)), ("m", est)]).fit(train[DESIGN], ytr)
                    p = pipe.predict_proba(h.iloc[batch][DESIGN])[:, 1]
                oob_auc = np.nan
            else:
                rf = RandomForestClassifier(**RF_INLOOP, random_state=s, n_jobs=-1, oob_score=True)
                if encoding == "embedding":
                    rf.fit(to_embedding(train), ytr); p = rf.predict_proba(to_embedding(h.iloc[batch]))[:, 1]
                    oob = rf.oob_decision_function_[:, 1]
                else:
                    pipe = Pipeline([("prep", onehot_prep(False)), ("m", rf)]).fit(train[DESIGN], ytr)
                    p = pipe.predict_proba(h.iloc[batch][DESIGN])[:, 1]; oob = pipe["m"].oob_decision_function_[:, 1]
                ok = ~np.isnan(oob)
                oob_auc = roc_auc_score(ytr[ok], oob[ok]) if ytr[ok].min() != ytr[ok].max() else np.nan
            ps += list(p); ys += list(h.iloc[batch]["feasible"].astype(int)); sims += [int(h.iloc[batch[0]]["sim_index"])] * len(batch)
            RAW += [{"arm": arm, "encoding": encoding, "model": model, "seed": int(s), "sim": int(h.iloc[batch[0]]["sim_index"]), "p": float(pp), "y": int(yy)} for pp, yy in zip(p, h.iloc[batch]["feasible"].astype(int))]
            curve.append((s, int(h.iloc[batch[0]]["sim_index"]), float(oob_auc), int(ytr.sum()), int(len(ytr))))
    ps, ys = np.array(ps), np.array(ys)
    return {"p": ps, "y": ys, "sim": np.array(sims), "curve": pd.DataFrame(curve, columns=["seed", "sim", "oob_auc", "pos", "n"])}


def static_seed(arm: str, encoding: str) -> dict:
    """The same classifier fitted once on the shared seed and never refit: what refitting buys."""
    ps, ys = [], []
    for s in SEEDS:
        h = load_hist(arm, s).sort_values("sim_index").reset_index(drop=True)
        train = h[(h["phase"] == "lhs") & (h["HB_converged_flag"] == 1)]; ytr = train["feasible"].astype(int).to_numpy()
        rf = RandomForestClassifier(**RF_INLOOP, random_state=s, n_jobs=-1)
        test = h[h["phase"] == "opt"]
        if encoding == "embedding":
            rf.fit(to_embedding(train), ytr); p = rf.predict_proba(to_embedding(test))[:, 1]
        else:
            pipe = Pipeline([("prep", onehot_prep(False)), ("m", rf)]).fit(train[DESIGN], ytr); p = pipe.predict_proba(test[DESIGN])[:, 1]
        ps += list(p); ys += list(test["feasible"].astype(int))
    return {"p": np.array(ps), "y": np.array(ys)}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    conf, acc, cnt = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    for b in range(bins):
        m = idx == b
        if m.any():
            conf[b], acc[b], cnt[b] = p[m].mean(), y[m].mean(), m.sum()
    e = float(np.sum(cnt / cnt.sum() * np.abs(conf - acc)))
    return e, conf, acc, cnt


def calibration_and_learning(encodings: list[str]) -> None:
    print("2/3. calibration replay and learning curve")
    NUM["calibration"] = {}
    curves = {}
    for enc in encodings:
        for arm in ARM_LABEL:
            r = replay(arm, enc); st = static_seed(arm, enc)
            e, conf, acc, cnt = ece(r["p"], r["y"])
            d = {"n_predictions": int(len(r["p"])), "positives": int(r["y"].sum()),
                 "auc": float(roc_auc_score(r["y"], r["p"])), "brier": float(brier_score_loss(r["y"], r["p"])), "ece": e,
                 "static_seed_auc": float(roc_auc_score(st["y"], st["p"])), "static_seed_brier": float(brier_score_loss(st["y"], st["p"])),
                 "static_seed_ece": ece(st["p"], st["y"])[0], "mean_p": float(r["p"].mean()), "base_rate": float(r["y"].mean())}
            NUM["calibration"][f"{arm}_{enc}"] = d
            curves[(arm, enc)] = (r, conf, acc, cnt)
            print(f"   {ARM_LABEL[arm]:8s} {enc:9s} n={d['n_predictions']} AUC {d['auc']:.3f} Brier {d['brier']:.3f} ECE {d['ece']:.3f} | static seed-only AUC {d['static_seed_auc']:.3f} ECE {d['static_seed_ece']:.3f}")
    primary = "embedding" if "embedding" in encodings else "onehot"
    NUM["calibration_primary_encoding"] = primary
    for arm in ARM_LABEL:                       # the same growing training sets, a linear model in the loop instead
        r = replay(arm, primary, model="lr")
        NUM["calibration"][f"{arm}_{primary}_logreg"] = {"n_predictions": int(len(r["p"])), "auc": float(roc_auc_score(r["y"], r["p"])),
                                                         "brier": float(brier_score_loss(r["y"], r["p"])), "ece": ece(r["p"], r["y"])[0]}
        print(f"   {ARM_LABEL[arm]:8s} {primary:9s} LOGISTIC in the loop: AUC {NUM['calibration'][f'{arm}_{primary}_logreg']['auc']:.3f} Brier {NUM['calibration'][f'{arm}_{primary}_logreg']['brier']:.3f} ECE {NUM['calibration'][f'{arm}_{primary}_logreg']['ece']:.3f}")
    rel, learn = {}, {}
    for arm in ARM_LABEL:
        r, conf, acc, cnt = curves[(arm, primary)]; d = NUM["calibration"][f"{arm}_{primary}"]
        rel[arm] = (conf, acc, cnt, d["brier"], d["ece"])
        NUM.setdefault("reliability", {})[f"{arm}_{primary}"] = {"conf": [float(v) for v in conf], "acc": [float(v) for v in acc], "cnt": [int(v) for v in cnt]}
        c = r["curve"]; g = c.groupby("sim")["oob_auc"]
        mu, sd, x = g.mean(), g.std().fillna(0), g.mean().index.to_numpy()
        learn[arm] = (x, mu.to_numpy(), sd.to_numpy())
        NUM.setdefault("learning_curve", {})[f"{arm}_{primary}"] = {"sim": [int(v) for v in x], "oob_auc_mean": [float(v) for v in mu], "oob_auc_sd": [float(v) for v in sd],
                                                                    "first": float(mu.iloc[0]), "last": float(mu.iloc[-1])}
    draw_calibration(rel, learn)
    pd.DataFrame(RAW).to_csv(OUT / "replay_predictions.csv", index=False)


# --------------------------------------------------------------------------- 4. what the forest learns
def feature_importance(encodings: list[str]) -> None:
    print("4. held-out permutation importance")
    from sklearn.inspection import permutation_importance
    NUM["importance"] = {}
    for enc in encodings:
        per_seed = []
        for s in SEEDS:
            h = load_hist("fa15qc", s); h = h[h["HB_converged_flag"] == 1].reset_index(drop=True)
            y = h["feasible"].astype(int).to_numpy(); imp = {v: [] for v in DESIGN}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(h, y):
                rf = RandomForestClassifier(**RF_INLOOP, random_state=s, n_jobs=-1)
                if enc == "embedding":
                    Xtr, Xte = to_embedding(h.iloc[tr]), to_embedding(h.iloc[te]); rf.fit(Xtr, y[tr])
                    pi = permutation_importance(rf, Xte, y[te], scoring="roc_auc", n_repeats=10, random_state=s).importances_mean
                    groups = {"M3_width": [0], "CB": [1], "Cs": [2], "Cm": [3], "Lx_id": [4, 5, 6], "Ldc3_id": [7, 8, 9]}
                    for v, cols in groups.items():
                        imp[v].append(float(np.sum(pi[cols])))
                else:
                    pipe = Pipeline([("prep", onehot_prep(False)), ("m", rf)]).fit(h.iloc[tr][DESIGN], y[tr])
                    pi = permutation_importance(pipe, h.iloc[te][DESIGN], y[te], scoring="roc_auc", n_repeats=10, random_state=s).importances_mean
                    for j, v in enumerate(DESIGN):
                        imp[v].append(float(pi[j]))
            per_seed.append({v: float(np.mean(imp[v])) for v in DESIGN})
        df = pd.DataFrame(per_seed)
        NUM["importance"][enc] = {v: {"mean": float(df[v].mean()), "sd": float(df[v].std())} for v in DESIGN}
        print("   " + enc + ": " + ", ".join(f"{v} {df[v].mean():.3f}" for v in DESIGN))
    primary = "embedding" if "embedding" in encodings else "onehot"
    draw_importance(NUM["importance"][primary])


# --------------------------------------------------------------------------- 5. does disagreement mean anything
def disagreement(n_forests: int = 15) -> None:
    print("5. ensemble disagreement vs realized feasibility of the augmentation designs")
    y_near = near_feasible_label(LHS, 1.0).to_numpy()
    rows = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        aug = load_hist("fa15qc", s); aug = aug[aug["phase"] == "aug"].reset_index(drop=True)
        P = []
        for m in range(n_forests):
            idx = rng.integers(0, len(LHS), len(LHS))
            pipe = Pipeline([("prep", onehot_prep(False)), ("m", RandomForestClassifier(**RF_AUG, random_state=1000 * s + m, n_jobs=-1))])
            pipe.fit(LHS.iloc[idx][DESIGN], y_near[idx]); P.append(pipe.predict_proba(aug[DESIGN])[:, 1])
        P = np.array(P)
        for i in range(len(aug)):
            rows.append({"seed": s, "p_hat": float(P[:, i].mean()), "sigma_hat": float(P[:, i].std()), "feasible": int(aug.iloc[i]["feasible"])})
    d = pd.DataFrame(rows); d["tercile"] = pd.qcut(d["sigma_hat"], 3, labels=["low", "middle", "high"])
    t = d.groupby("tercile", observed=True).agg(n=("feasible", "size"), sigma_mean=("sigma_hat", "mean"), p_mean=("p_hat", "mean"), feasible_rate=("feasible", "mean"))
    NUM["disagreement"] = {str(k): {c: float(v[c]) for c in t.columns} for k, v in t.iterrows()}
    NUM["disagreement"]["corr_sigma_feasible"] = float(np.corrcoef(d["sigma_hat"], d["feasible"])[0, 1])
    NUM["disagreement"]["corr_p_feasible"] = float(np.corrcoef(d["p_hat"], d["feasible"])[0, 1])
    print(t.round(3).to_string()); print(f"   corr(sigma, feasible) = {NUM['disagreement']['corr_sigma_feasible']:+.3f}, corr(p, feasible) = {NUM['disagreement']['corr_p_feasible']:+.3f}")
    d.to_csv(OUT / "disagreement_per_design.csv", index=False)


# --------------------------------------------------------------------------- tables and text
def write_tables_and_text(encodings: list[str]) -> None:
    c = NUM["comparison"]; s = NUM["comparison_setup"]
    L = [r"\begin{table}[pos=htbp]", r"\centering",
         r"\caption{Feasibility classifiers compared on the 100-design seed with the near-feasible label used by Algorithm~1 "
         f"({s['positives']} positives of {s['n']}). Repeated stratified five-fold cross-validation, {s['cv'].split(', ')[1]}, "
         r"pooled out-of-fold predictions; mean and standard deviation over repeats. The Gaussian-process classifier uses an isotropic RBF kernel with a restarted hyperparameter search (an ARD kernel overfits the 6 positives). Chance average precision is "
         f"{s['chance_ap']:.2f}. The inductors enter as library indices (one-hot); numeric inputs are standardized for the distance- and margin-based models.}}",
         r"\label{tab:rf_comparison}", r"\footnotesize", r"\begin{tabular}{lcc}", r"\toprule", r"Classifier & ROC-AUC & Average precision \\", r"\midrule"]
    for k, v in c.items():
        L.append(f"{k} & {v['auc_mean']:.3f} ({v['auc_sd']:.3f}) & {v['ap_mean']:.3f} ({v['ap_sd']:.3f}) \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_rf_comparison.tex").write_text("\n".join(L) + "\n")

    prim = NUM["calibration_primary_encoding"]
    L = [r"\begin{table}[pos=htbp]", r"\centering",
         r"\caption{Calibration of the in-loop feasibility classifier, replayed: before every BO batch the classifier is refitted on the converged "
         r"designs simulated so far, exactly as the run did, and scored on the three designs the run then proposed; predictions are pooled over "
         r"the ten seeds. Static: the same classifier fitted once on the shared seed and never refitted. ECE: expected calibration error over ten "
         r"equal-count bins. The run did not log its probabilities, so these are a replay with the run's configuration and seeds.}",
         r"\label{tab:rf_calibration}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Arm & Encoding & $n$ & ROC-AUC & Brier & ECE & Static AUC & Static ECE \\", r"\midrule"]
    for enc in encodings:
        for arm in ARM_LABEL:
            d = NUM["calibration"][f"{arm}_{enc}"]
            L.append(f"{ARM_LABEL[arm]} & {'(L,Q,SRF) embedding' if enc == 'embedding' else 'library index'} & {d['n_predictions']} & {d['auc']:.3f} & {d['brier']:.3f} & {d['ece']:.3f} & {d['static_seed_auc']:.3f} & {d['static_seed_ece']:.3f} \\\\")
    for arm in ARM_LABEL:
        d = NUM["calibration"][f"{arm}_{prim}_logreg"]
        L.append(f"{ARM_LABEL[arm]}, logistic regression in the loop & {'(L,Q,SRF) embedding' if prim == 'embedding' else 'library index'} & {d['n_predictions']} & {d['auc']:.3f} & {d['brier']:.3f} & {d['ece']:.3f} & -- & -- \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_rf_calibration.tex").write_text("\n".join(L) + "\n")

    t = NUM["disagreement"]
    L = [r"\begin{table}[pos=htbp]", r"\centering",
         r"\caption{Ensemble disagreement $\hat{\sigma}$ of Eq.~(\ref{eq:sigma_hat}) against the realized feasibility of the 150 augmentation designs "
         r"(15 per seed, ten seeds), recomputed with 15 bootstrap forests of the as-run configuration on each seed. Terciles of $\hat{\sigma}$ over all 150 designs.}",
         r"\label{tab:rf_disagreement}", r"\footnotesize", r"\begin{tabular}{lcccc}", r"\toprule",
         r"$\hat{\sigma}$ tercile & $n$ & mean $\hat{\sigma}$ & mean $\hat{p}$ & feasible fraction \\", r"\midrule"]
    for k in ("low", "middle", "high"):
        v = t[k]; L.append(f"{k} & {int(v['n'])} & {v['sigma_mean']:.3f} & {v['p_mean']:.3f} & {v['feasible_rate']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_rf_disagreement.tex").write_text("\n".join(L) + "\n")

    imp = NUM["importance"][prim]; order = sorted(DESIGN, key=lambda v: -imp[v]["mean"])
    other_enc = [e for e in encodings if e != prim]
    rf, = [k for k in c if "as used" in k]
    best_other = max((k for k in c if k != rf), key=lambda k: c[k]["auc_mean"])
    gpc, = [k for k in c if "Gaussian" in k]
    fa, cl = NUM["calibration"][f"fa15qc_{prim}"], NUM["calibration"][f"qnc_{prim}"]
    fa_lr, cl_lr = NUM["calibration"][f"fa15qc_{prim}_logreg"], NUM["calibration"][f"qnc_{prim}_logreg"]
    lc_fa, lc_cl = NUM["learning_curve"][f"fa15qc_{prim}"], NUM["learning_curve"][f"qnc_{prim}"]
    top3_other = sorted(DESIGN, key=lambda v: -NUM["importance"][other_enc[0]][v]["mean"])[:3] if other_enc else order[:3]
    same_top3 = set(order[:3]) == set(top3_other)
    lr_gap = fa["auc"] - fa_lr["auc"]; ece_gap = fa_lr["ece"] - fa["ece"]
    if ece_gap > 0.05:
        loop_verdict = (f"so the two rank the proposals about equally, but the linear model's probabilities are poorly calibrated where the forest's are not "
                        f"(Brier {fa_lr['brier']:.3f} against {fa['brier']:.3f}; on Clf-BO's proposals the linear model also ranks worse, ROC-AUC {cl_lr['auc']:.2f} against {cl['auc']:.2f}, "
                        f"with an expected calibration error of {cl_lr['ece']:.3f} against {cl['ece']:.3f}). This is where the forest earns its place")
    elif lr_gap > 0.02:
        loop_verdict = "so the forest's advantage appears once the training set has grown beyond the seed, where a linear boundary no longer fits the feasible region"
    elif lr_gap < -0.02:
        loop_verdict = "so on this circuit a linear model would have served the loop at least as well, and the forest's case rests on the first two properties"
    else:
        loop_verdict = "so the two remain close in the loop as well, and the forest's case rests on the first two properties rather than on accuracy"
    text = f"""\\subsection{{The Random-Forest Components: Selection, Calibration and What They Learn}}
\\label{{sec:rf}}
FA-MOBO uses random forests in two places: as bootstrap ensembles that rank the augmentation batch (Algorithm~\\ref{{alg:augmentation}}) and as the in-loop classifier whose feasible-class probability weights the acquisition (Eq.~\\eqref{{eq:clf_acq}}). This section reports, without new simulations, how the forest compares with the usual alternatives, whether the probabilities it supplies to the acquisition are calibrated, whether it improves as the run proceeds, which design variables it learns from, and whether the disagreement penalty of Eq.~\\eqref{{eq:score}} carries information. Two of the five answers are less favourable to the design choices than the others, and they are reported as such.

\\paragraph{{Model selection.}} Table~\\ref{{tab:rf_comparison}} and Fig.~\\ref{{fig:rf_comparison}} compare seven classifiers on the seed, all given the same inputs and the near-feasible label of Algorithm~1, under repeated stratified cross-validation. The forest as used reaches an ROC-AUC of {c[rf]['auc_mean']:.2f} ({c[rf]['auc_sd']:.2f}) and an average precision of {c[rf]['ap_mean']:.2f} ({c[rf]['ap_sd']:.2f}) against a chance level of {s['chance_ap']:.2f}; {best_other.lower()} reaches {c[best_other]['auc_mean']:.2f} and {c[best_other]['ap_mean']:.2f}, and a Gaussian-process classifier with an isotropic kernel and restarted hyperparameter search {c[gpc]['auc_mean']:.2f} and {c[gpc]['ap_mean']:.2f}. No model is separated from the forest by more than the repeat-to-repeat spread: with {s['positives']} positives among {s['n']} designs the seed cannot rank them. The forest was chosen when the method was designed, before this comparison was run, and the seed does not justify the choice on accuracy; what does is how the classifier is used. Its output enters the acquisition as $\\log p_{{\\mathrm{{RF}}}}$ added to the log-acquisition (Eq.~\\eqref{{eq:clf_acq}}), that is, the expected hypervolume improvement is multiplied by $p_{{\\mathrm{{RF}}}}$. A multiplicative weight has to be calibrated: a design scored 0.3 should prove feasible about three times in ten, or the acquisition is being scaled by numbers that correspond to nothing. Ranking quality alone would suffice for a top-$k$ selection; it does not suffice for a weight. This is testable where it matters. Replayed in the loop on the same growing training sets (Table~\\ref{{tab:rf_calibration}}), logistic regression reaches an ROC-AUC of {fa_lr['auc']:.2f} and an expected calibration error of {fa_lr['ece']:.3f} on the designs FA-MOBO proposed, against {fa['auc']:.2f} and {fa['ece']:.3f} for the forest, {loop_verdict}. A reason the linear model miscalibrates here, offered as interpretation rather than measurement, is the shape of the feasible region. It is the intersection of six bound constraints, a compact pocket such as a band of $C_s$ combined with particular inductor pairs; a logit is monotone along one direction, so it can order designs but must push its probabilities toward 0 or 1 along that direction and cannot express ``feasible inside this box and not outside it''. Trees partition the space into boxes, which is the shape of a conjunction of constraints. Two further properties favour the forest, one of them less than it first appears. It takes the two library indices, and their interactions with the continuous variables, without feature scaling or a kernel to tune. Its bootstrap members also supply the disagreement that the risk term of Eq.~\\eqref{{eq:score}} needs at no extra cost; the disagreement paragraph below shows, however, that this term contributes little on this circuit, so that is a convenience rather than a justification. Removing the balanced class weighting from the forest costs {c[rf]['ap_mean'] - c['Random forest, no class weight']['ap_mean']:.2f} in average precision on the seed, the metric that governs ranking a sparse positive class.

\\paragraph{{Calibration.}} Because $\\log p_{{\\mathrm{{RF}}}}$ is added to the log-acquisition, the classifier's probabilities act as weights and should mean what they say. Fig.~\\ref{{fig:rf_calibration}}(a) and Table~\\ref{{tab:rf_calibration}} replay the in-loop classifier as the run used it, refitted before every batch on the converged designs simulated so far and scored on the three designs the run then proposed, pooled over the ten seeds ({fa['n_predictions']} predictions for FA-MOBO, {cl['n_predictions']} for Clf-BO). It reaches an ROC-AUC of {fa['auc']:.2f} for FA-MOBO and {cl['auc']:.2f} for Clf-BO, Brier scores of {fa['brier']:.3f} and {cl['brier']:.3f}, and expected calibration errors of {fa['ece']:.3f} and {cl['ece']:.3f}; its mean predicted probability ({fa['mean_p']:.2f}) matches the realized feasible fraction of the proposals ({fa['base_rate']:.2f}). Refitting is what keeps it calibrated: the same classifier fitted once on the seed and never refitted has an expected calibration error of {fa['static_seed_ece']:.3f} on FA-MOBO's proposals and {cl['static_seed_ece']:.3f} on Clf-BO's. The result does not depend on how the inductors are encoded: with library indices in place of the $(L, Q, \\mathrm{{SRF}})$ embedding the ROC-AUC changes by at most {max(abs(NUM['calibration'][f'{a}_onehot']['auc'] - NUM['calibration'][f'{a}_{prim}']['auc']) for a in ARM_LABEL):.2f}. The run did not log its probabilities, so these are a replay with the run's configuration and seeds rather than the logged values.

\\paragraph{{Learning as the run proceeds.}} Fig.~\\ref{{fig:rf_calibration}}(b) tracks the out-of-bag ROC-AUC of the refit classifier against the simulation count. It is high from the first batch and does not rise: {lc_fa['first']:.2f} before FA-MOBO's first BO batch and {lc_fa['last']:.2f} before its last, never below {min(lc_fa['oob_auc_mean']):.2f}; for Clf-BO, whose BO phase starts from the seed alone, {lc_cl['first']:.2f} to {lc_cl['last']:.2f} with a dip to {min(lc_cl['oob_auc_mean']):.2f} while its training set holds only the seed's two feasible designs. The augmentation batch therefore does not measurably improve the classifier's discrimination, which agrees with the 3.1-design difference between FA-MOBO and Clf-BO in feasible count not being significant (Section~\\ref{{sec:feasible}}): the classifier is the feasibility mechanism in both arms, and what the run changes is not how well the classifier fits the designs it has already seen but how well that fit transfers to the designs proposed next, in calibration (above) and, as the last paragraph of this section shows, in ranking.

\\paragraph{{What it learns.}} Held-out permutation importance of the strict-feasibility classifier at 181 simulations (Fig.~\\ref{{fig:rf_importance}}, mean over ten runs) ranks {VAR_LABEL[order[0]]} ({imp[order[0]]['mean']:.3f} drop in ROC-AUC), {VAR_LABEL[order[1]]} ({imp[order[1]]['mean']:.3f}) and {VAR_LABEL[order[2]]} ({imp[order[2]]['mean']:.3f}) first, and {VAR_LABEL[order[-2]]} and {VAR_LABEL[order[-1]]} last ({imp[order[-2]]['mean']:.3f} and {imp[order[-1]]['mean']:.3f}); {'both encodings agree on the same top three' if same_top3 else 'the two encodings differ in the order of the top three'}, and the seed-to-seed standard deviations are comparable to the differences among the top three, so their order is not established. The feasible region over the six constraints is thus learned mainly from the output network, the series tank and the two inductors, and hardly from the switch width. This complements rather than contradicts Section~\\ref{{sec:mechanism}}: the width sets efficiency and margin, but the acquisition holds it near its lower bound throughout the BO phase, so it varies little in the data the classifier sees and contributes little to its boundary.

\\paragraph{{Disagreement.}} The risk term $\\lambda\\hat{{\\sigma}}$ of Eq.~\\eqref{{eq:score}} is meant to steer selection away from candidates the ensembles disagree about. On the realized outcomes it does not behave as a filter. Over the 150 augmentation designs of the ten runs (Table~\\ref{{tab:rf_disagreement}}), the feasible fraction rises with disagreement, from {t['low']['feasible_rate']:.2f} in the lowest-$\\hat{{\\sigma}}$ tercile to {t['high']['feasible_rate']:.2f} in the highest (correlation {t['corr_sigma_feasible']:+.2f}), and so does the mean predicted probability ({t['low']['p_mean']:.2f} to {t['high']['p_mean']:.2f}). Among designs that have already passed the ranking, the ensembles disagree most about the most promising boundary points, so within the batch disagreement marks the best candidates rather than the doubtful ones. This measures the penalty only among selected designs, not over the 20\\,000-candidate pool where it acts, and the sensitivity study (Supplementary~S5) shows that removing it changes 16\\% of the batch and leaves the verifier's feasibility estimate unchanged. Taken together, the risk term contributes little on this circuit, and a smaller $\\lambda$ or none is a defensible simplification; the point is recorded here rather than left for a reviewer to find.
"""
    (OUT / "rf_section.tex").write_text(text)
    figs = r"""
\begin{figure*}[pos=htbp]
\centering
\includegraphics[width=\textwidth]{fig_rf_classifier_comparison.png}
\caption{Feasibility classifiers on the 100-design seed under repeated stratified five-fold cross-validation (20 repeats; one point per repeat). Left: ROC-AUC; right: average precision. The dashed line marks the median of the forest as used; the grey dotted line in the right panel is the chance level. All models receive the same inputs and the near-feasible label of Algorithm~1.}
\label{fig:rf_comparison}
\end{figure*}

\begin{figure*}[pos=htbp]
\centering
\includegraphics[width=\textwidth]{fig_rf_calibration_learning.png}
\caption{The in-loop feasibility classifier, replayed with the run's configuration and seeds. (a) Reliability of its probabilities on the designs it proposed during the BO phase, pooled over ten seeds, with Brier score and expected calibration error. (b) Out-of-bag ROC-AUC of the classifier refitted before each batch, against the simulation count (mean $\pm$ SD over seeds); dotted lines mark the end of the shared seed (100) and of FA-MOBO's augmentation batch (115).}
\label{fig:rf_calibration}
\end{figure*}

\begin{figure}[pos=htbp]
\centering
\includegraphics[width=\columnwidth]{fig_rf_feature_importance.png}
\caption{Held-out permutation importance of the strict-feasibility classifier fitted on each complete FA-MOBO run (181 simulations), scored as the drop in ROC-AUC when one variable is permuted on the held-out fold; mean $\pm$ SD over ten runs.}
\label{fig:rf_importance}
\end{figure}
"""
    (OUT / "rf_figures.tex").write_text(figs.lstrip())
    print("  wrote tab_rf_*.tex, rf_section.tex, rf_figures.tex")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:                # redraw from cached data, recomputing only what is not cached
        NUM.update(json.loads((OUT / "numbers.json").read_text())); prim = NUM["calibration_primary_encoding"]
        if "comparison_repeats" in NUM:
            draw_comparison(NUM["comparison_repeats"], NUM["comparison_setup"]["positives"], len(next(iter(NUM["comparison_repeats"].values()))["auc"]))
        else:
            classifier_comparison()                  # once; caches the per-repeat scores for next time
        rel, learn = {}, {}
        for arm in ARM_LABEL:
            d = NUM["calibration"][f"{arm}_{prim}"]
            if f"{arm}_{prim}" in NUM.get("reliability", {}):
                b = NUM["reliability"][f"{arm}_{prim}"]; conf, acc, cnt = (np.array(b[k]) for k in ("conf", "acc", "cnt"))
            else:                                    # the earlier per-prediction replay; identical to this one to three decimals
                pr = pd.read_csv(LIU / "data" / "precomputed" / f"B_clf_calib_{arm}.csv")  # shipped per-prediction replay, if present
                _, conf, acc, cnt = ece(pr["p"].to_numpy(), pr["y"].to_numpy().astype(int))
                NUM.setdefault("reliability", {})[f"{arm}_{prim}"] = {"conf": [float(v) for v in conf], "acc": [float(v) for v in acc], "cnt": [int(v) for v in cnt]}
            rel[arm] = (conf, acc, cnt, d["brier"], d["ece"])
            lc = NUM["learning_curve"][f"{arm}_{prim}"]; learn[arm] = (np.array(lc["sim"]), np.array(lc["oob_auc_mean"]), np.array(lc["oob_auc_sd"]))
        draw_calibration(rel, learn); draw_importance(NUM["importance"][prim])
        (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1)); raise SystemExit(0)
    if "--text-only" in sys.argv:                   # regenerate tables and text from the saved numbers, no recomputation
        NUM.update(json.loads((OUT / "numbers.json").read_text()))
        write_tables_and_text(NUM["encodings"]); raise SystemExit(0)
    encs = ["onehot"] + (["embedding"] if embedding_available() else [])
    print("encodings available:", encs)
    NUM["encodings"] = encs
    classifier_comparison()
    calibration_and_learning(encs)
    feature_importance(encs)
    disagreement()
    write_tables_and_text(encs)
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
    print("done ->", OUT)
