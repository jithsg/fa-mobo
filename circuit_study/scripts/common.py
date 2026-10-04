"""Shared loaders, labels and plot style for the revised manuscript assets.

All numbers come from the multi-seed campaign histories (data/runs) and the
10-seed corner sweep (data/corner_sweep). Nothing is typed in by hand.

Visual language shared by every figure (make_figures.py, make_gabstract.py):
  * Okabe-Ito colour-blind-safe palette; FA-MOBO is vermillion and drawn heavier.
  * Phase colours (seed / augmentation / BO) are distinct from every arm colour.
  * 8 pt base font, 7 pt ticks and legends, panel titles "(a) ..." left-aligned.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # package root (04_code_and_data)
ROB = ROOT                                             # name kept from the original scripts
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
from problem import feasible_mask  # noqa: E402
from base import dominated_hypervolume  # noqa: E402

DPI = 1500
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
N_SEED = 100          # shared LHS seed
BUDGET = 181          # total simulations per run
# Arms in the paper (10 seeds each). Key -> (short label, long label, colour, linestyle)
# Colours: Okabe-Ito (vermillion, orange, reddish purple, sky blue, blue).
ARMS = {  # Okabe-Ito palette (colour-blind safe); (short, long, colour, linestyle, linewidth)
    "fa15qc": ("FA-MOBO", "FA-MOBO (augmentation + classifier-guided BO)", "#D55E00", "-", 1.8),
    "qnc":    ("Clf-BO",  "Classifier-guided BO, no augmentation",          "#E69F00", "--", 0.9),
    "fa15q":  ("Aug-BO",  "Augmentation + plain qNEHVI (no classifier)",    "#CC79A7", "-.", 0.9),
    "qnm":    ("qNEHVI+m", "qNEHVI with 0.3 dB OP1dB margin",               "#56B4E9", ":", 0.9),
    "qn":     ("qNEHVI",  "Plain constrained qNEHVI",                       "#0072B2", "-", 0.9),
}
EA_COLOR = "0.4"
PHASE_COLORS = {"seed": "0.78", "aug": "#009E73", "bo": "0.25"}  # neutral phase colours, distinct from every arm colour
MAIN_ARM = "fa15qc"
LW_MAIN, LW_OTHER = 1.7, 1.05
PHASE_LABELS = {"lhs": "shared LHS seed", "aug": "augmentation phase", "opt": "BO phase"}
BOX_COLORS = {"seed": "#E8EEF5", "aug": "#E3F3EC", "bo": "#FBEAE0", "out": "#F2F2F2"}
SIM_LABEL = "Harmonic-balance simulations"
# Corner-stability rule validated by the sweep (OP1dB >= 15.8 dBm, K >= 15 at TT/27C)
OP1DB_MARGIN = 15.8
K_MARGIN = 15.0
K_SEL = 10.0          # selection threshold used for the sweep's candidate list


def load_hist(arm: str, seed: int) -> pd.DataFrame:
    h = pd.read_csv(DATA / "runs" / f"{arm}_seed{seed}.csv")
    return h.iloc[:BUDGET].reset_index(drop=True)


def margin_rich(h: pd.DataFrame, k_min: float = K_MARGIN) -> pd.Series:
    return feasible_mask(h) & (h["OP1dB_dBm"] >= OP1DB_MARGIN) & (h["minKf"] >= k_min)


def per_seed_table() -> pd.DataFrame:
    """One row per (arm, seed): HV, feasible, corner-stable counts, phase splits, best PAE."""
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            h = load_hist(arm, s)
            fm = feasible_mask(h); mr = margin_rich(h)
            ph = h["phase"]
            rows.append({
                "arm": arm, "seed": s,
                "hv": dominated_hypervolume(h.loc[fm, "PAE_percent"].to_numpy(), h.loc[fm, "Psat_dBm"].to_numpy()),
                "feasible": int(fm.sum()),
                "feasible_lhs": int((fm & (ph == "lhs")).sum()),
                "feasible_aug": int((fm & (ph == "aug")).sum()), "feasible_opt": int((fm & (ph == "opt")).sum()),
                "n_aug": int((ph == "aug").sum()), "n_opt": int((ph == "opt").sum()),
                "k15": int(mr.sum()), "k15_lhs": int((mr & (ph == "lhs")).sum()),
                "k15_aug": int((mr & (ph == "aug")).sum()), "k15_opt": int((mr & (ph == "opt")).sum()),
                "k15_new": int(mr.sum()) - int((mr & (ph == "lhs")).sum()),
                "best_pae": float(h.loc[fm, "PAE_percent"].max()),
                "best_k15_pae": float(h.loc[mr, "PAE_percent"].max()) if mr.any() else np.nan,
            })
    return pd.DataFrame(rows)


def run_minutes() -> pd.DataFrame:
    """Wall-clock minutes per run parsed from the '[done] ... in N min' log lines."""
    rows = {}
    for f in glob.glob(str(DATA / "logs" / "**" / "*.log"), recursive=True):  # run logs are not distributed; unused by any figure or table
        txt = Path(f).read_text(errors="ignore")
        m = re.search(r"\[done\] (\w+) seed (\d+): .* in (\d+) min", txt)
        if m:
            rows[(m.group(1), int(m.group(2)))] = int(m.group(3))
    return pd.DataFrame([{"arm": a, "seed": s, "minutes": v} for (a, s), v in rows.items()])


# ----------------------------------------------------------------------------- style helpers
FS = {"base": 8.0, "tick": 7.0, "legend": 7.0, "title": 8.0, "note": 7.0}


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS["base"], "axes.labelsize": FS["base"], "axes.titlesize": FS["title"],
        "legend.fontsize": FS["legend"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "axes.linewidth": 0.6, "lines.linewidth": LW_OTHER, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.titlelocation": "left", "axes.titlepad": 4,
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5,
    })


def line_kw(arm: str) -> dict:
    """Line style of an arm: FA-MOBO heavier and on top, every other arm lighter."""
    _, _, col, ls = ARMS[arm]
    main = arm == MAIN_ARM
    return {"color": col, "ls": ls, "lw": LW_MAIN if main else LW_OTHER, "zorder": 4 if main else 3}


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=FS["title"])


def swarm(values, center: float, width: float = 0.32, resolution: float | None = None) -> np.ndarray:
    """Deterministic x offsets for a strip of points: ties are spread symmetrically, no random jitter."""
    v = np.asarray(values, dtype=float)
    if resolution is None:
        span = np.ptp(v) if len(v) > 1 else 1.0
        resolution = max(span / 12.0, 1e-9)
    keys = np.round(v / resolution).astype(int)
    xs = np.full(len(v), center, dtype=float)
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        n = len(idx)
        if n > 1:
            step = min(width / max(n - 1, 1), 0.12)
            xs[idx] = center + (np.arange(n) - (n - 1) / 2) * step
    return xs


def spread(ys, min_gap: float, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Push label positions apart (1-D) so that consecutive labels are at least `min_gap` apart."""
    y = np.asarray(ys, dtype=float)
    order = np.argsort(y)
    out = y[order].copy()
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < min_gap:
            out[i] = out[i - 1] + min_gap
    if hi is not None and len(out) and out[-1] > hi:
        out -= out[-1] - hi
        for i in range(len(out) - 2, -1, -1):
            if out[i + 1] - out[i] < min_gap:
                out[i] = out[i + 1] - min_gap
    if lo is not None and len(out) and out[0] < lo:
        out += lo - out[0]
    res = np.empty_like(out); res[order] = out
    return res


def save(fig, name: str) -> None:
    from PIL import Image
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor="white")
    Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.png", dpi=(DPI, DPI))  # RGB: no SMask in pdfTeX
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}_preview.png", dpi=110)
    print(f"wrote {name}.png ({DPI} dpi), .pdf, _preview.png")


# Protocol constants used only for labelling diagrams (values come from the run protocol, not from data files)
DESIGN_VARS = ["Lx_id", "Ldc3_id", "M3_width", "CB", "Cs", "Cm"]   # 2 discrete spiral ids + 4 continuous
N_CONSTRAINTS = 6      # the six strict specifications tested by problem.feasible_mask
Q_BATCH = 3            # designs simulated per BO batch (GPs and classifier refitted after every batch)


def beeswarm(values, width: float = 0.42, step: float | None = None) -> np.ndarray:
    """Deterministic x-offsets that spread equal/near-equal values sideways instead of random jitter."""
    v = np.asarray(values, float); n = len(v)
    if n == 0:
        return v
    step = step if step is not None else max((v.max() - v.min()) / 12.0, 1e-9)
    order = np.argsort(v); off = np.zeros(n); placed = []
    for i in order:
        k = sum(1 for (vv, _) in placed if abs(vv - v[i]) < step * 0.75)
        off[i] = ((k + 1) // 2) * (0.075 if k else 0) * (1 if k % 2 else -1)
        placed.append((v[i], off[i]))
    return np.clip(off, -width, width)


# ============================================================================= figure helpers
# Appended 2026-09-15 for the FA-MOBO revision figures. Definitions above this block are unchanged.
from problem import CONSTRAINTS as _CONSTRAINTS, CONTINUOUS_BOUNDS as _BOUNDS  # noqa: E402

OP1DB_SPEC = float(_CONSTRAINTS["OP1dB_dBm_min"])      # strict OP1dB constraint (dBm)
HM_SLACK = OP1DB_MARGIN - OP1DB_SPEC                   # OP1dB slack required of a high-margin design (dB)
W_MIN_UM = float(_BOUNDS["M3_width"][0])               # lower bound of the switch width (um)
JOINT_FEASIBLE, JOINT_NEW_HM = 10, 1                   # joint target: >= 10 strictly feasible AND >= 1 new high-margin design
CLF_ARMS = ("fa15qc", "qnc")                           # arms whose BO phase uses the in-loop feasibility classifier
ORIGINS = ("seed", "aug", "bo_clf", "bo_plain")        # priority order when one design occurs in several places
ORIGIN_LABELS = {"seed": "shared LHS seed", "aug": "augmentation phase",
                 "bo_clf": "BO phase, with classifier", "bo_plain": "BO phase, without classifier"}
ORIGIN_STYLE = {  # scatter styles shared by every design-level panel (neutral colours, distinct from the arm colours)
    "bo_plain": {"marker": "o", "s": 7.0, "facecolors": "none", "edgecolors": "0.60", "linewidths": 0.45, "alpha": 0.75, "zorder": 2},
    "bo_clf": {"marker": "o", "s": 5.0, "facecolors": "0.10", "edgecolors": "none", "linewidths": 0.0, "alpha": 0.45, "zorder": 3},
    "aug": {"marker": "^", "s": 17.0, "facecolors": PHASE_COLORS["aug"], "edgecolors": "white", "linewidths": 0.35, "alpha": 1.0, "zorder": 5},
    "seed": {"marker": "s", "s": 20.0, "facecolors": "white", "edgecolors": "black", "linewidths": 0.8, "alpha": 1.0, "zorder": 6},
}
K_BANDS = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, np.inf)


def design_key(df: pd.DataFrame, nd: int = 4) -> pd.Series:
    """Hashable key on the six design variables (continuous variables rounded to 10^-nd)."""
    cont = [c for c in DESIGN_VARS if c not in ("Lx_id", "Ldc3_id")]
    return df["Lx_id"].astype(str) + "|" + df["Ldc3_id"].astype(str) + "|" + df[cont].round(nd).astype(str).agg("|".join, axis=1)


def origin_of(arm: str, phase: str) -> str:
    """Origin of a design: shared seed, augmentation phase, or BO phase with/without the classifier."""
    if phase == "lhs":
        return "seed"
    if phase == "aug":
        return "aug"
    return "bo_clf" if arm in CLF_ARMS else "bo_plain"


def pooled_designs() -> pd.DataFrame:
    """Distinct designs over the first 181 rows of all 50 runs, one row each, labelled with the highest-priority origin.

    Returns:
        DataFrame with the history columns plus arm, seed, origin, feas (strictly feasible), high_margin, slack (OP1dB - spec).
    """
    d = pd.concat([load_hist(a, s).assign(arm=a, seed=s) for a in ARMS for s in SEEDS], ignore_index=True)
    d = d.assign(origin=[origin_of(a, p) for a, p in zip(d["arm"], d["phase"])], feas=feasible_mask(d).to_numpy(),
                 high_margin=margin_rich(d).to_numpy(), key=design_key(d).to_numpy(), slack=d["OP1dB_dBm"] - OP1DB_SPEC)
    d = d.assign(prio=d["origin"].map({o: i for i, o in enumerate(ORIGINS)}))
    return d.sort_values(["prio", "sim_index"], kind="stable").drop_duplicates("key").reset_index(drop=True)


def joint_target_sims(arm: str) -> np.ndarray:
    """Per seed (SEEDS order): first sim_index with cumulative strictly feasible >= 10 and cumulative new high-margin >= 1; NaN if never."""
    out = []
    for s in SEEDS:
        h = load_hist(arm, s)
        cf = np.cumsum(feasible_mask(h).to_numpy())
        cm = np.cumsum((margin_rich(h) & (h["sim_index"] > N_SEED)).to_numpy())
        hit = np.flatnonzero((cf >= JOINT_FEASIBLE) & (cm >= JOINT_NEW_HM))
        out.append(float(h["sim_index"].iloc[hit[0]]) if len(hit) else np.nan)
    return np.array(out)


def reach_fraction(reach: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Fraction of runs whose joint-target index is <= each simulation index (unreached runs never count)."""
    r = np.asarray(reach, float)
    return np.array([np.sum(np.isfinite(r) & (r <= x)) / len(r) for x in sims])


def bound_share_thirds(arm: str) -> np.ndarray:
    """Share of BO-phase proposals with switch width at the lower bound, per seed and per equal third of the BO phase (10 x 3)."""
    rows = []
    for s in SEEDS:
        h = load_hist(arm, s)
        w = h.loc[h["phase"] == "opt", "M3_width"].to_numpy()
        pinned = w <= W_MIN_UM + 1e-3
        third = np.minimum(2, 3 * np.arange(len(w)) // len(w))
        rows.append([pinned[third == t].mean() for t in range(3)])
    return np.array(rows)


def hv_margin(h: pd.DataFrame) -> float:
    """Margin-aware hypervolume: 3-D HV of strictly feasible designs in (PAE, Psat, K capped), reference and cap from T3a_metrics."""
    t3 = str(Path(__file__).resolve().parent)
    if t3 not in sys.path:
        sys.path.insert(0, t3)
    from T3a_metrics import K_CAP, K_REF, PSAT_REF, hv3  # noqa: E402
    f = h[feasible_mask(h)]
    return float(hv3(np.c_[f["PAE_percent"], f["Psat_dBm"], f["minKf"].clip(upper=K_CAP)], (0.0, PSAT_REF, K_REF)))


def paired_test(a, b) -> dict:
    """Paired comparison a - b over seeds: mean difference, wins, ties, Wilcoxon signed-rank p on the non-zero differences."""
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    return {"diff": float(d.mean()), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "p": p}


# ----------------------------------------------------------------------------- evolutionary single runs
EA_DIR = DATA / "evolutionary"
EA_METHODS = ("NSGA-II", "SMS-EMOA", "GDE3", "CMOPSO")   # pymoo, one run each, started from the shared LHS seed


def load_ea(method: str) -> pd.DataFrame:
    """First BUDGET rows of an evolutionary single run; its first 100 rows are the shared LHS seed."""
    return pd.read_csv(EA_DIR / f"history_{method}.csv").iloc[:BUDGET].reset_index(drop=True)


def joint_target_index(h: pd.DataFrame) -> float:
    """First sim_index with cumulative strictly feasible >= JOINT_FEASIBLE and new high-margin >= JOINT_NEW_HM (NaN if never)."""
    cf = np.cumsum(feasible_mask(h).to_numpy())
    cm = np.cumsum((margin_rich(h) & (h["sim_index"] > N_SEED)).to_numpy())
    hit = np.flatnonzero((cf >= JOINT_FEASIBLE) & (cm >= JOINT_NEW_HM))
    return float(h["sim_index"].iloc[hit[0]]) if len(hit) else float("nan")


def ea_outcomes(method: str) -> dict:
    """Outcomes of an evolutionary single run with the definitions of the Bayesian arms (Table ms_comparison)."""
    h = load_ea(method)
    fm, mr = feasible_mask(h), margin_rich(h)
    f = h[fm]
    reach = joint_target_index(h)
    return {"feasible": int(fm.sum()), "high_margin": int(mr.sum()), "joint_target": None if np.isnan(reach) else reach,
            "hv3": hv_margin(h), "hv2": float(dominated_hypervolume(f["PAE_percent"].to_numpy(), f["Psat_dBm"].to_numpy())),
            "best_pae": float(f["PAE_percent"].max()) if len(f) else None,
            "best_pae_hm": float(h.loc[mr, "PAE_percent"].max()) if mr.any() else None}
