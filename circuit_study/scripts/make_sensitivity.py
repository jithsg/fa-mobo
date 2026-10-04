"""Sensitivity of the Phase II augmentation batch (Algorithm 1) to its hyperparameters, in the configuration FA-MOBO ran.

No simulations are needed: Algorithm 1 only ranks Latin-hypercube candidates with random forests fitted on the shared
100-design seed. For each of the ten campaign seeds it is re-run with the FA-MOBO settings (efficiency weight exponent 1,
top 15 of the ranked shortlist), which is checked against the simulated augmentation batch, and then with one
hyperparameter changed at a time. Each changed batch is compared with the batch as run.

Writes out/sensitivity_current_per_seed.csv, out/sensitivity_current.json and out/tab_s_augmentation_sensitivity.tex.
Run from the scripts folder:  python make_sensitivity.py [--table-only]
(--table-only rebuilds the table from data/precomputed/sensitivity_current_per_seed.csv; the full sweep takes about 90 min)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from common import DATA, OUT, SEEDS, load_hist

from algo1 import NUM_COLS, PAPER, run_algo1, train_verifier, verifier_pfeas, with_params  # noqa: E402

LHS_CSV = DATA / "seed" / "lhs_100.csv"
MAIN = "fa15qc"
N_AUG = 15                      # FA-MOBO simulates the top 15 of the ranked shortlist
WIDE_UM = 950.0                 # switch-width level used in the mechanism section
BASE = with_params(PAPER, pae_weight=1.0)
SETTINGS = [  # (group, value shown in the table, configuration)
    ("As run", "", BASE),
    ("$\\lambda$ (0.5)", "0", with_params(BASE, lambda_risk=0.0)),
    ("$\\lambda$ (0.5)", "0.25", with_params(BASE, lambda_risk=0.25)),
    ("$\\lambda$ (0.5)", "1", with_params(BASE, lambda_risk=1.0)),
    ("Label relaxation (1\\,dB)", "0.5\\,dB", with_params(BASE, relax_op1db_db=0.5)),
    ("Label relaxation (1\\,dB)", "2\\,dB", with_params(BASE, relax_op1db_db=2.0)),
    ("Ensembles (5, 10, 15)", "5", with_params(BASE, ensemble_sizes=(5,))),
    ("Ensembles (5, 10, 15)", "15", with_params(BASE, ensemble_sizes=(15,))),
    ("Ensembles (5, 10, 15)", "10, 20, 30", with_params(BASE, ensemble_sizes=(10, 20, 30))),
    ("Efficiency exponent (1)", "0", with_params(BASE, pae_weight=0.0)),
    ("Efficiency exponent (1)", "2", with_params(BASE, pae_weight=2.0)),
]
PER_SEED = OUT / "sensitivity_current_per_seed.csv"
SUMMARY = OUT / "sensitivity_current.json"
TABLE = OUT / "tab_s_augmentation_sensitivity.tex"


def design_keys(d: pd.DataFrame) -> list[str]:
    return [f"{r['Lx_id']}|{r['Ldc3_id']}|" + "|".join(f"{float(r[k]):.4f}" for k in NUM_COLS) for _, r in d.iterrows()]


def diversity(sel: pd.DataFrame, lhs: pd.DataFrame) -> float:
    """Mean pairwise distance of the four continuous variables, each scaled to the seed's range."""
    Z = np.column_stack([(sel[k].to_numpy(float) - lhs[k].min()) / (lhs[k].max() - lhs[k].min()) for k in NUM_COLS])
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(len(Z), 1)].mean())


def sweep() -> pd.DataFrame:
    lhs = pd.read_csv(LHS_CSV)
    verifier = train_verifier(lhs)
    rows = []
    for s in SEEDS:
        logged = design_keys(load_hist(MAIN, s).query("phase == 'aug'"))
        ref: set[str] = set()
        for i, (group, value, cfg) in enumerate(SETTINGS):
            t0 = time.time()
            sel = run_algo1(lhs, cfg, seed=s).iloc[:N_AUG].reset_index(drop=True)
            k = design_keys(sel)
            if i == 0:
                ref = set(k)
            p = verifier_pfeas(verifier, sel)
            w = sel["M3_width"].to_numpy(float)
            rows.append({"seed": s, "setting": i, "group": group, "value": value,
                         "reproduces_simulated_batch": (set(k) == set(logged)) if i == 0 else None,  # same 15 designs; the order can differ
                         "overlap": len(set(k) & ref) / N_AUG, "verifier_p_mean": float(p.mean()), "diversity": diversity(sel, lhs),
                         "n_lx": int(sel["Lx_id"].nunique()), "n_ldc3": int(sel["Ldc3_id"].nunique()),
                         "median_width_um": float(np.median(w)), "share_wide": float(np.mean(w >= WIDE_UM)), "seconds": time.time() - t0})
            print(f"seed {s} setting {i} ({group} {value}): overlap {rows[-1]['overlap']:.2f} in {rows[-1]['seconds']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(PER_SEED, index=False)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> list[dict]:
    out = []
    for i, (group, value, _) in enumerate(SETTINGS):
        d = df[df.setting == i]
        out.append({"setting": i, "group": group, "value": value, "seeds": int(len(d)),
                    "overlap_mean": float(d.overlap.mean()), "overlap_min": float(d.overlap.min()),
                    "verifier_p_mean": float(d.verifier_p_mean.mean()), "diversity": float(d.diversity.mean()),
                    "median_width_um": float(d.median_width_um.mean()), "share_wide": float(d.share_wide.mean()),
                    "n_lx": float(d.n_lx.mean()), "n_ldc3": float(d.n_ldc3.mean()),
                    "reproduced_seeds": int(d.reproduces_simulated_batch.astype(str).eq("True").sum()) if i == 0 else None})
    return out


def table(summary: list[dict]) -> str:
    base = summary[0]
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{Sensitivity of the 15-design augmentation batch to the hyperparameters of Algorithm~1, without simulations. For each of the ten seeds, Algorithm~1 was re-run on the shared seed as FA-MOBO ran it ($\lambda=0.5$, $OP_{1dB}$ label relaxation 1\,dB, ensembles of 5, 10 and 15 forests, efficiency weight exponent 1) and then with one hyperparameter changed; the value in brackets is the one FA-MOBO used. "
         + f"The as-run setting selects the same 15 designs that were simulated on {base['reproduced_seeds']} of {base['seeds']} seeds. "
         r"Overlap: share of the 15 designs that are also in the batch as run (mean over seeds, lowest seed in brackets). Verifier $p$: mean feasibility probability of the batch predicted by the post-selection verifier forest, a prediction rather than a simulation outcome. Diversity: mean pairwise distance of the four continuous variables, each scaled to the seed's range. Width: median switch width $W_{\mathrm{M3}}$ of the batch; $\geq950\,\mu$m: share of batch designs with $W_{\mathrm{M3}}\geq950\,\mu$m; $L_x$ / $L_{\mathrm{dc3}}$: number of distinct library inductors used. Label relaxation: relaxation of the $OP_{1dB}$ threshold in the near-feasible label; efficiency exponent: exponent of the efficiency weight. All values are means over the ten seeds.}",
         r"\label{tab:s_aug_sensitivity}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llcccccc}", r"\hline",
         r"Hyperparameter & Value & Overlap & Verifier $p$ & Diversity & Width ($\mu$m) & $\geq950\,\mu$m & $L_x$ / $L_{\mathrm{dc3}}$ \\", r"\hline"]
    last = None
    for r in summary:
        g = r["group"] if r["group"] != last else ""
        last = r["group"]
        L.append(f"{g} & {r['value']} & {r['overlap_mean']:.2f} ({r['overlap_min']:.2f}) & {r['verifier_p_mean']:.2f} & {r['diversity']:.2f} & "
                 f"{r['median_width_um']:.0f} & {r['share_wide']:.2f} & {r['n_lx']:.1f} / {r['n_ldc3']:.1f} \\\\")
        if r["setting"] in (0, 3, 5, 8):
            L.append(r"\hline")
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-only", action="store_true", help="rebuild the table from the saved per-seed results")
    a = ap.parse_args()
    df = pd.read_csv(DATA / "precomputed" / PER_SEED.name) if a.table_only else sweep()
    summary = summarize(df)
    SUMMARY.write_text(json.dumps(summary, indent=1))
    TABLE.write_text(table(summary))
    print("wrote", PER_SEED.name, SUMMARY.name, TABLE.name)


if __name__ == "__main__":
    main()
