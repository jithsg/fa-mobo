"""Graphical abstract: headline statement with numbers, three-phase strip, yield bars and the reliability curve.

Output: out/gaabstract.pdf / .png (1500 dpi, RGB) / _preview.png. Every number is computed from the run histories.
Run from the scripts folder: python make_gabstract.py
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import BOX_COLORS, BUDGET, FS, JOINT_FEASIBLE, JOINT_NEW_HM, K_MARGIN, N_SEED, save  # noqa: E402

NOTE = FS["note"]   # smallest text size (7 pt)
from make_figures import ARMS, MAIN, N_RUNS, PS, W, col, paired_test, per_seed, reach_panel, reach_summary, short  # noqa: E402


def headline_numbers(reach: dict) -> dict:
    rs = reach_summary(reach)
    fa = PS[PS["arm"] == MAIN]
    hv_p = [paired_test(per_seed(MAIN, "hv"), per_seed(a, "hv"))["p"] for a in ARMS if a != MAIN]
    return {
        "reached": rs["runs_reached"], "sooner": rs["fa_sooner_than_both_ablations_seeds"], "sooner_all": rs["fa_sooner_than_every_other_arm_seeds"],
        "clf_gain_feasible": paired_test(per_seed(MAIN, "feasible"), per_seed("fa15q", "feasible"))["diff"],
        "aug_gain_high_margin": paired_test(per_seed(MAIN, "k15"), per_seed("qnc", "k15"))["diff"],
        "hv_p_min": float(min(hv_p)),
        "seed_feasible": int(fa["feasible_lhs"].iloc[0]), "n_aug": int(fa["n_aug"].iloc[0]), "n_opt": int(fa["n_opt"].iloc[0]),
        "aug_feasible": float(fa["feasible_aug"].mean()), "aug_hm": float(fa["k15_aug"].mean()), "bo_feasible": float(fa["feasible_opt"].mean()),
        "feas_ratio_qn": float(fa["feasible"].mean() / PS[PS["arm"] == "qn"]["feasible"].mean()),
        "hm_ratio_qn": float(fa["k15"].mean() / PS[PS["arm"] == "qn"]["k15"].mean()),
    }


def strip(ax, n: dict) -> None:
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, w, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, 3), w, 25, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="0.25", lw=0.7))
        ax.text(x + w / 2, 26.3, title, ha="center", va="top", fontsize=7.2, weight="bold", linespacing=1.15)
        ax.text(x + w / 2, 18.6, body, ha="center", va="top", fontsize=NOTE, linespacing=1.3)

    def arrow(x0, x1, t):
        ax.add_patch(FancyArrowPatch((x0, 15), (x1, 15), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="0.2"))
        ax.text((x0 + x1) / 2, 17.0, t, ha="center", fontsize=NOTE, color="0.25")

    box(0.5, 23.5, f"Phase I\n{N_SEED}-design LHS seed", f"2.4 GHz Class-E PA\n40 nm CMOS, PDK spirals\n{n['seed_feasible']} of {N_SEED} strictly feasible", BOX_COLORS["seed"])
    box(28, 25.5, f"Phase II\nfeasibility augmentation ({n['n_aug']})", f"RF ensembles rank candidates\nby P(feasible) × efficiency\n{n['aug_feasible']:.1f} feasible, {n['aug_hm']:.1f} high-margin", BOX_COLORS["aug"])
    box(57.5, 25.5, f"Phase III\nclassifier-guided MOBO ({n['n_opt']})", f"qLogNEHVI weighted by an\nin-loop feasibility classifier\n{n['bo_feasible']:.1f} of {n['n_opt']} feasible", BOX_COLORS["bo"])
    box(87, 12.5, f"Output ({BUDGET})\nvs qNEHVI", f"{n['feas_ratio_qn']:.1f}× feasible\n{n['hm_ratio_qn']:.1f}× high-margin\nper run", BOX_COLORS["out"])
    arrow(24.9, 27.6, f"{N_SEED}"); arrow(54.4, 57.1, f"{N_SEED + n['n_aug']}"); arrow(83.9, 86.6, f"{BUDGET}")


def yield_bars(ax) -> None:
    arms = list(ARMS); xs = np.arange(len(arms)); w = 0.38; g = PS.groupby("arm")
    f, fs, k, ks = g["feasible"].mean(), g["feasible"].std(ddof=1), g["k15"].mean(), g["k15"].std(ddof=1)
    ax.bar(xs - w / 2, [f[a] for a in arms], w, yerr=[fs[a] for a in arms], color=[col(a) for a in arms], capsize=1.5, error_kw={"lw": 0.6}, label="strictly feasible (arm colour)")
    ax.bar(xs + w / 2, [k[a] for a in arms], w, yerr=[ks[a] for a in arms], color="0.3", capsize=1.5, error_kw={"lw": 0.6}, label=f"high-margin (K ≥ {K_MARGIN:g})")
    for i, a in enumerate(arms):
        ax.text(i - w / 2, f[a] + fs[a] + 0.8, f"{f[a]:.1f}", ha="center", fontsize=NOTE, weight="bold" if a == MAIN else "normal")
        ax.text(i + w / 2, k[a] + ks[a] + 0.8, f"{k[a]:.1f}", ha="center", fontsize=NOTE, weight="bold" if a == MAIN else "normal")
    ax.set_xticks(xs); ax.set_xticklabels([short(a).replace("+", "\n+") for a in arms], fontsize=NOTE)   # 'qNEHVI+m' on two lines: no tick-label collision
    ax.set_ylabel("Designs per run")
    ax.set_ylim(0, max(f + fs) * 1.25); ax.legend(fontsize=NOTE, loc="upper right", handlelength=1.2)
    ax.set_title(f"Designs per run, mean ± s.d., {N_RUNS} seeds")


def main() -> None:
    fig = plt.figure(figsize=(W, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.5], hspace=0.16, wspace=0.42, width_ratios=[1.0, 1.05])
    ax_r = fig.add_subplot(gs[1, 1])
    reach = reach_panel(ax_r, fs=NOTE)
    ax_r.set_xlabel("Harmonic-balance simulations"); ax_r.set_ylabel("Fraction of runs at\nthe joint target")
    ax_r.set_title(f"Joint target: ≥ {JOINT_FEASIBLE} feasible and ≥ {JOINT_NEW_HM} new high-margin design")
    n = headline_numbers(reach)
    ax_s = fig.add_subplot(gs[0, :]); strip(ax_s, n)
    yield_bars(fig.add_subplot(gs[1, 0]))
    r = n["reached"]; p_floor = math.floor(n["hv_p_min"] * 100) / 100
    lines = (f"Joint target in {r['FA-MOBO']}/{N_RUNS} runs "
             f"(Aug-BO {r['Aug-BO']}, Clf-BO {r['Clf-BO']}, qNEHVI {r['qNEHVI']}, qNEHVI+m {r['qNEHVI+m']}), sooner than every other arm on {n['sooner_all']}/{N_RUNS} seeds.",
             f"The classifier supplies the feasibility (+{n['clf_gain_feasible']:.1f} feasible designs per run vs Aug-BO); "
             f"augmentation supplies the margin (+{n['aug_gain_high_margin']:.1f} high-margin designs vs Clf-BO).",
             f"2-D hypervolume: no significant difference from FA-MOBO (Wilcoxon, all p ≥ {p_floor:.2f}).")
    fig.text(0.5, 0.995, "FA-MOBO is the only configuration with high feasibility and high stability margin in every run, and it gets there fastest",
             ha="center", va="top", fontsize=8.6, weight="bold")
    for i, line in enumerate(lines):
        fig.text(0.5, 0.957 - i * 0.031, line, ha="center", va="top", fontsize=7.2, color="0.2")
    fig.subplots_adjust(top=0.875, left=0.075, right=0.87, bottom=0.11)
    pos = ax_s.get_position(); ax_s.set_position([0.02, pos.y0, 0.96, pos.height])   # strip spans the full width
    print({k: v for k, v in n.items()})
    save(fig, "gaabstract")


if __name__ == "__main__":
    main()
