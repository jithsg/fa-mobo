"""Graphical abstract: pipeline, yield bars and a per-run feasible vs new high-margin scatter, in one wide strip.

Applied Soft Computing asks for 531 x 1328 px (h x w) or proportionally more, readable at 5 x 13 cm; the figure is drawn
at 2.5 : 1 and twice the print size, so 11-13 pt text here prints at about 5.5-6.5 pt.
Output: outputs/gaabstract.pdf / .png / _preview.png. Every number is computed from the run histories.
Run from the scripts folder: python make_gabstract.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from common import BOX_COLORS, BUDGET, JOINT_FEASIBLE, JOINT_NEW_HM, K_MARGIN, N_SEED, joint_target_sims, save  # noqa: E402
from make_figures import ARMS, MAIN, N_RUNS, PS, col, per_seed, reach_summary, short  # noqa: E402

GA_W, GA_H = 10.4, 4.16          # 2.5 : 1, twice the 13 x 5 cm print size
TXT, SMALL, TITLE = 12, 10.5, 15
MARKERS = {"fa15qc": "o", "qnc": "s", "fa15q": "D", "qnm": "v", "qn": "^"}   # shape as well as colour per arm
plt.rcParams.update({"font.size": TXT, "axes.titlesize": TXT + 0.5, "axes.labelsize": TXT, "xtick.labelsize": SMALL,
                     "ytick.labelsize": SMALL, "legend.fontsize": SMALL, "axes.linewidth": 0.9})


def headline_numbers(reach: dict) -> dict:
    rs = reach_summary(reach)
    fa = PS[PS["arm"] == MAIN]
    return {
        "reached": rs["runs_reached"],
        "seed_feasible": int(fa["feasible_lhs"].iloc[0]), "n_aug": int(fa["n_aug"].iloc[0]), "n_opt": int(fa["n_opt"].iloc[0]),
        "aug_feasible": float(fa["feasible_aug"].mean()), "aug_hm": float(fa["k15_aug"].mean()), "bo_feasible": float(fa["feasible_opt"].mean()),
        "feas_ratio_qn": float(fa["feasible"].mean() / PS[PS["arm"] == "qn"]["feasible"].mean()),
        "hm_ratio_qn": float(fa["k15"].mean() / PS[PS["arm"] == "qn"]["k15"].mean()),
    }


def pipeline(ax, n: dict) -> None:
    """The three phases and the output as a vertical stack of boxes."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 40); ax.axis("off")
    boxes = [(f"Phase I · {N_SEED}-design LHS seed", f"{n['seed_feasible']} of {N_SEED} feasible", BOX_COLORS["seed"]),
             (f"Phase II · RF augmentation ({n['n_aug']})", f"{n['aug_feasible']:.1f} feasible, {n['aug_hm']:.1f} high-margin", BOX_COLORS["aug"]),
             (f"Phase III · classifier-guided BO ({n['n_opt']})", f"{n['bo_feasible']:.1f} of {n['n_opt']} feasible", BOX_COLORS["bo"]),
             (f"Output ({BUDGET} sims) vs qNEHVI", f"{n['feas_ratio_qn']:.1f}× feasible, {n['hm_ratio_qn']:.1f}× high-margin", BOX_COLORS["out"])]
    h, gap, top = 7.6, 2.2, 39.0
    for i, (head, body, fc) in enumerate(boxes):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.15, y), 9.7, h, boxstyle="round,pad=0.12,rounding_size=0.5", fc=fc, ec="0.3", lw=0.8))
        ax.text(5.0, y + h * 0.66, head, ha="center", va="center", fontsize=SMALL, weight="bold")
        ax.text(5.0, y + h * 0.28, body, ha="center", va="center", fontsize=SMALL)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y - 0.15), (5.0, y - gap + 0.15), arrowstyle="-|>", mutation_scale=11, lw=1.0, color="0.25",
                                         shrinkA=0, shrinkB=0))
    ax.set_title("2.4 GHz Class-E PA, 40 nm CMOS", loc="center")


def yield_bars(ax) -> None:
    arms = list(ARMS); xs = np.arange(len(arms)); w = 0.38; g = PS.groupby("arm")
    f, fs, k, ks = g["feasible"].mean(), g["feasible"].std(ddof=1), g["k15"].mean(), g["k15"].std(ddof=1)
    ax.bar(xs - w / 2, [f[a] for a in arms], w, yerr=[fs[a] for a in arms], color=[col(a) for a in arms], capsize=2, error_kw={"lw": 0.8},
           label="feasible")
    ax.bar(xs + w / 2, [k[a] for a in arms], w, yerr=[ks[a] for a in arms], color="0.3", capsize=2, error_kw={"lw": 0.8},
           label=f"high-margin (K ≥ {K_MARGIN:g})")
    for i, a in enumerate(arms):
        b = a == MAIN
        ax.text(i - w / 2, f[a] + fs[a] + 0.8, f"{f[a]:.1f}", ha="center", fontsize=SMALL - 1, weight="bold" if b else "normal")
        ax.text(i + w / 2, k[a] + ks[a] + 0.8, f"{k[a]:.1f}", ha="center", fontsize=SMALL - 1, weight="bold" if b else "normal")
    ax.set_xticks(xs); ax.set_xticklabels([short(a) for a in arms], rotation=35, ha="right", rotation_mode="anchor")
    for t, a in zip(ax.get_xticklabels(), arms):
        t.set_color(col(a)); t.set_weight("bold" if a == MAIN else "normal")
    ax.set_ylabel("Designs per run")
    ax.set_ylim(0, max(f + fs) * 1.32); ax.legend(loc="upper right", handlelength=1.1, frameon=False, borderaxespad=0.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Mean ± s.d., {N_RUNS} seeds")


def run_scatter(ax) -> dict:
    """One point per run: feasible designs (x) vs new high-margin designs beyond the shared seed (y), joint-target region shaded."""
    arms = list(ARMS); off = dict(zip(arms, np.linspace(0.16, -0.16, len(arms))))   # small fixed row offset per arm: integer counts never hide each other
    xmax = float(PS["feasible"].max()) * 1.08; ymax = float((PS["k15"] - PS["k15_lhs"]).max()) + 0.9
    ax.add_patch(Rectangle((JOINT_FEASIBLE, JOINT_NEW_HM - 0.5), xmax, ymax, facecolor="#EDEDED", edgecolor="none", zorder=0))
    ax.axvline(JOINT_FEASIBLE, color="0.45", lw=0.8, ls="--", zorder=1); ax.axhline(JOINT_NEW_HM - 0.5, color="0.45", lw=0.8, ls="--", zorder=1)
    inside, handles = {}, []
    for a in [x for x in arms if x != MAIN] + [MAIN]:                   # FA-MOBO drawn last (on top)
        f = per_seed(a, "feasible"); h = per_seed(a, "k15") - per_seed(a, "k15_lhs")
        inside[short(a)] = int(np.sum((f >= JOINT_FEASIBLE) & (h >= JOINT_NEW_HM)))
        big = a == MAIN
        ax.scatter(f, h + off[a], s=46 if big else 26, marker=MARKERS[a], facecolors=col(a) if big else "none", edgecolors=col(a),
                   linewidths=0.8 if big else 1.0, zorder=6 if big else 4)
    for a in arms:
        big = a == MAIN
        handles.append(Line2D([], [], ls="", marker=MARKERS[a], markersize=6.5 if big else 5.5, mfc=col(a) if big else "none", mec=col(a),
                              mew=1.0, label=f"{short(a)} {inside[short(a)]}/{N_RUNS}"))
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, ymax); ax.set_yticks(range(int(ymax) + 1))
    ax.set_xlabel("Feasible designs per run"); ax.set_ylabel("New high-margin\ndesigns per run")
    leg = ax.legend(handles=handles, title="runs in the shaded target", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                    handletextpad=0.2, labelspacing=0.45, title_fontsize=SMALL)
    leg.get_texts()[0].set_weight("bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Every run: ≥ {JOINT_FEASIBLE} feasible and ≥ {JOINT_NEW_HM} new high-margin")
    return inside


def main() -> None:
    fig = plt.figure(figsize=(GA_W, GA_H))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.40, left=0.012, right=0.845, top=0.80, bottom=0.20)
    reach = {a: joint_target_sims(a) for a in ARMS}
    inside = run_scatter(fig.add_subplot(gs[0, 2]))
    n = headline_numbers(reach)
    assert inside == n["reached"], (inside, n["reached"])   # end-of-budget region count must equal the joint-target count
    pipeline(fig.add_subplot(gs[0, 0]), n)
    yield_bars(fig.add_subplot(gs[0, 1]))
    fig.text(0.5, 0.975, "FA-MOBO: high feasibility and high stability margin in every run",
             ha="center", va="top", fontsize=TITLE, weight="bold")
    print({k: v for k, v in n.items()})
    plt.rcParams["savefig.bbox"] = "standard"   # keep the exact 2.5 : 1 page; a tight crop would change the aspect ratio
    save(fig, "gaabstract")


if __name__ == "__main__":
    main()
