"""Manuscript figures for the FA-MOBO revision (10 paired seeds, 181 simulations per run), drawn at print width 7.0 in.

Figures: fig_reliability (lead), fig_progression, fig_ablation, fig_yield_tradeoff, fig_mechanism,
fig_corner_sweep (Fig. 1, fig_architecture, is drawn by make_fig_architecture.py). Every number drawn on a figure is computed here from the run histories and also written to
out/fig_numbers.json. Run from the scripts folder: python make_figures.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from common import (ARMS, BOX_COLORS, EA_COLOR, EA_METHODS, joint_target_index, load_ea, BUDGET, DESIGN_VARS, FS, HM_SLACK, JOINT_FEASIBLE, JOINT_NEW_HM, K_BANDS, K_MARGIN, N_CONSTRAINTS, N_SEED,
                    OP1DB_MARGIN, ORIGIN_LABELS, ORIGIN_STYLE, ORIGINS, OUT, PHASE_COLORS, Q_BATCH, SEEDS, W_MIN_UM, beeswarm,
                    bound_share_thirds, hv_margin, joint_target_sims, load_hist, margin_rich, paired_test, per_seed_table,
                    pooled_designs, reach_fraction, save, spread, style)
from problem import feasible_mask  # noqa: E402

style()
W = 7.0                                                    # print width (in)
MAIN = "fa15qc"
DRAW_ORDER = [a for a in ARMS if a != MAIN] + [MAIN]       # FA-MOBO drawn last (on top)
N_RUNS = len(SEEDS)
PS = per_seed_table()
PS["hv_m"] = [hv_margin(load_hist(a, s)) for a, s in zip(PS["arm"], PS["seed"])]
FIGNUM: dict = {}                                          # every number printed on a figure
NOTE = FS["note"]                                          # smallest text size on any figure (7 pt; prints at ~6.7 pt at 468 pt text width)
ORIGIN_SHORT = {"seed": "shared LHS seed", "aug": "augmentation phase", "bo_clf": "BO with classifier", "bo_plain": "BO without classifier"}
K_TICKS = [5, 7.5, 10, 15, 20, 30]                         # nominal-K ticks shared by the design-level panels


# ----------------------------------------------------------------------------- small helpers
def short(arm: str) -> str:
    return ARMS[arm][0]


def col(arm: str) -> str:
    return ARMS[arm][2]


def lkw(arm: str) -> dict:
    """Line style of an arm (FA-MOBO thick and on top)."""
    _, _, c, ls, lw = ARMS[arm]
    return {"color": c, "ls": ls, "lw": lw, "zorder": 6 if arm == MAIN else 4}


def per_seed(arm: str, met: str) -> np.ndarray:
    return PS[PS["arm"] == arm].set_index("seed")[met].loc[SEEDS].to_numpy(float)


def log_k_axis(ax, lo: float, hi: float, ticks) -> None:
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def origin_scatter(ax, F: pd.DataFrame, x: str, y: str, faint: bool = False) -> dict:
    """Designs coloured and marked by origin; augmentation and seed designs drawn on top. Returns counts per origin."""
    counts = {}
    for o in ("bo_plain", "bo_clf", "aug", "seed"):
        g = F[F["origin"] == o]; st = dict(ORIGIN_STYLE[o])
        if faint and o.startswith("bo"):
            st["alpha"] = st["alpha"] * 0.55
        ax.scatter(g[x], g[y], **st)
        counts[o] = int(len(g))
    return counts


def origin_handles(counts: dict) -> list:
    hs = []
    for o in ORIGINS:
        st = ORIGIN_STYLE[o]
        hs.append(Line2D([], [], ls="", marker=st["marker"], markersize=np.sqrt(st["s"]) * 1.15, mfc=st["facecolors"],
                         mec=st["edgecolors"] if st["edgecolors"] != "none" else st["facecolors"], mew=max(st["linewidths"], 0.5),
                         label=f"{ORIGIN_SHORT[o]} ({counts[o]})"))
    return hs


def direct_labels(ax, x: float, values: dict, fmt, min_gap: float, lo: float, hi: float, fs: float = 7.0) -> None:
    """Arm labels right of the axes, pushed apart vertically so they never overlap."""
    arms = list(values)
    ys = spread([values[a] for a in arms], min_gap, lo, hi)
    for a, y in zip(arms, ys):
        ax.text(x, y, fmt(a), color=col(a), fontsize=fs, va="center", ha="left", clip_on=False,
                weight="bold" if a == MAIN else "normal")


# ----------------------------------------------------------------------------- Fig. reliability (lead)
def reach_panel(ax, fs: float = 7.0) -> dict:
    """Step curves: fraction of runs that have reached the joint target by each simulation index."""
    sims = np.arange(N_SEED, BUDGET + 1)
    reach = {a: joint_target_sims(a) for a in ARMS}
    for arm in DRAW_ORDER:
        ax.step(sims, reach_fraction(reach[arm], sims), where="post", **lkw(arm))
    ax.set_xlim(N_SEED, BUDGET); ax.set_ylim(-0.04, 1.06); ax.set_xticks([100, 120, 140, 160, 181])
    ends = {a: int(np.isfinite(reach[a]).sum()) for a in ARMS}
    direct_labels(ax, BUDGET + 2, {a: ends[a] / N_RUNS for a in ARMS}, lambda a: f"{short(a)} {ends[a]}/{N_RUNS}", 0.09, -0.02, 1.04, fs)
    return reach


def reach_summary(reach: dict) -> dict:
    """Counts, medians and per-seed 'sooner than both ablations' (unreached = later than any reached run)."""
    capped = {a: np.where(np.isfinite(r), r, np.inf) for a, r in reach.items()}
    sooner = int(np.sum((capped[MAIN] < capped["fa15q"]) & (capped[MAIN] < capped["qnc"])))
    sooner_all = int(np.sum(np.all([capped[MAIN] < capped[a] for a in capped if a != MAIN], axis=0)))
    return {"per_seed": {short(a): [None if not np.isfinite(v) else int(v) for v in r] for a, r in reach.items()},
            "runs_reached": {short(a): int(np.isfinite(r).sum()) for a, r in reach.items()},
            "median_sim_reached": {short(a): (float(np.nanmedian(r)) if np.isfinite(r).any() else None) for a, r in reach.items()},
            "fa_sooner_than_both_ablations_seeds": sooner, "fa_sooner_than_every_other_arm_seeds": sooner_all}


def margin_map_panel(ax, D: pd.DataFrame) -> None:
    F = D[D["feas"]]
    ax.add_patch(Rectangle((K_MARGIN, HM_SLACK), 100, 10, facecolor="#EDEDED", edgecolor="none", zorder=0))
    ax.axvline(K_MARGIN, color="0.45", lw=0.6, ls="--", zorder=1); ax.axhline(HM_SLACK, color="0.45", lw=0.6, ls="--", zorder=1)
    counts = origin_scatter(ax, F, "minKf", "slack")
    kmax = float(F["minKf"].max()); smax = float(F["slack"].max())
    log_k_axis(ax, K_BANDS[0], K_BANDS[-2], K_TICKS); ax.set_ylim(-0.03, smax * 1.80)
    hm = F[F["high_margin"]]; by = hm["origin"].value_counts()
    parts = {"seed": "seed", "aug": "augmentation", "bo_clf": "BO with classifier", "bo_plain": "BO without classifier"}
    body = "\n".join(f"{int(by.get(o, 0))} {parts[o]}" for o in ORIGINS if by.get(o, 0))
    ax.text(K_BANDS[-2] * 0.97, smax * 1.76, f"high-margin (shaded):\nK ≥ {K_MARGIN:g}, margin ≥ {HM_SLACK:.1f} dB\n{len(hm)} designs: {body}",
            ha="right", va="top", fontsize=NOTE, color="0.15", linespacing=1.12, zorder=8)
    ax.legend(handles=origin_handles(counts), loc="upper left", fontsize=NOTE, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.22,
              title=f"{len(F)} distinct feasible designs", title_fontsize=NOTE, alignment="left")
    ax.set_xlabel("Nominal stability score K"); ax.set_ylabel("Output-power margin (dB)")
    ax.set_title("(b) Margin map of the returned designs")
    FIGNUM["reliability_margin_map"] = {"n_feasible_distinct": int(len(F)), "by_origin": counts, "n_high_margin": int(len(hm)),
                                        "high_margin_by_origin": {o: int(by.get(o, 0)) for o in ORIGINS}, "max_K": kmax, "max_slack": smax}


def fig_reliability() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.85), gridspec_kw={"width_ratios": [1.0, 1.08]})
    ax = axes[0]
    reach = reach_panel(ax)
    ea_reach = {m: joint_target_index(load_ea(m)) for m in EA_METHODS}
    hits = sorted({r for r in ea_reach.values() if np.isfinite(r)})
    for v in hits:
        ax.plot([v], [1.0], marker="v", ms=5.5, color=EA_COLOR, mec="white", mew=0.4, ls="none", zorder=8, clip_on=False)
    # key in the empty upper-left corner (no curve rises before the first reach at ~117): marker + text, clear of every step curve
    lines = [", ".join(m for m, r in ea_reach.items() if r == v) + f"\nat sim. {v:.0f};" for v in hits]   # short lines: right edge stays left of the FA-MOBO steps
    lines += [f"{m} not reached" for m, r in ea_reach.items() if not np.isfinite(r)]
    if hits:
        ax.plot([101.8], [0.955], marker="v", ms=5.5, color=EA_COLOR, mec="white", mew=0.4, ls="none", zorder=8)
        ax.text(103.6, 0.985, "single runs:\n" + "\n".join(lines), ha="left", va="top", fontsize=NOTE, color=EA_COLOR, linespacing=1.12, zorder=8)
    FIGNUM["reliability_ea_reach"] = {m: (None if not np.isfinite(r) else int(r)) for m, r in ea_reach.items()}
    ax.text(0.97, 0.47, f"joint target:\n≥ {JOINT_FEASIBLE} strictly feasible designs\nand ≥ {JOINT_NEW_HM} new high-margin design",
            transform=ax.transAxes, ha="right", va="top", fontsize=NOTE, color="0.3", linespacing=1.15)
    ax.set_xlabel("Circuit simulations"); ax.set_ylabel(f"Fraction of the {N_RUNS} runs\nthat reached the joint target")
    ax.set_title("(a) Runs reaching the joint target")
    FIGNUM["reliability_reach"] = reach_summary(reach)
    margin_map_panel(axes[1], pooled_designs())
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_reliability")


# ----------------------------------------------------------------------------- Fig. progression
def phase_marks(ax, n_aug: int) -> None:
    for xv in (N_SEED, N_SEED + n_aug):
        ax.axvline(xv, color="0.55", lw=0.6, ls=":", zorder=0)
    y1 = ax.get_ylim()[1]
    ax.text(N_SEED + n_aug / 2, y1, "aug.", fontsize=NOTE, color="0.35", va="top", ha="center")
    ax.text((N_SEED + n_aug + BUDGET) / 2, y1, "BO phase", fontsize=NOTE, color="0.35", va="top", ha="center")


def progress_panel(ax, curves: dict, ylabel: str, title: str) -> None:
    x = np.arange(1, BUDGET + 1)
    for arm in DRAW_ORDER:
        Y = curves[arm]; m, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.plot(x, m, label=short(arm), **lkw(arm))
        if arm in (MAIN, "qn"):
            ax.fill_between(x, m - sd, m + sd, color=col(arm), alpha=0.13, lw=0, zorder=1)
    ax.set_xlim(N_SEED - 4, BUDGET); ax.set_xticks([100, 120, 140, 160, 181]); ax.set_xlabel("Circuit simulations")
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.10 * (hi - lo))
    ax.set_ylabel(ylabel); ax.set_title(title)


def hidden_until(curves: dict, top: str, under: str) -> int:
    """First simulation index at which the mean curve of `under` differs from that of `top` (BUDGET if never)."""
    differ = np.flatnonzero(~np.isclose(curves[top].mean(0), curves[under].mean(0)))
    return int(differ[0]) + 1 if len(differ) else BUDGET


def overlap_note(ax, curves: dict) -> None:
    """Say where a mean curve is hidden under another: Aug-BO under FA-MOBO (same augmentation batch), qNEHVI+m under qNEHVI."""
    until = hidden_until(curves, MAIN, "fa15q"); until_m = hidden_until(curves, "qn", "qnm")
    ax.text(0.98, 0.30, f"{short('fa15q')} (dash-dot) is\nhidden by {short(MAIN)}\nuntil simulation {until}", transform=ax.transAxes,   # 3 short lines: clear of the dotted line at 115
            ha="right", va="bottom", fontsize=NOTE, color=col("fa15q"))
    ax.text(0.98, 0.03, f"{short('qnm')} (dotted) is\nhidden by {short('qn')}\nuntil simulation {until_m}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=NOTE, color=col("qn"))
    FIGNUM["progression_aug_bo_under_fa_mobo_until"] = until
    FIGNUM["progression_qnehvi_m_under_qnehvi_until"] = until_m


def fig_progression() -> None:
    n_aug = int(PS[PS["arm"] == MAIN]["n_aug"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.7))
    seed_hm = int(PS[PS["arm"] == MAIN]["k15_lhs"].iloc[0])
    metrics = (("hv", lambda h: h["cum_hv"].to_numpy(), "Hypervolume of the\n(PAE, Psat) front", "(a) Hypervolume"),
               ("feasible", lambda h: np.cumsum(feasible_mask(h).to_numpy()), "Strictly feasible designs", "(b) Strictly feasible designs"),
               ("k15", lambda h: np.cumsum(margin_rich(h).to_numpy()), f"High-margin designs\n(incl. {seed_hm} from the shared seed)", "(c) High-margin designs"))
    for ax, (key, fn, ylab, title) in zip(axes, metrics):
        curves = {a: np.array([fn(load_hist(a, s)) for s in SEEDS]) for a in ARMS}
        for a in ARMS:  # the cumulative curves must end at the per-run totals used everywhere else
            assert np.allclose(curves[a][:, -1], per_seed(a, key)), (key, a)
        progress_panel(ax, curves, ylab, title)
        phase_marks(ax, n_aug)
        FIGNUM[f"progression_{key}_at_{BUDGET}"] = {short(a): float(curves[a][:, -1].mean()) for a in ARMS}
        if key == "k15":
            ax.set_ylim(-0.2, ax.get_ylim()[1])                           # room for the hidden-curve note below the 2-design lines
            overlap_note(ax, curves)
    handles = [Line2D([], [], color=col(a), ls=ARMS[a][3], lw=ARMS[a][4], label=short(a)) for a in ARMS]
    handles.append(Patch(facecolor="0.5", alpha=0.25, lw=0, label="± s.d. (FA-MOBO, qNEHVI)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), handlelength=2.2, fontsize=NOTE, columnspacing=1.4)
    FIGNUM["progression_seed_high_margin"] = seed_hm
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92))   # top strip reserved for the shared legend
    save(fig, "fig_progression")


# ----------------------------------------------------------------------------- Fig. ablation
def fig_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
    others = [a for a in ARMS if a != MAIN]
    specs = (("feasible", "strictly feasible designs", "(a) Strictly feasible designs per seed"),
             ("k15", "high-margin designs", "(b) High-margin designs per seed"))
    out = {}
    for ax, (met, lab, title) in zip(axes, specs):
        for i, a in enumerate(others):
            diff = per_seed(MAIN, met) - per_seed(a, met); t = paired_test(per_seed(MAIN, met), per_seed(a, met))
            ax.scatter(i + beeswarm(diff, step=1.0), diff, s=16, facecolor="none", edgecolor=col(a), lw=0.9, zorder=3)
            ax.hlines(diff.mean(), i - 0.32, i + 0.32, color="k", lw=1.4, zorder=4)
            wins = f"{t['wins']}/{N_RUNS} wins" if t["ties"] == 0 else f"{t['wins']} win{'s' * (t['wins'] != 1)}, {t['ties']} ties"
            ax.annotate(f"p = {t['p']:.3f}\n{wins}", (i, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=NOTE, color="0.25")
            out[f"{met}_vs_{short(a)}"] = t
        ax.axhline(0, color="0.5", lw=0.6, ls="--")
        ax.set_xticks(range(len(others))); ax.set_xticklabels([f"vs {short(a)}" for a in others]); ax.set_xlim(-0.6, len(others) - 0.4)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * 0.40)
        ax.set_ylabel(f"FA-MOBO minus other arm,\n{lab}"); ax.set_title(title)
        ax.set_xlabel("Comparison arm (difference > 0 favours FA-MOBO)")
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", label="one seed"),
                            Line2D([], [], color="k", lw=1.4, label=f"mean of {N_RUNS} seeds"),
                            Line2D([], [], color="0.5", lw=0.6, ls="--", label="no difference")], loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=NOTE)
    FIGNUM["ablation"] = out
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig_ablation")


# ----------------------------------------------------------------------------- Fig. yield / hypervolume
def phase_origin_panel(ax) -> None:
    arms = list(ARMS); xs = np.arange(len(arms)); g = PS.groupby("arm")
    seed_part = g["k15_lhs"].mean(); aug = g["k15_aug"].mean(); opt = g["k15_opt"].mean(); tot = g["k15"].mean()
    ax.bar(xs, [seed_part[a] for a in arms], color=PHASE_COLORS["seed"], width=0.62, label="shared LHS seed")
    ax.bar(xs, [aug[a] for a in arms], bottom=[seed_part[a] for a in arms], color=PHASE_COLORS["aug"], width=0.62, label="augmentation phase")
    ax.bar(xs, [opt[a] for a in arms], bottom=[seed_part[a] + aug[a] for a in arms], color=PHASE_COLORS["bo"], width=0.62, label="BO phase")
    for i, a in enumerate(arms):
        v = per_seed(a, "k15")
        ax.scatter(i + beeswarm(v, step=1.0), v, s=13, facecolor="none", edgecolor="0.25", lw=0.7, zorder=3)
        ax.text(i, max(tot[a], v.max()) + 0.35, f"{tot[a]:.1f}", ha="center", fontsize=7.2, weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([short(a) for a in arms]); ax.set_ylabel("High-margin designs per run"); ax.set_ylim(0, 10.8); ax.set_yticks(range(0, 11, 2))
    ax.set_title("(a) Where high-margin designs come from")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=NOTE, handlelength=1.2, title="high-margin designs from", title_fontsize=NOTE)
    ax.text(0.02, 0.985, f"bars: mean of {N_RUNS} seeds\ncircles: seeds", transform=ax.transAxes, fontsize=NOTE, color="0.35", va="top")
    FIGNUM["yield_phase_origin"] = {short(a): {"total": float(tot[a]), "seed": float(seed_part[a]), "aug": float(aug[a]), "bo": float(opt[a])} for a in arms}


def hv_dot_panel(ax, met: str, xlabel: str, show_names: bool) -> dict:
    """Horizontal dot plot: one row per arm, seeds as circles, mean as a bar with its value, p against FA-MOBO at the right."""
    arms = list(ARMS); res = {}
    vals = np.concatenate([per_seed(a, met) for a in arms]); span = np.ptp(vals)
    lo, hi = vals.min() - 0.10 * span, vals.max() + 0.10 * span
    for y, a in enumerate(arms):
        v = per_seed(a, met); m = v.mean()
        ax.scatter(v, y + beeswarm(v, width=0.28, step=span / 30), s=11, facecolor="none", edgecolor=col(a), lw=0.75, zorder=3)
        ax.vlines(m, y - 0.34, y + 0.34, color="k", lw=1.6, zorder=4)
        entry = {"mean": float(m)}
        label = f"{m:.0f}"
        if a != MAIN:
            t = paired_test(per_seed(MAIN, met), v); entry.update(t)
            label += f"  (p = {t['p']:.3f})"
        frac = (m - lo) / (hi - lo)
        ha = "left" if frac < 0.3 else ("right" if frac > 0.7 else "center")
        ax.text(m, y - 0.37, label, ha=ha, va="bottom", fontsize=NOTE, zorder=5, weight="bold" if a == MAIN else "normal")
        res[short(a)] = entry
    ax.set_xlim(lo, hi); ax.set_ylim(len(arms) - 0.45, -0.75)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([short(a) for a in arms] if show_names else [])
    if show_names:  # arm names on the rows; FA-MOBO bold as everywhere else
        for t, a in zip(ax.get_yticklabels(), arms):
            t.set_weight("bold" if a == MAIN else "normal")
    ax.tick_params(axis="y", length=0); ax.set_xlabel(xlabel)
    for y in range(len(arms)):
        ax.axhline(y, color="0.92", lw=0.5, zorder=0)
    return res


def fig_yield_tradeoff() -> None:
    fig = plt.figure(figsize=(W, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 0.78, 0.78])
    phase_origin_panel(fig.add_subplot(gs[0, 0]))
    ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])   # no sharey: it hid the arm names of ax2
    r2 = hv_dot_panel(ax2, "hv", "2-D hypervolume\n(PAE, Psat)", True)
    r3 = hv_dot_panel(ax3, "hv_m", "Margin-aware hypervolume\n(PAE, Psat, K)", False)
    ax2.set_title("(b) 2-D vs margin-aware hypervolume (p vs FA-MOBO)")
    FIGNUM["yield_hv2"] = r2; FIGNUM["yield_hv_margin"] = r3
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_yield_tradeoff")


# ----------------------------------------------------------------------------- Fig. mechanism
def bound_panel(ax) -> None:
    xs = np.arange(1, 4); offs = dict(zip(ARMS, np.linspace(-0.14, 0.14, len(ARMS)))); res = {}
    for arm in DRAW_ORDER:
        v = bound_share_thirds(arm) * 100; m, sd = v.mean(0), v.std(0, ddof=1)
        yerr = np.vstack([np.minimum(sd, m), np.minimum(sd, 100 - m)])   # s.d. whiskers clipped to the 0-100 % range
        ax.errorbar(xs + offs[arm], m, yerr=yerr, color=col(arm), ls=ARMS[arm][3], lw=ARMS[arm][4] + (0.3 if arm == MAIN else 0), marker="o",
                    ms=4.6 if arm == MAIN else 3.4, mec="white", mew=0.4, elinewidth=0.5, ecolor=matplotlib.colors.to_rgba(col(arm), 0.45), capsize=0, zorder=6 if arm == MAIN else 4)
        res[short(arm)] = {"mean_pct": [float(x) for x in m], "sd_pct": [float(x) for x in sd]}
    ends = {a: res[short(a)]["mean_pct"][2] for a in ARMS}
    direct_labels(ax, 3.32, ends, lambda a: f"{short(a)} {res[short(a)]['mean_pct'][0]:.0f} → {res[short(a)]['mean_pct'][2]:.0f} %", 9.0, 0, 100, NOTE)
    ax.set_xticks(xs); ax.set_xticklabels(["first", "middle", "last"]); ax.set_xlim(0.7, 3.25); ax.set_ylim(-3, 113); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Third of the BO phase"); ax.set_ylabel(f"BO proposals at the minimum\ntransistor width, {W_MIN_UM:.0f} µm (%)")
    ax.set_title("(a) Drift to the minimum transistor width")
    ax.text(0.03, 0.97, f"mean ± s.d. of {N_RUNS} seeds (clipped at 0 and 100 %)", transform=ax.transAxes, fontsize=NOTE, color="0.35", va="top")
    FIGNUM["mechanism_bound_share"] = res


def band_panel(ax, D: pd.DataFrame) -> None:
    F = D[D["feas"]]
    counts = origin_scatter(ax, F, "minKf", "PAE_percent", faint=True)
    xmax = K_BANDS[-2]; rows = []
    for lo, hi in zip(K_BANDS[:-1], K_BANDS[1:]):
        g = F[(F["minKf"] >= lo) & (F["minKf"] < hi)]
        rows.append({"band": f"[{lo:g}, {hi:g})", "n": int(len(g)), "best_pae": float(g["PAE_percent"].max()) if len(g) else None})
        if g.empty:
            continue
        best = float(g["PAE_percent"].max()); xr = min(hi, xmax)
        ax.hlines(best, lo, xr, color="k", lw=1.5, zorder=7)
        ax.text(np.sqrt(lo * xr), best + 0.5, f"{best:.1f}", ha="center", va="bottom", fontsize=NOTE, zorder=8)
    log_k_axis(ax, K_BANDS[0], xmax, K_TICKS)
    ymin = float(F["PAE_percent"].min()); ymax = max(r["best_pae"] for r in rows if r["best_pae"] is not None)
    ax.set_ylim(ymin - 1.0, ymax + 11.0); ax.set_yticks(np.arange(5 * np.ceil(ymin / 5), 5 * np.ceil(ymax / 5) + 0.1, 5))
    ax.vlines(K_MARGIN, ymin - 1.0, 5 * np.ceil(ymax / 5), color="0.45", lw=0.6, ls="--", zorder=1)   # K = 15 guide, below the legend
    hs = [Line2D([], [], color="k", lw=1.5, label="best PAE in the K band")] + origin_handles(counts)
    leg = ax.legend(handles=hs, loc="upper center", ncol=2, fontsize=NOTE, handlelength=1.2, handletextpad=0.5, labelspacing=0.2, borderaxespad=0.2, columnspacing=0.8)   # line handle clear of its text and of the y spine
    ax.set_xlabel("Nominal stability score K"); ax.set_ylabel("PAE (%)")
    ax.set_title("(b) Best PAE per stability-margin band")
    FIGNUM["mechanism_k_bands"] = {"rows": rows, "n_feasible_distinct": int(len(F)), "by_origin": counts}


def fig_mechanism() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [1.0, 1.12]})
    bound_panel(axes[0])
    band_panel(axes[1], pooled_designs())
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_mechanism")


# ----------------------------------------------------------------------------- Fig. corner sweep (unchanged content)
def _ktrans(k):
    k = np.asarray(k, float); return np.sign(k) * np.log10(1 + np.abs(k))


def fig_corner_sweep() -> None:
    from make_tables import corner_designs  # same distinct-design rule as the corner table
    _, per_arm, pooled = corner_designs()
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for arm, mk in (("fa15qc", "o"), ("qn", "s")):
        rows = per_arm[per_arm.arm == arm]
        g = rows.groupby("sig"); des = pd.DataFrame({"tt_K": g.tt_K.first(), "stable_all": g.minKf.min() >= 1})
        kff = rows[rows.corner == "FF/-40C"].set_index("sig").minKf.loc[des.index]
        for stable, fc in ((True, ARMS[arm][2]), (False, "white")):
            m = des.stable_all == stable
            ax.scatter(des.tt_K[m], _ktrans(kff[m]), marker=mk, s=22, facecolor=fc, edgecolor=ARMS[arm][2], lw=0.9, zorder=3)
    ticks = [-10, -1, 0, 1, 10]; ax.set_yticks(_ktrans(ticks)); ax.set_yticklabels([f"{t:g}".replace("-", "−") for t in ticks]); ax.set_ylim(_ktrans(-25), _ktrans(13))
    ax.axhline(_ktrans(1), color="k", lw=0.7, ls="--"); ax.text(26.3, _ktrans(1.15), "K = 1 (stability limit)", fontsize=NOTE, ha="right", va="bottom")
    ax.axvline(K_MARGIN, color="0.4", lw=0.7, ls=":"); ax.text(K_MARGIN + 0.3, _ktrans(-24), f"K = {K_MARGIN:.0f}", fontsize=NOTE, color="0.3", va="bottom")
    g = pooled.groupby("sig"); des = pd.DataFrame({"tt_K": g.tt_K.first(), "stable_all": g.minKf.min() >= 1})
    hi = des[des.tt_K >= K_MARGIN]; lo = des[des.tt_K < K_MARGIN]
    ax.text(0.98, 0.62, f"{len(des)} distinct designs\nnominal K ≥ {K_MARGIN:.0f}: {int(hi.stable_all.sum())} / {len(hi)} stable in every scenario\nnominal K < {K_MARGIN:.0f}: {int(lo.stable_all.sum())} / {len(lo)} ({lo.stable_all.mean() * 100:.0f} %)\nno design with nominal K in {lo.tt_K.max():.1f}–{hi.tt_K.min():.1f}",
            transform=ax.transAxes, fontsize=NOTE, color="0.2", ha="right", va="top")   # top below the K = 1 line (axes 0.67), clear of its label
    ax.set_xlabel("Nominal stability score K"); ax.set_ylabel("K in the hardest scenario,\nFF −40 °C (log-compressed)"); ax.set_title("(a) Stability in the hardest scenario")
    handles = [Line2D([], [], marker="o", ls="", mfc=ARMS["fa15qc"][2], mec=ARMS["fa15qc"][2], label="FA-MOBO design"), Line2D([], [], marker="s", ls="", mfc=ARMS["qn"][2], mec=ARMS["qn"][2], label="qNEHVI design"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", label="hollow: unstable in some scenario")]
    ax.legend(handles=handles, loc="lower right", fontsize=NOTE, handletextpad=0.4)
    ax = axes[1]; corners = ["TT/27C", "FF/27C", "SS/27C", "FF/-40C", "SS/85C"]; xs = np.arange(len(corners)); w = 0.36
    n = pooled.sig.nunique(); shares = {}
    for k, (cname, lab, colr) in enumerate((("stable", "stable (K ≥ 1)", "0.72"), ("pass_strict", "all six requirements met", "0.25"))):
        vals = [pooled[pooled.corner == c][cname].mean() * 100 for c in corners]; shares[cname] = dict(zip(corners, vals))
        bars = ax.bar(xs + (k - 0.5) * w, vals, width=w, color=colr, lw=0, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=NOTE, color="0.25")
    ax.set_xticks(xs); ax.set_xticklabels([c.replace("C", " °C").replace("-", "−").replace("/", "\n") for c in corners])
    ax.set_ylabel(f"Share of the {n} distinct designs (%)"); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_title("(b) Outcome per operating scenario")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=NOTE)
    FIGNUM["corner_sweep"] = {"n_distinct": int(len(des)), "hi_stable": int(hi.stable_all.sum()), "hi_n": int(len(hi)), "lo_stable": int(lo.stable_all.sum()),
                              "lo_n": int(len(lo)), "k_gap": [float(lo.tt_K.max()), float(hi.tt_K.min())], "corner_share_pct": shares}
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_corner_sweep")


if __name__ == "__main__":
    fig_reliability(); fig_progression(); fig_ablation(); fig_yield_tradeoff(); fig_mechanism(); fig_corner_sweep()
    PS.drop(columns="hv_m").to_csv(OUT / "per_seed_paper.csv", index=False); print("per_seed_paper.csv written")
    (OUT / "fig_numbers.json").write_text(json.dumps(FIGNUM, indent=1)); print("fig_numbers.json written")
