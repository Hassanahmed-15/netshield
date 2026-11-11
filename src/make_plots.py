"""Draw the final report figures from outputs/final_model/tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "outputs" / "final_model"
T, P = BASE / "tables", BASE / "plots"
P.mkdir(exist_ok=True)
BLUE, ORANGE, MUTED, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#b9b7b0", "#0b0b0b", "#52514e", "#e6e4df", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "axes.edgecolor": GRID, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
r = pd.read_csv(T / "results.csv")
few = pd.read_csv(T / "few_labels.csv")


def get(regime, direction, contains):
    s = r[(r.regime == regime) & (r.direction == direction) & r.method.str.contains(contains, regex=False)]
    return s.iloc[0]


def few_mean(direction, n, col):
    return few[(few.direction == direction) & (few.labelled == n)][col].mean()


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bars(ax, names, values, colors, xmax=1.0):
    y = range(len(names))[::-1]
    ax.barh(list(y), values, height=0.34, color=colors)
    ax.set_yticks(list(y), names)
    for yy, v in zip(y, values):
        ax.text(v + 0.012 * xmax, yy, f"{v:.2f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, xmax * 1.08)
    style(ax)


# Figure 1: ranking quality of each approach
fig, axes = plt.subplots(2, 1, figsize=(9, 6.2))
a = [("Earlier RF", get("baseline", "UNSW->CICIDS", "RF 11").roc_auc, MUTED),
     ("Earlier CORAL", get("baseline", "UNSW->CICIDS", "CORAL").roc_auc, MUTED),
     ("New CORAL, corrected features (no labels)", get("transfer_selected", "UNSW->CICIDS", "CORAL").roc_auc, BLUE),
     ("Full UNSW capture (no labels, not selected)", get("transfer_alternative", "UNSW->CICIDS", "same").roc_auc, MUTED),
     ("New + 500 labelled CICIDS flows", few_mean("UNSW->CICIDS", 500, "roc_auc"), BLUE)]
b = [("Earlier RF", get("baseline", "CICIDS->UNSW", "RF 11").roc_auc, MUTED),
     ("Earlier CORAL", get("baseline", "CICIDS->UNSW", "CORAL").roc_auc, MUTED),
     ("New CORAL, corrected features (no labels)", get("transfer_selected", "CICIDS->UNSW", "official").roc_auc, BLUE),
     ("New + 500 labelled UNSW flows", few_mean("CICIDS->UNSW", 500, "roc_auc"), BLUE)]
for ax, data, title in zip(axes, (a, b), ("Train on UNSW, test on CICIDS (Thursday and Friday)", "Train on CICIDS, test on UNSW (official test file)")):
    bars(ax, [x[0] for x in data], [x[1] for x in data], [x[2] for x in data])
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=8)
axes[0].text(0.51, 3.5, "coin flip", color=INK2, fontsize=9, va="center")
axes[1].set_xlabel("ROC-AUC on the frozen test data")
fig.tight_layout()
fig.savefig(P / "01_method_comparison_roc.png", dpi=170)

# Figure 2: labels needed
fig, ax = plt.subplots(figsize=(7.6, 4.6))
xs = [0, 1, 2, 3]
ns = [100, 500, 2000]
for d, col, sel, lab in [("UNSW->CICIDS", BLUE, get("transfer_selected", "UNSW->CICIDS", "CORAL").f1, "Train UNSW, test CICIDS"),
                         ("CICIDS->UNSW", ORANGE, get("transfer_selected", "CICIDS->UNSW", "official").f1, "Train CICIDS, test UNSW")]:
    v = [sel] + [few_mean(d, n, "f1") for n in ns]
    ax.plot(xs, v, color=col, lw=2, marker="o", ms=8, mec=BG, mew=2, label=lab)
    ax.annotate(f"{v[-1]:.2f}", (3, v[-1]), xytext=(9, 8 if d == "UNSW->CICIDS" else -8), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.annotate(f"{v[0]:.2f}", (0, v[0]), xytext=(-9, 0), textcoords="offset points", va="center", ha="right", color=INK, fontsize=10)
ax.set_xticks(xs, ["0", "100", "500", "2000"])
ax.set_xlim(-0.35, 3.3); ax.set_ylim(0, 1.02)
ax.set_xlabel("Labelled flows from the test dataset")
ax.set_ylabel("F1 (mean of 3 repeats)")
ax.set_title("A small labelled sample closes most of the gap", loc="left", fontsize=12, pad=8)
ax.legend(frameon=False, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(length=0); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(P / "02_labels_needed_f1.png", dpi=170)

# Figure 3: attack families
f = pd.read_csv(T / "transfer_by_attack_family.csv")
f = f[f.method == "selected"]
f["family"] = f.family.str.replace(r"Web Attack \S+ ", "Web: ", regex=True)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
for ax, (d, ev, title) in zip(axes, [("UNSW->CICIDS", "all frozen days", "Train UNSW, test CICIDS"), ("CICIDS->UNSW", "official file", "Train CICIDS, test UNSW")]):
    s = f[(f.direction == d) & (f["eval"] == ev)].sort_values("recall", ascending=False)
    bars(ax, [f"{a} ({n:,})" for a, n in zip(s.family, s.n)], s.recall.tolist(), [BLUE] * len(s))
    ax.set_title(title, loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Share of attack flows caught (no labels)")
fig.tight_layout()
fig.savefig(P / "03_recall_by_attack_family.png", dpi=170)
