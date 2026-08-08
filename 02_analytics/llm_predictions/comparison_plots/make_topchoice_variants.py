import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
COL_LLM = "#4f9e6f"; COL_HUM = "#3f6f9e"; CHANCE = 0.25

# from the expanded-AOI top-choice analysis
data = {
    "restrictive":     {"llm": 0.854545, "human": 0.436364, "n_human": 55},
    "non-restrictive": {"llm": 0.384615, "human": 0.288462, "n_human": 52},
}
for k, d in data.items():
    lo, hi = proportion_confint(round(d["human"]*d["n_human"]), d["n_human"], method="wilson")
    d["human_lo"], d["human_hi"] = lo, hi

conds = ["restrictive", "non-restrictive"]

# =========================================================
# 2. Dumbbell chart (with Wilson CI on the human point)
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
y = np.arange(len(conds))
close_pair = abs(data["non-restrictive"]["llm"] - data["non-restrictive"]["human"]) < 0.12
for i, c in enumerate(conds):
    d = data[c]
    ax.plot([d["human"], d["llm"]], [i, i], color="#c3c2b7", lw=2, zorder=1)
    ax.errorbar(d["human"], i, xerr=[[d["human"]-d["human_lo"]], [d["human_hi"]-d["human"]]],
                fmt="none", ecolor=COL_HUM, elinewidth=1.5, capsize=4, zorder=2)
    ax.scatter(d["human"], i, s=140, color=COL_HUM, zorder=3, label="Human" if i == 0 else None)
    ax.scatter(d["llm"], i, s=140, color=COL_LLM, zorder=3, label="LLM" if i == 0 else None)
    if c == "non-restrictive" and close_pair:
        # stack labels vertically to avoid horizontal collision on a close pair
        ax.text(d["llm"], i+0.18, f"{d['llm']:.0%}", va="bottom", ha="center", fontsize=10, fontweight="bold", color=COL_LLM)
        ax.text(d["human"], i-0.18, f"{d['human']:.0%}", va="top", ha="center", fontsize=10, fontweight="bold", color=COL_HUM)
    else:
        ax.text(d["llm"]+0.02, i, f"{d['llm']:.0%}", va="center", fontsize=10, fontweight="bold", color=COL_LLM)
        ax.text(d["human_lo"]-0.02, i, f"{d['human']:.0%}", va="center", ha="right", fontsize=10, fontweight="bold", color=COL_HUM)
ax.axvline(CHANCE, color="#999", ls=":", lw=1.2, zorder=0)
ax.text(CHANCE, -0.42, "chance (25%)", fontsize=8.5, color="#777", ha="center", va="top")
ax.set_yticks(y); ax.set_yticklabels(conds, fontsize=11)
ax.set_ylim(-0.5, len(conds)-0.5)
ax.set_xlim(-0.02, 1.0)
ax.set_xlabel("% of trials where target is the top choice", fontsize=11)
ax.set_title("Gap between LLM and human — top choice = target\n(human error bars: 95% Wilson CI)", fontsize=13, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=10, loc="center right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_dumbbell_topchoice.png")
plt.close(fig)
print("saved 6_dumbbell_topchoice.png")

# =========================================================
# 3. Waffle / pictogram (10x10 grid) per condition x source
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=150)
for row, c in enumerate(conds):
    for col, (src, color) in enumerate([("human", COL_HUM), ("llm", COL_LLM)]):
        ax = axes[row, col]
        pct = data[c][src]
        n_filled = round(pct*100)
        grid = np.zeros((10, 10))
        flat = grid.flatten()
        flat[:n_filled] = 1
        grid = flat.reshape(10, 10)[::-1]
        for yy in range(10):
            for xx in range(10):
                filled = grid[yy, xx] == 1
                sq = plt.Rectangle((xx, yy), 0.86, 0.86,
                                    facecolor=color if filled else "#eceae4",
                                    edgecolor="white", linewidth=1.2)
                ax.add_patch(sq)
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.axis("off")
        label = "Human" if src == "human" else "GPT-2"
        ax.set_title(f"{c}\n{label}: {pct:.0%} pick target", fontsize=11, fontweight="bold")
fig.suptitle("Out of 100 trials/items, how many have the target as the top choice?", fontsize=13, fontweight="bold", y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT_DIR}/7_waffle_topchoice.png")
plt.close(fig)
print("saved 7_waffle_topchoice.png")

# =========================================================
# 4. Deviation-from-chance diverging bar
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
labels = []
vals = []
colors = []
for c in conds:
    labels += [f"LLM\n{c}", f"Human\n{c}"]
    vals += [data[c]["llm"]-CHANCE, data[c]["human"]-CHANCE]
    colors += [COL_LLM, COL_HUM]
y = np.arange(len(labels))
bars = ax.barh(y, vals, color=colors, height=0.6)
ax.axvline(0, color="#1a1a1a", lw=1)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
for yi, v in zip(y, vals):
    ax.text(v + (0.015 if v >= 0 else -0.015), yi, f"{v:+.0%}", va="center",
            ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold")
ax.set_xlabel("Percentage points above/below chance (25%)", fontsize=11)
ax.set_title("How far above chance is each source at picking the target?", fontsize=13, fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", color="#e1e0d9", lw=0.7, zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/8_deviation_from_chance.png")
plt.close(fig)
print("saved 8_deviation_from_chance.png")
