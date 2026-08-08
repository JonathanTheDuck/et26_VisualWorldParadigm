import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

COL_R = "#2a78d6"; COL_N = "#e34948"

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"

llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_target = llm[llm.Is_Target].copy()
llm_target["sentence_id"] = llm_target["Item"] - 1
llm_target["cond_key"] = llm_target["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

fp = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/fixation_proportions.csv")
human_item = fp.groupby(["sentence_id", "condition"])["prop_target"].mean().reset_index()
human_n = fp.groupby(["sentence_id", "condition"]).size().reset_index(name="n_trials")

merged = llm_target.merge(human_item, left_on=["sentence_id", "cond_key"], right_on=["sentence_id", "condition"])
merged = merged.merge(human_n, on=["sentence_id", "condition"])
merged = merged.rename(columns={"P_norm": "llm_p_target", "prop_target": "human_prop_target"})
print("merged rows:", len(merged))

# =========================================================
# 1. Item-level scatter: LLM P(target) vs Human prop(target), with correlation
# =========================================================
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
for cond, color, label in [("restrictive", COL_R, "restrictive"), ("non-restrictive", COL_N, "non-restrictive")]:
    d = merged[merged.cond_key == cond]
    ax.scatter(d.llm_p_target, d.human_prop_target, s=60, color=color, alpha=0.75,
               edgecolor="white", linewidth=0.8, label=label, zorder=3)
r_all, p_all = spearmanr(merged.llm_p_target, merged.human_prop_target)
ax.plot([0, 1], [0, 1], "--", color="#c3c2b7", lw=1.2, zorder=1, label="y = x (perfect match)")
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, max(0.5, merged.human_prop_target.max() * 1.15))
ax.set_xlabel("GPT-2: P(target) per item", fontsize=12)
ax.set_ylabel("Human: mean fixation proportion on target per item", fontsize=12)
ax.set_title(f"Item-level agreement (n=50 items × 2 conditions = {len(merged)} points)\nSpearman ρ = {r_all:.2f}, p = {p_all:.3f}",
             fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_scatter_item_agreement.png")
plt.close(fig)
print("saved 1_scatter_item_agreement.png")

# =========================================================
# 2. Headline bar chart: mean target-advantage, LLM vs Human, by condition
# =========================================================
llm_dist = llm[~llm.Is_Target].groupby("Condition")["P_norm"].mean()
llm_tgt = llm[llm.Is_Target].groupby("Condition")["P_norm"].mean()
llm_adv = (llm_tgt - llm_dist).rename({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

fp_valid = fp  # keep zeros (no_object_gaze trials are real 0 observations, not missing)
human_adv = fp_valid.groupby("condition")["target_advantage"].mean()

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
bar_w = 0.35
x = np.arange(2)
ax.bar(x - bar_w/2, [llm_adv["restrictive"], llm_adv["non-restrictive"]], bar_w,
       color=[COL_R, COL_N], alpha=0.9, label="GPT-2 (P_norm target − mean distractor)")
ax.bar(x + bar_w/2, [human_adv["restrictive"], human_adv["non-restrictive"]], bar_w,
       color=[COL_R, COL_N], alpha=0.45, hatch="//", label="Human (fixation prop. target − mean distractor)")
ax.axhline(0, color="#1a1a1a", lw=1)
ax.set_xticks(x); ax.set_xticklabels(["restrictive", "non-restrictive"], fontsize=11)
ax.set_ylabel("Target advantage", fontsize=12)
ax.set_title("Headline comparison: target advantage, GPT-2 vs. human\n(all 50 items pooled)", fontsize=13, fontweight="bold")
for i, cond in enumerate(["restrictive", "non-restrictive"]):
    ax.text(i - bar_w/2, llm_adv[cond] + 0.02, f"{llm_adv[cond]:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + bar_w/2, human_adv[cond] + (0.01 if human_adv[cond]>=0 else -0.03), f"{human_adv[cond]:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_bar_headline_comparison.png")
plt.close(fig)
print("saved 2_bar_headline_comparison.png")

# =========================================================
# 3. Cleveland dot plot: all 50 items ranked by LLM confidence, human alongside
# =========================================================
d_r = merged[merged.cond_key == "restrictive"].sort_values("llm_p_target", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 13), dpi=150)
y = np.arange(len(d_r))
ax.hlines(y, 0, d_r.llm_p_target, color="#e1e0d9", lw=1, zorder=1)
ax.scatter(d_r.llm_p_target, y, s=45, color=COL_R, label="GPT-2 P(target)", zorder=3)
ax.scatter(d_r.human_prop_target, y, s=45, color="#1a1a1a", marker="D", label="Human fixation prop.", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"item {i+1}" for i in d_r["Item"]], fontsize=7)
ax.set_xlabel("Probability / fixation proportion", fontsize=12)
ax.set_title("All 50 items, restrictive condition\nranked by GPT-2 confidence — human data alongside", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(-0.02, 1.05)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_dotplot_all_items_ranked.png")
plt.close(fig)
print("saved 3_dotplot_all_items_ranked.png")

merged.to_csv(f"{OUT_DIR}/item_level_llm_vs_human.csv", index=False)
print("saved item_level_llm_vs_human.csv")
print(f"\nSpearman rho (all): {r_all:.3f}, p={p_all:.4f}")
for cond in ["restrictive", "non-restrictive"]:
    d = merged[merged.cond_key == cond]
    r, p = spearmanr(d.llm_p_target, d.human_prop_target)
    print(f"  {cond}: rho={r:.3f}, p={p:.4f}, n={len(d)}")
