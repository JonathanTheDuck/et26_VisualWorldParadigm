import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
COL_R = "#2a78d6"; COL_N = "#e34948"

llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_target = llm[llm.Is_Target].copy()
llm_target["sentence_id"] = llm_target["Item"] - 1
llm_target["cond_key"] = llm_target["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

human = pd.read_csv(f"{OUT_DIR}/item_level_human_angularAOI.csv")

merged = llm_target.merge(human, left_on=["sentence_id", "cond_key"], right_on=["sentence_id", "condition"])
merged = merged.rename(columns={"P_norm": "llm_p", "prop_target": "human_p"})
merged.to_csv(f"{OUT_DIR}/item_level_llm_vs_human_angularAOI.csv", index=False)
print("n =", len(merged))

fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
for cond, color, label in [("restrictive", COL_R, "restrictive"), ("non-restrictive", COL_N, "non-restrictive")]:
    d = merged[merged.cond_key == cond]
    ax.scatter(d.llm_p, d.human_p, s=65, color=color, alpha=0.8, edgecolor="white", linewidth=0.8, label=label, zorder=3)
r, p = spearmanr(merged.llm_p, merged.human_p)
ax.plot([0, 1], [0, 1], "--", color="#c3c2b7", lw=1.2, zorder=1, label="y = x (perfect match)")
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, max(0.5, merged.human_p.max()*1.15))
ax.set_xlabel("GPT-2: P(target) per item", fontsize=12)
ax.set_ylabel("Human: mean fixation proportion on target per item\n(angular AOI)", fontsize=11)
ax.set_title(f"Item-level agreement — angular AOI (n={len(merged)} item×condition points)\nSpearman ρ = {r:.2f}, p = {p:.3f}",
             fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/11_scatter_angularAOI.png")
plt.close(fig)
print("saved 11_scatter_angularAOI.png")

for cond in ["restrictive", "non-restrictive"]:
    d = merged[merged.cond_key == cond]
    r, p = spearmanr(d.llm_p, d.human_p)
    print(f"  {cond}: rho={r:.3f}, p={p:.4f}, n={len(d)}")
