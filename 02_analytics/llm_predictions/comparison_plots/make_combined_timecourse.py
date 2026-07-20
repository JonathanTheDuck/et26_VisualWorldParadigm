import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_R = "#2a78d6"; COL_N = "#e34948"; COL_WINDOW = "#eda100"

llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_agg = llm.groupby(["Condition", "Is_Target"])["P_norm"].mean().unstack()
p_target_r = llm_agg.loc["Restrictive", True]
p_target_n = llm_agg.loc["Non-restr.", True]
p_dist_r = llm_agg.loc["Restrictive", False]
p_dist_n = llm_agg.loc["Non-restr.", False]

gc = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/growth_curves.csv")
gc = gc[gc.aoi.str.startswith("pos")].copy()
gc["is_target"] = gc.aoi == ("pos" + gc.target_pos.astype(str))
agg = gc.groupby(["condition", "bin_start_ms", "is_target"])["prop_fixating"].mean().unstack()
support = gc.drop_duplicates(["condition", "bin_start_ms", "trial", "subject_nr"]).groupby(["condition", "bin_start_ms"]).size()

RELIABLE_MS = 300  # support stays >=~35 trials up to here; craters after

fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=150)

ax.axvspan(0, RELIABLE_MS, color="#f1f3f9", zorder=0)
ax.text(RELIABLE_MS/2, 0.97, "dense data\n(≥35 trials/bin)", ha="center", va="top", fontsize=8.5, color="#6b7280", style="italic")
ax.text((RELIABLE_MS + 900)/2, 0.97, "sparse tail (≤11 trials/bin) — shown faded", ha="center", va="top", fontsize=8.5, color="#b0aca0", style="italic")

styles = [
    ("restrictive", COL_R, "restrictive"),
    ("non-restrictive", COL_N, "non-restrictive"),
]
human_handles = []
for cond, color, label in styles:
    d = agg.loc[cond].reset_index()
    dense = d[d.bin_start_ms <= RELIABLE_MS]
    sparse = d[d.bin_start_ms >= RELIABLE_MS]
    h1, = ax.plot(dense.bin_start_ms, dense[True], "-o", color=color, lw=2.2, ms=5, label=f"{label} — target (human)", zorder=5)
    h2, = ax.plot(dense.bin_start_ms, dense[False], "--o", color=color, lw=1.6, ms=4, mfc="white", label=f"{label} — distractor (human)", zorder=4, alpha=0.85)
    human_handles += [h1, h2]
    ax.plot(sparse.bin_start_ms, sparse[True], "-o", color=color, lw=2.2, ms=5, alpha=0.28, zorder=3)
    ax.plot(sparse.bin_start_ms, sparse[False], "--o", color=color, lw=1.6, ms=4, mfc="white", alpha=0.22, zorder=2)

# LLM reference lines (averaged across all 50 items)
llm_handles = []
for val, color, style, lab in [
    (p_target_r, COL_R, "-", f"GPT-2 P(target | restrictive) = {p_target_r:.2f}"),
    (p_target_n, COL_N, "-", f"GPT-2 P(target | non-restr.) = {p_target_n:.2f}"),
    (p_dist_r, COL_R, ":", f"GPT-2 P(distractor | restrictive) = {p_dist_r:.2f}"),
    (p_dist_n, COL_N, ":", f"GPT-2 P(distractor | non-restr.) = {p_dist_n:.2f}"),
]:
    ln = ax.axhline(val, color=color, lw=1.5, linestyle=style, alpha=0.55, zorder=1, label=lab)
    llm_handles.append(ln)
    ax.text(905, val, f"{val:.2f}", color=color, fontsize=9, fontweight="bold", va="center")

ax.set_xlim(0, 970); ax.set_ylim(0, 1.02)
ax.set_xlabel("Time from verb offset (ms)", fontsize=12)
ax.set_ylabel("Probability  /  fixation proportion", fontsize=12)
ax.set_title("All 50 items × 5 participants — human gaze vs. GPT-2 prediction", fontsize=15, fontweight="bold", pad=34)
fig.text(0.5, 0.925, "solid = target, dashed/dotted = mean distractor   ·   lines = human (binned)   ·   flat refs = GPT-2 (item-averaged)",
          ha="center", fontsize=10, color="#52514e")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", lw=0.8, zorder=0)
human_legend = ax.legend(handles=human_handles, loc="upper left", fontsize=9, framealpha=0.9, ncol=1,
                          title="Human (binned, this study)", title_fontsize=9)
ax.add_artist(human_legend)
ax.legend(handles=llm_handles, loc="lower center", bbox_to_anchor=(0.68, 0.45),
          fontsize=9, framealpha=0.9, title="GPT-2 (item-averaged)", title_fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_timecourse_50items.png")
plt.savefig(OUT)
print("saved", OUT)
print(f"p_target_r={p_target_r:.3f} p_target_n={p_target_n:.3f} p_dist_r={p_dist_r:.3f} p_dist_n={p_dist_n:.3f}")
