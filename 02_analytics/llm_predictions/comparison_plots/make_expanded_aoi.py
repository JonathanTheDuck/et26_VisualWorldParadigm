import pandas as pd, numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"

# ---------------------------------------------------------------
# 1. AOI geometry (matches quality_control.py's compute_aoi_boxes_img_px)
# ---------------------------------------------------------------
CANVAS_W, CANVAS_H = 2560, 1440
SCALE = 0.8; SCREEN_W, SCREEN_H = 2560, 1440
X_OFF = (SCREEN_W - CANVAS_W*SCALE)/2; Y_OFF = (SCREEN_H - CANVAS_H*SCALE)/2
def to_img_px(bx, by):
    sx = bx*SCREEN_W; sy = by*SCREEN_H
    return (sx-X_OFF)/SCALE, (sy-Y_OFF)/SCALE

SUBJECT_CENTER = (1200, 400)
SUBJECT_HALF = (420, 300)   # SUBJECT_SIZE/2
OBJ_HALF = 200               # OPTION_SIZE/2
OBJ_CENTERS = {
    "pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3),
    "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6),
}
CORRIDOR_HALF_WIDTH = 150

def in_box(px, py, cx, cy, half_w, half_h=None):
    half_h = half_w if half_h is None else half_h
    return abs(px-cx) <= half_w and abs(py-cy) <= half_h

def classify_expanded(px, py):
    if in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1]):
        return "subject"
    for name, (ox, oy) in OBJ_CENTERS.items():
        if in_box(px, py, ox, oy, OBJ_HALF):
            return name
    # corridor check: perpendicular distance to subject->object segment, nearest wins
    best_name, best_dist = None, None
    sx, sy = SUBJECT_CENTER
    for name, (ox, oy) in OBJ_CENTERS.items():
        vx, vy = ox-sx, oy-sy
        seg_len2 = vx*vx + vy*vy
        t = ((px-sx)*vx + (py-sy)*vy) / seg_len2
        t_clamped = max(0.0, min(1.0, t))
        cx, cy = sx + t_clamped*vx, sy + t_clamped*vy
        dist = math.hypot(px-cx, py-cy)
        if 0.0 <= t <= 1.0 and dist <= CORRIDOR_HALF_WIDTH:
            if best_dist is None or dist < best_dist:
                best_name, best_dist = name, dist
    return best_name if best_name else "elsewhere"

# ---------------------------------------------------------------
# 2. Reclassify every raw critical-window sample
# ---------------------------------------------------------------
raw = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/raw_gaze_critical_window.csv")
fp = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/fixation_proportions.csv")

meta = fp[["subject_nr", "trial", "sentence_id", "target_position"]].drop_duplicates()
raw = raw.merge(meta, on=["subject_nr", "trial"], how="left")
raw = raw.dropna(subset=["sentence_id"])

img_xy = raw.apply(lambda r: to_img_px(r.BPOGX, r.BPOGY), axis=1, result_type="expand")
raw["img_x"], raw["img_y"] = img_xy[0], img_xy[1]
raw["aoi_expanded"] = raw.apply(lambda r: classify_expanded(r.img_x, r.img_y), axis=1)

print("Original AOI counts:\n", raw["aoi"].value_counts())
print("\nExpanded AOI counts:\n", raw["aoi_expanded"].value_counts())

# ---------------------------------------------------------------
# 3. Per-trial proportions with expanded AOI (sample-count based)
# ---------------------------------------------------------------
def trial_props(g):
    n = len(g)
    tgt_pos = f"pos{int(g['target_position'].iloc[0])}"
    counts = g["aoi_expanded"].value_counts()
    out = {f"prop_{p}": counts.get(p, 0)/n for p in ["pos1", "pos2", "pos3", "pos4", "subject"]}
    out["prop_target"] = counts.get(tgt_pos, 0)/n
    obj_counts_by_pos = {p: counts.get(p, 0) for p in ["pos1", "pos2", "pos3", "pos4"]}
    total_obj = sum(obj_counts_by_pos.values())
    out["no_object_gaze"] = total_obj == 0
    out["n_samples"] = n
    if total_obj > 0:
        top_pos = max(obj_counts_by_pos, key=obj_counts_by_pos.get)
        out["human_picks_target"] = (top_pos == tgt_pos)
    else:
        out["human_picks_target"] = np.nan
    return pd.Series(out)

trial_agg = raw.groupby(["subject_nr", "trial", "sentence_id", "condition"]).apply(trial_props, include_groups=False).reset_index()
print(f"\n{len(trial_agg)} trials reclassified; no_object_gaze rate (expanded AOI): {trial_agg.no_object_gaze.mean():.1%}"
      f"  (was {fp.no_object_gaze.mean():.1%} with original AOI)")

item_human = trial_agg.groupby(["sentence_id", "condition"])[["prop_pos1","prop_pos2","prop_pos3","prop_pos4","prop_target"]].mean().reset_index()
item_human.to_csv(f"{OUT_DIR}/item_level_human_expandedAOI.csv", index=False)

# ---------------------------------------------------------------
# 4. Rebuild "LLM vs Human" scatter (matching the teammate's plot style)
# ---------------------------------------------------------------
llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_target = llm[llm.Is_Target].copy()
llm_target["sentence_id"] = llm_target["Item"] - 1
llm_target["cond_key"] = llm_target["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

merged = llm_target.merge(item_human, left_on=["sentence_id", "cond_key"], right_on=["sentence_id", "condition"])
merged = merged.rename(columns={"P_norm": "llm_p", "prop_target": "human_p"})
merged.to_csv(f"{OUT_DIR}/item_level_llm_vs_human_expandedAOI.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
for cond, marker, color, label in [("non-restrictive", "o", "#3f6f9e", "non-restrictive"), ("restrictive", "x", "#4f9e6f", "restrictive")]:
    d = merged[merged.cond_key == cond]
    ax.scatter(d.human_p, d.llm_p, marker=marker, s=70, color=color, alpha=0.85, label=cond,
               linewidth=1.6 if marker == "x" else 0.8, edgecolor=None if marker == "x" else "white")
ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, zorder=0)
r, p = spearmanr(merged.human_p, merged.llm_p)
ax.set_xlabel("Human probability (expanded AOI, includes subject→object corridor)", fontsize=11)
ax.set_ylabel("LLM probability", fontsize=12)
ax.set_title(f"LLM predictions vs. human predictions — expanded AOI\nSpearman ρ = {r:.2f}, p = {p:.3f}  (n={len(merged)})",
             fontsize=13, fontweight="bold")
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#e1e0d9", lw=0.7)
ax.legend(title="condition", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_scatter_expandedAOI.png")
plt.close(fig)
print("saved 4_scatter_expandedAOI.png")

# ---------------------------------------------------------------
# 5. Top-choice (argmax) agreement: is target the #1 pick for LLM / human?
#    LLM: per item x condition (deterministic). Human: per trial (only
#    trials with at least one on-object sample have a defined top pick).
# ---------------------------------------------------------------
llm_top = llm.loc[llm.groupby(["Item", "Condition"])["P_norm"].idxmax()][["Item", "Condition", "Is_Target"]]
llm_top = llm_top.rename(columns={"Is_Target": "llm_picks_target"})
llm_top["sentence_id"] = llm_top["Item"] - 1
llm_top["cond_key"] = llm_top["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

trial_human_top = trial_agg.dropna(subset=["human_picks_target"]).merge(
    llm_top[["sentence_id", "cond_key", "llm_picks_target"]],
    left_on=["sentence_id", "condition"], right_on=["sentence_id", "cond_key"], how="left")

summary = trial_human_top.groupby("condition")[["llm_picks_target", "human_picks_target"]].mean()
n_scored = trial_human_top.groupby("condition").size()
print("\nTop-choice = target, by condition (human: trials with >=1 on-object sample only):\n", summary)
print("n trials scored (human):\n", n_scored)

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
conds = ["restrictive", "non-restrictive"]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [summary.loc[c, "llm_picks_target"] for c in conds], w, label="LLM top choice = target", color="#4f9e6f")
ax.bar(x + w/2, [summary.loc[c, "human_picks_target"] for c in conds], w, label="Human top choice = target (expanded AOI)", color="#3f6f9e")
ax.axhline(0.25, color="#999", ls=":", lw=1.2, label="chance (1 of 4 objects)")
for i, c in enumerate(conds):
    ax.text(i - w/2, summary.loc[c, "llm_picks_target"] + 0.02, f"{summary.loc[c,'llm_picks_target']:.0%}", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + w/2, summary.loc[c, "human_picks_target"] + 0.02, f"{summary.loc[c,'human_picks_target']:.0%}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=11)
ax.set_ylabel("% of items where target is the #1 pick", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title("Is the target the top choice? LLM vs. human (expanded AOI)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", lw=0.7)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_topchoice_comparison.png")
plt.close(fig)
print("saved 5_topchoice_comparison.png")
