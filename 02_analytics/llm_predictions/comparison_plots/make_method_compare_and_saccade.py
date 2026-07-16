import pandas as pd, numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
COL_LLM = "#4f9e6f"; COL_HUM = "#3f6f9e"; CHANCE = 0.25

# =========================================================
# Part A: AOI-method comparison chart
# =========================================================
methods = ["Original AOI", "Corridor\n(150px)", "Angular\n(direction)"]
no_obj_rate = [0.794, 0.563, 0.286]
top_choice = [0.436, 0.436, 0.433]
n_scored = [55, 55, 90]
ci = [proportion_confint(round(tc*n), n, method="wilson") for tc, n in zip(top_choice, n_scored)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)

bars = ax1.bar(methods, no_obj_rate, color="#c0392b", alpha=0.85, width=0.55)
for b, v in zip(bars, no_obj_rate):
    ax1.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.set_ylabel("no_object_gaze rate", fontsize=11)
ax1.set_title("Coverage: how often is gaze\nclassified to NO object at all", fontsize=12, fontweight="bold")
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)

x = np.arange(3)
ax2.bar(x, top_choice, color=COL_HUM, alpha=0.85, width=0.55,
        yerr=[[tc-lo for tc, (lo, hi) in zip(top_choice, ci)], [hi-tc for tc, (lo, hi) in zip(top_choice, ci)]],
        capsize=6, ecolor="#1a1a1a")
ax2.axhline(CHANCE, color="#999", ls=":", lw=1.2)
ax2.text(2.5, CHANCE+0.015, "chance (25%)", fontsize=8.5, color="#777", ha="right")
for i, (tc, n) in enumerate(zip(top_choice, n_scored)):
    ax2.text(i, ci[i][1]+0.03, f"{tc:.0%}\n(n={n})", ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(methods)
ax2.set_ylim(0, 0.75)
ax2.set_ylabel("% trials: target = top choice (restrictive)", fontsize=11)
ax2.set_title("The finding itself: stable across methods,\nCI narrows as more trials become scoreable", fontsize=12, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)

fig.suptitle("Comparing three AOI definitions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/9_aoi_method_comparison.png", bbox_inches="tight")
plt.close(fig)
print("saved 9_aoi_method_comparison.png")

# =========================================================
# Part B: First-saccade direction analysis
# =========================================================
CANVAS_W, CANVAS_H = 2400, 1400
SCALE = 0.8; SCREEN_W, SCREEN_H = 2560, 1440
X_OFF = (SCREEN_W - CANVAS_W*SCALE)/2; Y_OFF = (SCREEN_H - CANVAS_H*SCALE)/2
def to_img_px(bx, by):
    sx = bx*SCREEN_W; sy = by*SCREEN_H
    return (sx-X_OFF)/SCALE, (sy-Y_OFF)/SCALE

SUBJECT_CENTER = (1200, 400); SUBJECT_HALF = (420, 300); OBJ_HALF = 200
OBJ_CENTERS = {"pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3), "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6)}
OBJ_ANGLES = {n: math.atan2(cy-SUBJECT_CENTER[1], cx-SUBJECT_CENTER[0]) for n, (cx, cy) in OBJ_CENTERS.items()}

def in_box(px, py, cx, cy, hw, hh=None):
    hh = hw if hh is None else hh
    return abs(px-cx) <= hw and abs(py-cy) <= hh

def angdiff(a, b):
    d = (a-b) % (2*math.pi)
    return min(d, 2*math.pi-d)

def classify_angular(px, py):
    if in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1]):
        return "subject"
    for n, (ox, oy) in OBJ_CENTERS.items():
        if in_box(px, py, ox, oy, OBJ_HALF):
            return n
    ang = math.atan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
    return min(OBJ_ANGLES, key=lambda n: angdiff(ang, OBJ_ANGLES[n]))

raw = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/raw_gaze_critical_window.csv")
fp = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/fixation_proportions.csv")
meta = fp[["subject_nr", "trial", "sentence_id", "target_position"]].drop_duplicates()
raw = raw.merge(meta, on=["subject_nr", "trial"], how="left").dropna(subset=["sentence_id"])
xy = raw.apply(lambda r: to_img_px(r.BPOGX, r.BPOGY), axis=1, result_type="expand")
raw["img_x"], raw["img_y"] = xy[0], xy[1]
raw["aoi_angular"] = raw.apply(lambda r: classify_angular(r.img_x, r.img_y), axis=1)
raw = raw.sort_values(["subject_nr", "trial", "TIME"])

def first_saccade(g):
    tgt_pos = f"pos{int(g['target_position'].iloc[0])}"
    outward = g[g["aoi_angular"] != "subject"]
    if len(outward) == 0:
        return pd.Series({"has_first_saccade": False, "first_toward_target": np.nan})
    first = outward.iloc[0]
    return pd.Series({"has_first_saccade": True, "first_toward_target": first["aoi_angular"] == tgt_pos})

sacc = raw.groupby(["subject_nr", "trial", "sentence_id", "condition"]).apply(first_saccade, include_groups=False).reset_index()
print(f"\nTrials with >=1 sample outside the subject box (i.e. a scoreable 'first movement'): "
      f"{sacc.has_first_saccade.mean():.1%} of {len(sacc)}")

scored = sacc.dropna(subset=["first_toward_target"])
summary_sacc = scored.groupby("condition")["first_toward_target"].agg(["mean", "count"])
print("\nFirst movement out of the subject region -- is it toward the target?\n", summary_sacc)

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
conds = ["restrictive", "non-restrictive"]
vals = [summary_sacc.loc[c, "mean"] for c in conds]
ns = [int(summary_sacc.loc[c, "count"]) for c in conds]
cis = [proportion_confint(round(v*n), n, method="wilson") for v, n in zip(vals, ns)]
bars = ax.bar(conds, vals, color=[COL_HUM, "#8ea6bd"], width=0.5,
              yerr=[[v-lo for v, (lo, hi) in zip(vals, cis)], [hi-v for v, (lo, hi) in zip(vals, cis)]],
              capsize=6, ecolor="#1a1a1a")
ax.axhline(CHANCE, color="#999", ls=":", lw=1.2)
ax.text(1.35, CHANCE+0.015, "chance (25%)", fontsize=8.5, color="#777", ha="right")
for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(i, cis[i][1]+0.03, f"{v:.0%}\n(n={n})", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.6)
ax.set_ylabel("% of trials where the first movement\nout of the subject region is toward the target", fontsize=10.5)
ax.set_title("First-saccade direction: is the very first look\ntoward the target, or somewhere else?", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_first_saccade_direction.png")
plt.close(fig)
print("saved 10_first_saccade_direction.png")

sacc.to_csv(f"{OUT_DIR}/first_saccade_by_trial.csv", index=False)
print("saved first_saccade_by_trial.csv")
