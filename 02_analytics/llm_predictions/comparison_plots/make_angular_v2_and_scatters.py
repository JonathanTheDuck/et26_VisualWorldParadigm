import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import spearmanr

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
COL_R = "#2a78d6"; COL_N = "#e34948"

CANVAS_W, CANVAS_H = 2400, 1400
SCALE = 0.8; SCREEN_W, SCREEN_H = 2560, 1440
X_OFF = (SCREEN_W - CANVAS_W*SCALE)/2; Y_OFF = (SCREEN_H - CANVAS_H*SCALE)/2
def to_img_px(bx, by):
    sx = bx*SCREEN_W; sy = by*SCREEN_H
    return (sx-X_OFF)/SCALE, (sy-Y_OFF)/SCALE

SUBJECT_CENTER = (1200, 400); SUBJECT_HALF = (420, 300); OBJ_HALF = 200
OBJ_CENTERS = {"pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3), "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6)}
OBJ_ANGLES = {n: math.atan2(cy-SUBJECT_CENTER[1], cx-SUBJECT_CENTER[0]) for n, (cx, cy) in OBJ_CENTERS.items()}

# The 4 objects sit 50 degrees apart (15,65,115,165 -- a 150-degree semicircle).
# The old "nearest angle, unbounded" version let the two EDGE objects (pos1,
# pos4) claim the entire empty 210-degree arc behind the subject, giving them
# a much bigger wedge than pos2/pos3. Fix: cap every wedge at the true local
# half-gap (25 degrees either side of the object's own angle) so all 4
# wedges are equal-sized; anything beyond that (the empty arc) is "elsewhere",
# same as it would be in the original AOI.
WEDGE_HALF = math.radians(25)

def in_box(px, py, cx, cy, hw, hh=None):
    hh = hw if hh is None else hh
    return (np.abs(px-cx) <= hw) & (np.abs(py-cy) <= hh)

def angdiff(a, b):
    d = (a-b) % (2*math.pi)
    return np.minimum(d, 2*math.pi-d)

def classify_angular_v2(px, py):
    if in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1]):
        return "subject"
    for n, (ox, oy) in OBJ_CENTERS.items():
        if in_box(px, py, ox, oy, OBJ_HALF):
            return n
    ang = math.atan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
    diffs = {n: angdiff(ang, a) for n, a in OBJ_ANGLES.items()}
    best = min(diffs, key=diffs.get)
    return best if diffs[best] <= WEDGE_HALF else "elsewhere"

# =========================================================
# 1. Recompute human proportions with the FIXED angular AOI
# =========================================================
raw = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/raw_gaze_critical_window.csv")
fp = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/fixation_proportions.csv")
meta = fp[["subject_nr", "trial", "sentence_id", "target_position"]].drop_duplicates()
raw = raw.merge(meta, on=["subject_nr", "trial"], how="left").dropna(subset=["sentence_id"])
xy = raw.apply(lambda r: to_img_px(r.BPOGX, r.BPOGY), axis=1, result_type="expand")
raw["img_x"], raw["img_y"] = xy[0], xy[1]
raw["aoi_angular_v2"] = raw.apply(lambda r: classify_angular_v2(r.img_x, r.img_y), axis=1)
print("Angular AOI v2 (equal-width wedges) counts:\n", raw["aoi_angular_v2"].value_counts())

def trial_props(g):
    n = len(g)
    tgt_pos = f"pos{int(g['target_position'].iloc[0])}"
    counts = g["aoi_angular_v2"].value_counts()
    out = {f"prop_{p}": counts.get(p, 0)/n for p in ["pos1","pos2","pos3","pos4","subject"]}
    out["prop_target"] = counts.get(tgt_pos, 0)/n
    total_obj = sum(counts.get(p, 0) for p in ["pos1","pos2","pos3","pos4"])
    out["no_object_gaze"] = total_obj == 0
    return pd.Series(out)

trial_agg = raw.groupby(["subject_nr","trial","sentence_id","condition"]).apply(trial_props, include_groups=False).reset_index()
print(f"\nno_object_gaze rate (angular v2, equal wedges): {trial_agg.no_object_gaze.mean():.1%}"
      f"  (angular v1 unequal wedges: 28.6%, corridor: 56.3%, original: 79.4%)")

item_human_v2 = trial_agg.groupby(["sentence_id","condition"])[["prop_pos1","prop_pos2","prop_pos3","prop_pos4","prop_target"]].mean().reset_index()
item_human_v2.to_csv(f"{OUT_DIR}/item_level_human_angularAOI_v2.csv", index=False)
print("saved item_level_human_angularAOI_v2.csv")

# =========================================================
# 2. Redo the illustration (panel 3 only, equal wedges now)
# =========================================================
IMG_PATH = "/Users/ladidadida2025/et26_VisualWorldParadigm/01_experiment/stimuli/img_composition/1_pos1_sub.png"
img = Image.open(IMG_PATH)
IMG_W, IMG_H = img.size
OBJ_NAMES = {"pos1": "cake", "pos2": "toy car", "pos3": "toy train", "pos4": "ball"}
OBJ_COLOR = {"pos1": "#2a78d6", "pos2": "#e34948", "pos3": "#1baf7a", "pos4": "#eda100"}
SUBJ_COLOR = "#6b7280"

gx, gy = np.meshgrid(np.linspace(0, IMG_W, 480), np.linspace(0, IMG_H, 280))
px, py = gx.ravel(), gy.ravel()
subj_mask = in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1])
ang = np.arctan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
names = list(OBJ_CENTERS.keys())
diffs = np.stack([angdiff(ang, OBJ_ANGLES[n]) for n in names], axis=0)
nearest_idx = np.argmin(diffs, axis=0)
nearest_diff = np.min(diffs, axis=0)
nearest = np.array(names)[nearest_idx]
within = nearest_diff <= WEDGE_HALF
amask = {n: (~subj_mask) & within & (nearest == n) for n in names}
for n, (ox, oy) in OBJ_CENTERS.items():
    amask[n] = amask[n] | in_box(px, py, ox, oy, OBJ_HALF)

fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
ax.imshow(img)
overlay = np.zeros((*gx.shape, 4))
for n in OBJ_CENTERS:
    rgb = matplotlib.colors.to_rgb(OBJ_COLOR[n])
    overlay[amask[n].reshape(gx.shape)] = (*rgb, 0.4)
overlay[subj_mask.reshape(gx.shape)] = (*matplotlib.colors.to_rgb(SUBJ_COLOR), 0.3)
ax.imshow(overlay, extent=(0, IMG_W, IMG_H, 0))
ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.axis("off")
ax.set_title("3. Angular AOI v2\n(equal ±25° wedges — fixes the uneven areas)", fontsize=13, fontweight="bold")
handles = [patches.Patch(color=SUBJ_COLOR, alpha=0.4, label="subject")]
handles += [patches.Patch(color=OBJ_COLOR[n], alpha=0.55, label=OBJ_NAMES[n]) for n in OBJ_CENTERS]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{OUT_DIR}/13b_angular_AOI_v2_fixed.png", bbox_inches="tight")
plt.close(fig)

total = px.shape[0]
cov = subj_mask.mean() + sum(m.mean() for m in amask.values())
print(f"\nAngular v2 screen coverage: {cov:.1%}  (v1 was 100% by construction; v2 leaves the empty top arc as 'elsewhere', same as intent)")
print("saved 13b_angular_AOI_v2_fixed.png")

# =========================================================
# 3. Scatter plots, human on X axis, for method 2 (corridor) and method 3 (angular v2)
# =========================================================
llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_target = llm[llm.Is_Target].copy()
llm_target["sentence_id"] = llm_target["Item"] - 1
llm_target["cond_key"] = llm_target["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

def make_scatter(human_csv, title_suffix, out_name):
    human = pd.read_csv(human_csv)
    merged = llm_target.merge(human, left_on=["sentence_id", "cond_key"], right_on=["sentence_id", "condition"])
    merged = merged.rename(columns={"P_norm": "llm_p", "prop_target": "human_p"})
    r, p = spearmanr(merged.llm_p, merged.human_p)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    for cond, color, label in [("restrictive", COL_R, "restrictive"), ("non-restrictive", COL_N, "non-restrictive")]:
        d = merged[merged.cond_key == cond]
        ax.scatter(d.human_p, d.llm_p, s=65, color=color, alpha=0.8, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#c3c2b7", lw=1.2, zorder=1, label="y = x (perfect match)")
    ax.set_ylim(-0.03, 1.03); ax.set_xlim(-0.03, max(0.5, merged.human_p.max()*1.15))
    ax.set_ylabel("GPT-2: P(target) per item", fontsize=12)
    ax.set_xlabel(f"Human: mean fixation proportion on target per item\n({title_suffix})", fontsize=11)
    ax.set_title(f"Item-level agreement — {title_suffix} (n={len(merged)})\nSpearman ρ = {r:.2f}, p = {p:.3f}",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e1e0d9", lw=0.7, zorder=0)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{out_name}")
    plt.close(fig)
    print(f"saved {out_name}  (rho={r:.3f}, p={p:.4f}, n={len(merged)})")

make_scatter(f"{OUT_DIR}/item_level_human_expandedAOI.csv", "corridor AOI, 150px", "15_scatter_corridorAOI_humanX.png")
make_scatter(f"{OUT_DIR}/item_level_human_angularAOI_v2.csv", "angular AOI v2, equal wedges", "16_scatter_angularAOI_v2_humanX.png")
