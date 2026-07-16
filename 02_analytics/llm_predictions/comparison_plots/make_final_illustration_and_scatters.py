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

IMG_PATH = "/Users/ladidadida2025/et26_VisualWorldParadigm/01_experiment/stimuli/img_composition/1_pos1_sub.png"
img = Image.open(IMG_PATH)
IMG_W, IMG_H = img.size

SUBJECT_CENTER = (1200, 400); SUBJECT_HALF = (420, 300); OBJ_HALF = 200
OBJ_CENTERS = {"pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3), "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6)}
OBJ_NAMES = {"pos1": "cake", "pos2": "toy car", "pos3": "toy train", "pos4": "ball"}
OBJ_COLOR = {"pos1": "#2a78d6", "pos2": "#e34948", "pos3": "#1baf7a", "pos4": "#eda100"}
SUBJ_COLOR = "#6b7280"
CORRIDOR_HALF_WIDTH = 150
WEDGE_HALF = math.radians(25)
OBJ_ANGLES = {n: math.atan2(cy-SUBJECT_CENTER[1], cx-SUBJECT_CENTER[0]) for n, (cx, cy) in OBJ_CENTERS.items()}

def in_box(px, py, cx, cy, hw, hh=None):
    hh = hw if hh is None else hh
    return (np.abs(px-cx) <= hw) & (np.abs(py-cy) <= hh)

def angdiff(a, b):
    d = (a-b) % (2*math.pi)
    return np.minimum(d, 2*math.pi-d)

gx, gy = np.meshgrid(np.linspace(0, IMG_W, 480), np.linspace(0, IMG_H, 280))
px, py = gx.ravel(), gy.ravel()
SUBJ_MASK = in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1])

def original_mask():
    m = {n: in_box(px, py, ox, oy, OBJ_HALF) for n, (ox, oy) in OBJ_CENTERS.items()}
    return SUBJ_MASK, m

def corridor_mask():
    m = {}
    sx, sy = SUBJECT_CENTER
    for n, (ox, oy) in OBJ_CENTERS.items():
        box = in_box(px, py, ox, oy, OBJ_HALF)
        vx, vy = ox-sx, oy-sy
        seg_len2 = vx*vx+vy*vy
        t = ((px-sx)*vx + (py-sy)*vy) / seg_len2
        tc = np.clip(t, 0, 1)
        cx_, cy_ = sx+tc*vx, sy+tc*vy
        dist = np.hypot(px-cx_, py-cy_)
        corridor = (t >= 0) & (t <= 1) & (dist <= CORRIDOR_HALF_WIDTH)
        m[n] = box | corridor
    return SUBJ_MASK, m

def angular_v2_mask():
    ang = np.arctan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
    names = list(OBJ_CENTERS.keys())
    diffs = np.stack([angdiff(ang, OBJ_ANGLES[n]) for n in names], axis=0)
    nearest_idx = np.argmin(diffs, axis=0)
    nearest_diff = np.min(diffs, axis=0)
    nearest = np.array(names)[nearest_idx]
    within = nearest_diff <= WEDGE_HALF
    m = {n: (~SUBJ_MASK) & within & (nearest == n) for n in names}
    for n, (ox, oy) in OBJ_CENTERS.items():
        m[n] = m[n] | in_box(px, py, ox, oy, OBJ_HALF)
    return SUBJ_MASK, m

def coverage(subj_mask, m):
    return subj_mask.mean() + sum(v.mean() for v in m.values())

def draw_panel(ax, subj_mask, m, title, note):
    ax.imshow(img)
    overlay = np.zeros((*gx.shape, 4))
    for n in OBJ_CENTERS:
        rgb = matplotlib.colors.to_rgb(OBJ_COLOR[n])
        overlay[m[n].reshape(gx.shape)] = (*rgb, 0.4)
    overlay[subj_mask.reshape(gx.shape)] = (*matplotlib.colors.to_rgb(SUBJ_COLOR), 0.3)
    ax.imshow(overlay, extent=(0, IMG_W, IMG_H, 0))
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    cov = coverage(subj_mask, m)
    ax.text(0.5, -0.045, f"{note}   |   screen coverage: {cov:.0%}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color="#333")

fig, axes = plt.subplots(1, 3, figsize=(19, 7.3), dpi=150)

s1, m1 = original_mask()
draw_panel(axes[0], s1, m1, "1. Original AOI\n(object box only, 200px half-width)",
           "no free parameter — box = actual object size")

s2, m2 = corridor_mask()
draw_panel(axes[1], s2, m2, "2. Corridor AOI\n(+ subject→object strip)",
           "corridor half-width = 150px")

s3, m3 = angular_v2_mask()
draw_panel(axes[2], s3, m3, "3. Angular AOI v2 — fixed\n(equal wedges, direction from subject)",
           "wedge = ±25° (equal for all 4 objects)")

handles = [patches.Patch(color=SUBJ_COLOR, alpha=0.4, label="subject")]
handles += [patches.Patch(color=OBJ_COLOR[n], alpha=0.55, label=OBJ_NAMES[n]) for n in OBJ_CENTERS]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("How much did the AOI actually grow? (item 1, restrictive)\nAngular method now uses equal ±25° wedges — v1's unbounded version gave the two edge objects (cake, ball) ~3× more area",
             fontsize=14.5, fontweight="bold", y=1.05)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{OUT_DIR}/17_aoi_expansion_illustration_v2.png", bbox_inches="tight")
plt.close(fig)
print("saved 17_aoi_expansion_illustration_v2.png")
print(f"coverage: original={coverage(s1,m1):.1%}  corridor={coverage(s2,m2):.1%}  angular_v2={coverage(s3,m3):.1%}")

# =========================================================
# Combined scatter: 3 methods side by side, human on x-axis
# =========================================================
llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_target = llm[llm.Is_Target].copy()
llm_target["sentence_id"] = llm_target["Item"] - 1
llm_target["cond_key"] = llm_target["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

fp = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/fixation_proportions.csv")
human_orig = fp.groupby(["sentence_id", "condition"])["prop_target"].mean().reset_index()

human_corridor = pd.read_csv(f"{OUT_DIR}/item_level_human_expandedAOI.csv")[["sentence_id", "condition", "prop_target"]]
human_angular_v2 = pd.read_csv(f"{OUT_DIR}/item_level_human_angularAOI_v2.csv")[["sentence_id", "condition", "prop_target"]]

methods = [
    ("1. Original AOI", human_orig, "#6b7280"),
    ("2. Corridor AOI", human_corridor, "#1baf7a"),
    ("3. Angular AOI v2 (fixed)", human_angular_v2, "#c1666b"),
]

merged_all = []
for name, human_df, _ in methods:
    merged = llm_target.merge(human_df, left_on=["sentence_id", "cond_key"], right_on=["sentence_id", "condition"])
    merged = merged.rename(columns={"P_norm": "llm_p", "prop_target": "human_p"})
    merged_all.append(merged)

xmax = max(0.5, max(m.human_p.max() for m in merged_all) * 1.15)

fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), dpi=150)
for ax, (name, human_df, panel_color), merged in zip(axes, methods, merged_all):
    for cond, color, label in [("restrictive", COL_R, "restrictive"), ("non-restrictive", COL_N, "non-restrictive")]:
        d = merged[merged.cond_key == cond]
        ax.scatter(d.human_p, d.llm_p, s=55, color=color, alpha=0.8, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    r, p = spearmanr(merged.human_p, merged.llm_p)
    ax.plot([0, 1], [0, 1], "--", color="#c3c2b7", lw=1.2, zorder=1, label="y = x")
    ax.set_xlim(-0.03, xmax); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Human: mean fixation proportion on target", fontsize=10.5)
    if ax is axes[0]:
        ax.set_ylabel("GPT-2: P(target) per item", fontsize=11)
    ax.set_title(f"{name}\nρ = {r:.2f}, p = {p:.3f}, n = {len(merged)}", fontsize=12.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e1e0d9", lw=0.7, zorder=0)

axes[0].legend(fontsize=9.5, loc="upper right")
fig.suptitle("Item-level agreement: LLM vs. human, across all three AOI definitions",
             fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/18_scatter_all3methods_combined.png", bbox_inches="tight")
plt.close(fig)
print("saved 18_scatter_all3methods_combined.png")
for (name, _, _), merged in zip(methods, merged_all):
    r, p = spearmanr(merged.human_p, merged.llm_p)
    print(f"  {name}: rho={r:.3f}, p={p:.4f}, n={len(merged)}")
