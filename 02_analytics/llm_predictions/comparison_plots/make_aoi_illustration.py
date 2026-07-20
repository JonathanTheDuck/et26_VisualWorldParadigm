import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
IMG_PATH = "/Users/ladidadida2025/et26_VisualWorldParadigm/01_experiment/stimuli/img_composition/1_pos1_sub.png"
img = Image.open(IMG_PATH)
IMG_W, IMG_H = img.size

SUBJECT_CENTER = (1200, 400); SUBJECT_HALF = (420, 300); OBJ_HALF = 200
OBJ_CENTERS = {"pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3), "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6)}
OBJ_NAMES = {"pos1": "cake", "pos2": "toy car", "pos3": "toy train", "pos4": "ball"}
OBJ_COLOR = {"pos1": "#2a78d6", "pos2": "#e34948", "pos3": "#1baf7a", "pos4": "#eda100"}
SUBJ_COLOR = "#6b7280"
CORRIDOR_HALF_WIDTH = 150
OBJ_ANGLES = {n: math.atan2(cy-SUBJECT_CENTER[1], cx-SUBJECT_CENTER[0]) for n, (cx, cy) in OBJ_CENTERS.items()}

def in_box(px, py, cx, cy, hw, hh=None):
    hh = hw if hh is None else hh
    return (abs(px-cx) <= hw) & (abs(py-cy) <= hh)

def angdiff(a, b):
    d = (a-b) % (2*math.pi)
    return np.minimum(d, 2*math.pi-d)

# ---- classify a grid of points for panels 2 & 3 (vectorized) ----
gx, gy = np.meshgrid(np.linspace(0, IMG_W, 480), np.linspace(0, IMG_H, 280))
px, py = gx.ravel(), gy.ravel()

def corridor_mask():
    m = {n: np.zeros(px.shape, dtype=bool) for n in OBJ_CENTERS}
    subj = in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1])
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
    return subj, m

def angular_mask():
    subj = in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1])
    ang = np.arctan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
    names = list(OBJ_CENTERS.keys())
    diffs = np.stack([angdiff(ang, OBJ_ANGLES[n]) for n in names], axis=0)
    nearest = np.array(names)[np.argmin(diffs, axis=0)]
    m = {n: (~subj) & (nearest == n) for n in names}
    for n, (ox, oy) in OBJ_CENTERS.items():
        m[n] = m[n] | in_box(px, py, ox, oy, OBJ_HALF)
    return subj, m

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=150)

# ---- Panel 1: original AOI (boxes only) ----
ax = axes[0]
ax.imshow(img)
subj_box = patches.Rectangle((SUBJECT_CENTER[0]-SUBJECT_HALF[0], SUBJECT_CENTER[1]-SUBJECT_HALF[1]),
                              2*SUBJECT_HALF[0], 2*SUBJECT_HALF[1], facecolor=SUBJ_COLOR, alpha=0.25, edgecolor=SUBJ_COLOR, lw=2)
ax.add_patch(subj_box)
for n, (ox, oy) in OBJ_CENTERS.items():
    r = patches.Rectangle((ox-OBJ_HALF, oy-OBJ_HALF), 2*OBJ_HALF, 2*OBJ_HALF,
                           facecolor=OBJ_COLOR[n], alpha=0.35, edgecolor=OBJ_COLOR[n], lw=2)
    ax.add_patch(r)
ax.set_title("1. Original AOI\n(object box only)", fontsize=13, fontweight="bold")

# ---- Panel 2: corridor AOI ----
ax = axes[1]
ax.imshow(img)
subj_mask, cmask = corridor_mask()
overlay = np.zeros((*gx.shape, 4))
for n in OBJ_CENTERS:
    rgb = matplotlib.colors.to_rgb(OBJ_COLOR[n])
    sel = cmask[n].reshape(gx.shape)
    overlay[sel] = (*rgb, 0.4)
subj_rgb = matplotlib.colors.to_rgb(SUBJ_COLOR)
overlay[subj_mask.reshape(gx.shape)] = (*subj_rgb, 0.3)
ax.imshow(overlay, extent=(0, IMG_W, IMG_H, 0))
ax.set_title("2. Corridor AOI\n(+150px strip toward subject)", fontsize=13, fontweight="bold")

# ---- Panel 3: angular AOI ----
ax = axes[2]
ax.imshow(img)
subj_mask, amask = angular_mask()
overlay = np.zeros((*gx.shape, 4))
for n in OBJ_CENTERS:
    rgb = matplotlib.colors.to_rgb(OBJ_COLOR[n])
    sel = amask[n].reshape(gx.shape)
    overlay[sel] = (*rgb, 0.4)
overlay[subj_mask.reshape(gx.shape)] = (*subj_rgb, 0.3)
ax.imshow(overlay, extent=(0, IMG_W, IMG_H, 0))
ax.set_title("3. Angular AOI\n(direction from subject — no gaps)", fontsize=13, fontweight="bold")

for ax in axes:
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.axis("off")

handles = [patches.Patch(color=SUBJ_COLOR, alpha=0.4, label="subject")]
handles += [patches.Patch(color=OBJ_COLOR[n], alpha=0.55, label=OBJ_NAMES[n]) for n in OBJ_CENTERS]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("How much did the AOI actually grow? (item 1, restrictive)", fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f"{OUT_DIR}/13_aoi_expansion_illustration.png", bbox_inches="tight")
plt.close(fig)

# ---- print coverage stats ----
total_px = px.shape[0]
subj_orig = in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1]).mean()
orig_cov = subj_orig + sum(in_box(px, py, ox, oy, OBJ_HALF).mean() for ox, oy in OBJ_CENTERS.values())
subj_c, cmask = corridor_mask()
corridor_cov = subj_c.mean() + sum(m.mean() for m in cmask.values())
subj_a, amask = angular_mask()
angular_cov = subj_a.mean() + sum(m.mean() for m in amask.values())
print(f"Screen coverage (fraction of pixels classified to SOME AOI, incl. subject):")
print(f"  Original AOI:  {orig_cov:.1%}")
print(f"  Corridor AOI:  {corridor_cov:.1%}")
print(f"  Angular AOI:   {angular_cov:.1%}")
print("saved 13_aoi_expansion_illustration.png")
