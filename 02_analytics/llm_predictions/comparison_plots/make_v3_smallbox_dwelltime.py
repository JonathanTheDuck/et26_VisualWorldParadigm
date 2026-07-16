import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
COL_R = "#2a78d6"; COL_N = "#e34948"

# =========================================================
# 0. Geometry — subject box measured directly off the real
#    stimulus image (boy figure silhouette), not eyeballed.
#    Old box: half-extents (420, 300) -> way wider than the
#    actual figure (measured tight bbox half-width = 88.5px).
#    New box: half-width 150 (tight 88.5 + ~60px margin for
#    fixation/gaze imprecision), half-height kept at 300
#    (already close to the measured 287.5).
# =========================================================
CANVAS_W, CANVAS_H = 2400, 1400
SCALE = 0.8; SCREEN_W, SCREEN_H = 2560, 1440
X_OFF = (SCREEN_W - CANVAS_W*SCALE)/2; Y_OFF = (SCREEN_H - CANVAS_H*SCALE)/2
def to_img_px(bx, by):
    sx = bx*SCREEN_W; sy = by*SCREEN_H
    return (sx-X_OFF)/SCALE, (sy-Y_OFF)/SCALE

SUBJECT_CENTER = (1200, 400)
SUBJECT_HALF_OLD = (420, 300)
SUBJECT_HALF_NEW = (150, 300)
OBJ_HALF = 200
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

def classify_original(px, py, subj_half):
    subj = in_box(px, py, *SUBJECT_CENTER, subj_half[0], subj_half[1])
    out = np.full(px.shape, "elsewhere", dtype=object)
    out[subj] = "subject"
    for n, (ox, oy) in OBJ_CENTERS.items():
        m = in_box(px, py, ox, oy, OBJ_HALF)
        out[m] = n
    return out

def classify_corridor(px, py, subj_half):
    subj = in_box(px, py, *SUBJECT_CENTER, subj_half[0], subj_half[1])
    out = np.full(px.shape, "elsewhere", dtype=object)
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
        out[box | corridor] = n
    out[subj] = "subject"
    return out

def classify_angular_v2(px, py, subj_half):
    subj = in_box(px, py, *SUBJECT_CENTER, subj_half[0], subj_half[1])
    ang = np.arctan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
    names = list(OBJ_CENTERS.keys())
    diffs = np.stack([angdiff(ang, OBJ_ANGLES[n]) for n in names], axis=0)
    nearest_idx = np.argmin(diffs, axis=0)
    nearest_diff = np.min(diffs, axis=0)
    nearest = np.array(names)[nearest_idx]
    within = nearest_diff <= WEDGE_HALF
    out = np.full(px.shape, "elsewhere", dtype=object)
    for i, n in enumerate(names):
        out[within & (nearest_idx == i)] = n
    for n, (ox, oy) in OBJ_CENTERS.items():
        out[in_box(px, py, ox, oy, OBJ_HALF)] = n
    out[subj] = "subject"
    return out

METHODS = {
    "original": classify_original,
    "corridor": classify_corridor,
    "angular_v2": classify_angular_v2,
}

# =========================================================
# 1. Load raw critical-window samples (verb OFFSET -> target
#    onset, already filtered by the teammate's pipeline) +
#    trial metadata (sentence_id, target_position)
# =========================================================
raw = pd.read_csv(f"{OUT_DIR}/../../Analysis_pipeline_m/Output/raw_gaze_critical_window.csv")
fp = pd.read_csv(f"{OUT_DIR}/../../Analysis_pipeline_m/Output/fixation_proportions.csv")
meta = fp[["subject_nr", "trial", "sentence_id", "target_position"]].drop_duplicates()
raw = raw.merge(meta, on=["subject_nr", "trial"], how="left").dropna(subset=["sentence_id"]).copy()
xy = raw.apply(lambda r: to_img_px(r.BPOGX, r.BPOGY), axis=1, result_type="expand")
raw["img_x"], raw["img_y"] = xy[0], xy[1]
print(f"{len(raw)} raw samples, {raw.groupby(['subject_nr','trial']).ngroups} trials")

# ---- dwell duration per sample: gap to next sample within the
#      same trial, capped at 20ms (~3x the ~6.7ms sample interval)
#      so blinks/data-loss gaps don't get counted as dwell time ----
raw = raw.sort_values(["subject_nr", "trial", "TIME"]).reset_index(drop=True)
GAP_CAP_S = 0.020
MEDIAN_DT = 0.0066
raw["next_time"] = raw.groupby(["subject_nr", "trial"])["TIME"].shift(-1)
raw["dur_s"] = (raw["next_time"] - raw["TIME"]).clip(upper=GAP_CAP_S)
raw["dur_s"] = raw["dur_s"].fillna(MEDIAN_DT)
raw.loc[raw["dur_s"] < 0, "dur_s"] = MEDIAN_DT

# =========================================================
# 2. Classify with OLD vs NEW subject box, all 3 AOI methods,
#    then aggregate to trial level: sample-count top-choice AND
#    dwell-time (ms) top-choice
# =========================================================
def aggregate(subj_half, box_label):
    rows_summary = []
    item_tables = {}
    for method_name, fn in METHODS.items():
        raw[f"aoi_{method_name}"] = fn(raw["img_x"].values, raw["img_y"].values, subj_half)

        def trial_agg(g):
            tgt_pos = f"pos{int(g['target_position'].iloc[0])}"
            n = len(g)
            counts = g[f"aoi_{method_name}"].value_counts()
            dwell = g.groupby(f"aoi_{method_name}")["dur_s"].sum()
            total_obj_n = sum(counts.get(p, 0) for p in ["pos1","pos2","pos3","pos4"])
            total_obj_ms = sum(dwell.get(p, 0) for p in ["pos1","pos2","pos3","pos4"]) * 1000
            out = {"n_samples": n, "prop_target": counts.get(tgt_pos, 0)/n,
                   "no_object_gaze": total_obj_n == 0,
                   "dwell_target_ms": dwell.get(tgt_pos, 0)*1000,
                   "dwell_total_obj_ms": total_obj_ms,
                   "no_object_dwell": total_obj_ms == 0,
                   "prop_target_dwell": (dwell.get(tgt_pos, 0)*1000/total_obj_ms) if total_obj_ms > 0 else np.nan}
            picks_target_n = np.nan
            if total_obj_n > 0:
                obj_counts = {p: counts.get(p, 0) for p in ["pos1","pos2","pos3","pos4"]}
                top = max(obj_counts, key=obj_counts.get)
                picks_target_n = (top == tgt_pos)
            picks_target_dwell = np.nan
            if total_obj_ms > 0:
                obj_ms = {p: dwell.get(p, 0)*1000 for p in ["pos1","pos2","pos3","pos4"]}
                top = max(obj_ms, key=obj_ms.get)
                picks_target_dwell = (top == tgt_pos)
            out["human_picks_target_samples"] = picks_target_n
            out["human_picks_target_dwell"] = picks_target_dwell
            return pd.Series(out)

        trial_agg_df = raw.groupby(["subject_nr","trial","sentence_id","condition"]).apply(trial_agg, include_groups=False).reset_index()

        no_obj_dwell_rate = trial_agg_df.no_object_dwell.mean()
        no_obj_sample_rate = trial_agg_df.no_object_gaze.mean()

        scored_dwell = trial_agg_df.dropna(subset=["human_picks_target_dwell"])
        cond_summary = scored_dwell.groupby("condition")["human_picks_target_dwell"].agg(["mean", "count"])

        rows_summary.append({
            "box": box_label, "method": method_name,
            "no_object_dwell_rate": no_obj_dwell_rate,
            "no_object_sample_rate": no_obj_sample_rate,
            "restrictive_topchoice_dwell": cond_summary.loc["restrictive","mean"] if "restrictive" in cond_summary.index else np.nan,
            "restrictive_n": int(cond_summary.loc["restrictive","count"]) if "restrictive" in cond_summary.index else 0,
            "nonrestr_topchoice_dwell": cond_summary.loc["non-restrictive","mean"] if "non-restrictive" in cond_summary.index else np.nan,
            "nonrestr_n": int(cond_summary.loc["non-restrictive","count"]) if "non-restrictive" in cond_summary.index else 0,
        })

        item_level = trial_agg_df.groupby(["sentence_id","condition"])[["prop_target","prop_target_dwell"]].mean().reset_index()
        item_tables[method_name] = item_level

    return pd.DataFrame(rows_summary), item_tables

summary_old, items_old = aggregate(SUBJECT_HALF_OLD, "old (840x600)")
summary_new, items_new = aggregate(SUBJECT_HALF_NEW, "new (300x600)")

summary = pd.concat([summary_old, summary_new], ignore_index=True)
summary.to_csv(f"{OUT_DIR}/v3_smallbox_dwelltime_summary.csv", index=False)
print("\n=== summary (old vs new subject box, dwell-time top-choice) ===")
print(summary.to_string(index=False))

for method_name in METHODS:
    items_new[method_name].to_csv(f"{OUT_DIR}/item_level_human_{method_name}_v3_smallbox_dwell.csv", index=False)

# =========================================================
# 3. Wilson CIs for the dwell-time top-choice, angular_v2, new box
#    (the "best" method, smallest subject box, dwell-weighted)
# =========================================================
row = summary_new[summary_new.method == "angular_v2"].iloc[0]
for cond, val_col, n_col in [("restrictive","restrictive_topchoice_dwell","restrictive_n"),
                               ("non-restrictive","nonrestr_topchoice_dwell","nonrestr_n")]:
    v, n = row[val_col], row[n_col]
    lo, hi = proportion_confint(round(v*n), n, method="wilson")
    print(f"angular_v2 + new box, dwell-time top-choice, {cond}: {v:.1%} (n={n}), 95% CI [{lo:.1%}, {hi:.1%}]")

# =========================================================
# 4. Illustration: old vs new subject box, all 3 AOI methods (item 1)
# =========================================================
IMG_PATH = "/Users/ladidadida2025/et26_VisualWorldParadigm/01_experiment/stimuli/img_composition/1_pos1_sub.png"
img = Image.open(IMG_PATH)
IMG_W, IMG_H = img.size
gx, gy = np.meshgrid(np.linspace(0, IMG_W, 480), np.linspace(0, IMG_H, 280))
gpx, gpy = gx.ravel(), gy.ravel()

def masks_for(subj_half, method_name):
    cls = METHODS[method_name](gpx, gpy, subj_half)
    m = {n: (cls == n) for n in OBJ_CENTERS}
    subj = (cls == "subject")
    return subj, m

fig, axes = plt.subplots(2, 3, figsize=(19, 13.6), dpi=150)
method_titles = {"original": "1. Original AOI", "corridor": "2. Corridor AOI", "angular_v2": "3. Angular AOI v2"}
for row_i, (subj_half, box_label) in enumerate([(SUBJECT_HALF_OLD, "OLD subject box (840×600px)"), (SUBJECT_HALF_NEW, "NEW subject box (300×600px, measured off the real figure)")]):
    for col_i, method_name in enumerate(METHODS):
        ax = axes[row_i, col_i]
        subj_mask, m = masks_for(subj_half, method_name)
        ax.imshow(img)
        overlay = np.zeros((*gx.shape, 4))
        for n in OBJ_CENTERS:
            rgb = matplotlib.colors.to_rgb(OBJ_COLOR[n])
            overlay[m[n].reshape(gx.shape)] = (*rgb, 0.4)
        overlay[subj_mask.reshape(gx.shape)] = (*matplotlib.colors.to_rgb(SUBJ_COLOR), 0.35)
        ax.imshow(overlay, extent=(0, IMG_W, IMG_H, 0))
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.axis("off")
        cov = subj_mask.mean() + sum(v.mean() for v in m.values())
        ax.set_title(f"{method_titles[method_name]}\n{box_label}\ncoverage: {cov:.0%}", fontsize=11.5, fontweight="bold" if col_i==0 else "normal")

handles = [patches.Patch(color=SUBJ_COLOR, alpha=0.4, label="subject")]
handles += [patches.Patch(color=OBJ_COLOR[n], alpha=0.55, label=OBJ_NAMES[n]) for n in OBJ_CENTERS]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Shrinking the subject box to match the real figure (measured: 177×575px silhouette)",
             fontsize=15, fontweight="bold", y=1.005)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(f"{OUT_DIR}/19_subjectbox_oldvsnew_illustration.png", bbox_inches="tight")
plt.close(fig)
print("saved 19_subjectbox_oldvsnew_illustration.png")

# =========================================================
# 5. Headline bar chart: dwell-time top-choice, new box,
#    old box vs new box side by side, angular_v2 method
# =========================================================
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
x = np.arange(2); w = 0.35
conds = ["restrictive", "non-restrictive"]
old_row = summary_old[summary_old.method == "angular_v2"].iloc[0]
new_row = summary_new[summary_new.method == "angular_v2"].iloc[0]
old_vals = [old_row.restrictive_topchoice_dwell, old_row.nonrestr_topchoice_dwell]
old_n = [old_row.restrictive_n, old_row.nonrestr_n]
new_vals = [new_row.restrictive_topchoice_dwell, new_row.nonrestr_topchoice_dwell]
new_n = [new_row.restrictive_n, new_row.nonrestr_n]
old_ci = [proportion_confint(round(v*n), n, method="wilson") for v, n in zip(old_vals, old_n)]
new_ci = [proportion_confint(round(v*n), n, method="wilson") for v, n in zip(new_vals, new_n)]
ax.bar(x - w/2, old_vals, w, color="#a9b4bd", label="old subject box",
       yerr=[[v-lo for v,(lo,hi) in zip(old_vals,old_ci)], [hi-v for v,(lo,hi) in zip(old_vals,old_ci)]], capsize=5)
ax.bar(x + w/2, new_vals, w, color="#3f6f9e", label="new (smaller) subject box",
       yerr=[[v-lo for v,(lo,hi) in zip(new_vals,new_ci)], [hi-v for v,(lo,hi) in zip(new_vals,new_ci)]], capsize=5)
ax.axhline(0.25, color="#999", ls=":", lw=1.2, label="chance (25%)")
for i in range(2):
    ax.text(i-w/2, old_ci[i][1]+0.02, f"{old_vals[i]:.0%}\n(n={old_n[i]})", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(i+w/2, new_ci[i][1]+0.02, f"{new_vals[i]:.0%}\n(n={new_n[i]})", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=11.5)
ax.set_ylim(0, 0.8)
ax.set_ylabel("% trials: target has the most dwell time (ms) among the 4 objects", fontsize=10.5)
ax.set_title("Dwell-time top-choice — angular AOI v2, old vs. new subject box", fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=9.5, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/20_dwelltime_topchoice_oldvsnewbox.png")
plt.close(fig)
print("saved 20_dwelltime_topchoice_oldvsnewbox.png")

# =========================================================
# 6. Combined scatter, new box, dwell-time weighted, 3 methods, human on x
# =========================================================
llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_target = llm[llm.Is_Target].copy()
llm_target["sentence_id"] = llm_target["Item"] - 1
llm_target["cond_key"] = llm_target["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), dpi=150)
titles = {"original": "1. Original AOI", "corridor": "2. Corridor AOI", "angular_v2": "3. Angular AOI v2"}
merged_stats = {}
for ax, method_name in zip(axes, METHODS):
    human = items_new[method_name].copy()
    human["prop_target_dwell"] = human["prop_target_dwell"].fillna(0)
    merged = llm_target.merge(human, left_on=["sentence_id","cond_key"], right_on=["sentence_id","condition"])
    merged = merged.rename(columns={"P_norm": "llm_p"})
    r, p = spearmanr(merged.prop_target_dwell, merged.llm_p)
    merged_stats[method_name] = (r, p, len(merged))
    for cond, color, label in [("restrictive", COL_R, "restrictive"), ("non-restrictive", COL_N, "non-restrictive")]:
        d = merged[merged.cond_key == cond]
        ax.scatter(d.prop_target_dwell, d.llm_p, s=55, color=color, alpha=0.8, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    ax.plot([0,1],[0,1], "--", color="#c3c2b7", lw=1.2, zorder=1, label="y = x")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Human: dwell-time proportion on target (of object dwell)", fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("GPT-2: P(target) per item", fontsize=11)
    ax.set_title(f"{titles[method_name]}\nρ = {r:.2f}, p = {p:.3f}, n = {len(merged)}", fontsize=12.5, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#e1e0d9", lw=0.7, zorder=0)
axes[0].legend(fontsize=9.5, loc="upper right")
fig.suptitle("Item-level agreement — dwell-time weighted, new (smaller) subject box",
             fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/21_scatter_dwelltime_newbox_all3methods.png", bbox_inches="tight")
plt.close(fig)
print("saved 21_scatter_dwelltime_newbox_all3methods.png")
print(merged_stats)
