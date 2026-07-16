import pandas as pd, numpy as np, math

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"

CANVAS_W, CANVAS_H = 2560, 1440
SCALE = 0.8; SCREEN_W, SCREEN_H = 2560, 1440
X_OFF = (SCREEN_W - CANVAS_W*SCALE)/2; Y_OFF = (SCREEN_H - CANVAS_H*SCALE)/2
def to_img_px(bx, by):
    sx = bx*SCREEN_W; sy = by*SCREEN_H
    return (sx-X_OFF)/SCALE, (sy-Y_OFF)/SCALE

SUBJECT_CENTER = (1200, 400)
SUBJECT_HALF = (420, 300)
OBJ_HALF = 200
OBJ_CENTERS = {
    "pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3),
    "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6),
}
# angle of each object as seen from the subject center (matplotlib/screen y grows downward)
OBJ_ANGLES = {name: math.atan2(cy-SUBJECT_CENTER[1], cx-SUBJECT_CENTER[0]) for name, (cx, cy) in OBJ_CENTERS.items()}

def in_box(px, py, cx, cy, half_w, half_h=None):
    half_h = half_w if half_h is None else half_h
    return abs(px-cx) <= half_w and abs(py-cy) <= half_h

def angdiff(a, b):
    d = (a-b) % (2*math.pi)
    return min(d, 2*math.pi-d)

def classify_angular(px, py):
    """Object box first; else if outside the subject box, assign by nearest
    angular sector (unbounded wedge from the subject center) -- this covers
    every pixel on screen with no arbitrary corridor-width parameter."""
    if in_box(px, py, *SUBJECT_CENTER, SUBJECT_HALF[0], SUBJECT_HALF[1]):
        return "subject"
    for name, (ox, oy) in OBJ_CENTERS.items():
        if in_box(px, py, ox, oy, OBJ_HALF):
            return name
    ang = math.atan2(py-SUBJECT_CENTER[1], px-SUBJECT_CENTER[0])
    best_name = min(OBJ_ANGLES, key=lambda n: angdiff(ang, OBJ_ANGLES[n]))
    return best_name

raw = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/raw_gaze_critical_window.csv")
fp = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Output/fixation_proportions.csv")
meta = fp[["subject_nr", "trial", "sentence_id", "target_position"]].drop_duplicates()
raw = raw.merge(meta, on=["subject_nr", "trial"], how="left").dropna(subset=["sentence_id"])

xy = raw.apply(lambda r: to_img_px(r.BPOGX, r.BPOGY), axis=1, result_type="expand")
raw["img_x"], raw["img_y"] = xy[0], xy[1]
raw["aoi_angular"] = raw.apply(lambda r: classify_angular(r.img_x, r.img_y), axis=1)

print("Angular AOI counts (every non-subject pixel now belongs to some object -- no 'elsewhere'):")
print(raw["aoi_angular"].value_counts())

def trial_props(g):
    n = len(g)
    tgt_pos = f"pos{int(g['target_position'].iloc[0])}"
    counts = g["aoi_angular"].value_counts()
    obj_counts = {p: counts.get(p, 0) for p in ["pos1", "pos2", "pos3", "pos4"]}
    total_obj = sum(obj_counts.values())
    out = {f"prop_{p}": obj_counts[p]/n for p in obj_counts}
    out["prop_subject"] = counts.get("subject", 0)/n
    out["prop_target"] = obj_counts[tgt_pos]/n
    out["no_object_gaze"] = total_obj == 0
    if total_obj > 0:
        top_pos = max(obj_counts, key=obj_counts.get)
        out["human_picks_target"] = (top_pos == tgt_pos)
    else:
        out["human_picks_target"] = np.nan
    return pd.Series(out)

trial_agg = raw.groupby(["subject_nr", "trial", "sentence_id", "condition"]).apply(trial_props, include_groups=False).reset_index()
print(f"\nno_object_gaze rate (angular): {trial_agg.no_object_gaze.mean():.1%}  "
      f"(corridor version: 56.3%, original AOI: 79.4%)")

llm = pd.read_csv("/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/result_vwp50_scene_table.csv")
llm_top = llm.loc[llm.groupby(["Item", "Condition"])["P_norm"].idxmax()][["Item", "Condition", "Is_Target"]]
llm_top = llm_top.rename(columns={"Is_Target": "llm_picks_target"})
llm_top["sentence_id"] = llm_top["Item"] - 1
llm_top["cond_key"] = llm_top["Condition"].map({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

trial_top = trial_agg.dropna(subset=["human_picks_target"]).merge(
    llm_top[["sentence_id", "cond_key", "llm_picks_target"]],
    left_on=["sentence_id", "condition"], right_on=["sentence_id", "cond_key"], how="left")
summary = trial_top.groupby("condition")[["llm_picks_target", "human_picks_target"]].mean()
n_scored = trial_top.groupby("condition").size()
print("\nTop-choice = target (angular AOI):\n", summary)
print("n trials scored:\n", n_scored)

item_human_ang = trial_agg.groupby(["sentence_id", "condition"])[["prop_pos1","prop_pos2","prop_pos3","prop_pos4","prop_target"]].mean().reset_index()
item_human_ang.to_csv(f"{OUT_DIR}/item_level_human_angularAOI.csv", index=False)
print("\nsaved item_level_human_angularAOI.csv")
