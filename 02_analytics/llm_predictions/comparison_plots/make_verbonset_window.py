import sys, math
sys.path.insert(0, "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m")
import analysis_pipeline as ap
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
INPUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Input"
COL_LLM = "#4f9e6f"; COL_HUM = "#3f6f9e"; CHANCE = 0.25

# ---------------------------------------------------------------
# 1. Annotation: build (id, condition) -> (verb_onset_s, target_onset_s, target_word)
# ---------------------------------------------------------------
words_df = ap.load_annotation_words(f"{INPUT_DIR}/annotation_audiov2.csv")
verbs = (words_df[words_df["WordRole"] == "ROOT"][["id", "condition", "start_s"]]
         .rename(columns={"start_s": "verb_onset_s"}))
targets = (words_df[words_df["WordRole"] == "dobj"][["id", "condition", "start_s", "word"]]
           .rename(columns={"start_s": "target_onset_s", "word": "target_word"}))
ann_df = verbs.merge(targets, on=["id", "condition"])
ann_lookup = {(int(r.id), str(r.condition)): (float(r.verb_onset_s), float(r.target_onset_s))
              for r in ann_df.itertuples()}
print(f"{len(ann_lookup)} (item, condition) annotation entries loaded")

# ---------------------------------------------------------------
# 2. AOI geometry (angular method, matches make_angular_aoi.py)
# ---------------------------------------------------------------
CANVAS_W, CANVAS_H = 2560, 1440
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

# ---------------------------------------------------------------
# 3. Walk every subject, every trial: extract raw samples in
#    [verb_onset, target_onset], classify, aggregate
# ---------------------------------------------------------------
rows = []
for subj_nr in range(1, 6):
    gaze = ap.load_gaze(f"{INPUT_DIR}/subject-{subj_nr}.tsv")
    trials = ap.load_trials(f"{INPUT_DIR}/subject-{subj_nr}.csv")
    segs = ap.segment_trials(gaze)
    n = min(len(segs), len(trials))
    for i in range(n):
        seg = segs[i]
        t_row = trials.iloc[i]
        try:
            audio_id = int(str(t_row["audio_file"]).split("_")[0])
        except ValueError:
            continue
        condition = str(t_row["condition"])
        key = (audio_id, condition)
        if key not in ann_lookup:
            continue
        verb_onset_s, target_onset_s = ann_lookup[key]
        onset = seg["audio_onset_s"]
        win_start, win_end = onset + verb_onset_s, onset + target_onset_s
        g = seg["gaze"]
        w = g[(g["TIME"] >= win_start) & (g["TIME"] < win_end)]
        if "BPOGV" in w.columns:
            w = w[w["BPOGV"] == 1]
        if len(w) == 0:
            continue
        target_position = int(t_row["target_position"])
        tgt_pos = f"pos{target_position}"
        classes = [classify_angular(*to_img_px(r.BPOGX, r.BPOGY)) for r in w.itertuples()]
        n_samp = len(classes)
        obj_counts = {p: classes.count(p) for p in ["pos1", "pos2", "pos3", "pos4"]}
        total_obj = sum(obj_counts.values())
        prop_target = obj_counts[tgt_pos] / n_samp
        no_obj = total_obj == 0
        picks_target = np.nan
        if total_obj > 0:
            top_pos = max(obj_counts, key=obj_counts.get)
            picks_target = (top_pos == tgt_pos)
        rows.append({
            "subject_nr": subj_nr, "trial": i, "sentence_id": audio_id, "condition": condition,
            "n_samples": n_samp, "prop_target": prop_target, "no_object_gaze": no_obj,
            "human_picks_target": picks_target,
            "window_ms": (target_onset_s - verb_onset_s) * 1000,
        })

df = pd.DataFrame(rows)
print(f"\n{len(df)} trials processed (verb ONSET -> target onset window)")
print(f"mean window length: {df.window_ms.mean():.0f}ms  (critical window verb OFFSET->onset was ~250-350ms)")
print(f"no_object_gaze rate: {df.no_object_gaze.mean():.1%}")

scored = df.dropna(subset=["human_picks_target"])
summary = scored.groupby("condition")["human_picks_target"].agg(["mean", "count"])
print("\nTop choice = target, verb-ONSET window (angular AOI):\n", summary)

df.to_csv(f"{OUT_DIR}/verbonset_window_by_trial.csv", index=False)

# ---------------------------------------------------------------
# 4. Chart: compare verb-OFFSET window (already computed) vs verb-ONSET window
# ---------------------------------------------------------------
prior = {"restrictive": (0.433, 90), "non-restrictive": (0.329, 85)}  # verb-offset window, angular AOI
conds = ["restrictive", "non-restrictive"]
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
x = np.arange(2); w = 0.35
prior_vals = [prior[c][0] for c in conds]; prior_n = [prior[c][1] for c in conds]
new_vals = [summary.loc[c, "mean"] for c in conds]; new_n = [int(summary.loc[c, "count"]) for c in conds]
prior_ci = [proportion_confint(round(v*n), n, method="wilson") for v, n in zip(prior_vals, prior_n)]
new_ci = [proportion_confint(round(v*n), n, method="wilson") for v, n in zip(new_vals, new_n)]

ax.bar(x - w/2, prior_vals, w, color="#8ea6bd", label="verb OFFSET → target onset (original window)",
       yerr=[[v-lo for v,(lo,hi) in zip(prior_vals,prior_ci)], [hi-v for v,(lo,hi) in zip(prior_vals,prior_ci)]], capsize=5)
ax.bar(x + w/2, new_vals, w, color=COL_HUM, label="verb ONSET → target onset (this window)",
       yerr=[[v-lo for v,(lo,hi) in zip(new_vals,new_ci)], [hi-v for v,(lo,hi) in zip(new_vals,new_ci)]], capsize=5)
ax.axhline(CHANCE, color="#999", ls=":", lw=1.2)
ax.text(1.4, CHANCE+0.015, "chance (25%)", fontsize=8.5, color="#777", ha="right")
for i, c in enumerate(conds):
    ax.text(i-w/2, prior_ci[i][1]+0.02, f"{prior_vals[i]:.0%}\n(n={prior_n[i]})", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(i+w/2, new_ci[i][1]+0.02, f"{new_vals[i]:.0%}\n(n={new_n[i]})", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=11)
ax.set_ylim(0, 0.75)
ax.set_ylabel("% trials: target = top choice (angular AOI)", fontsize=11)
ax.set_title("Does widening the window (include the verb itself) change the result?", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/12_verbonset_vs_verboffset_window.png")
plt.close(fig)
print("saved 12_verbonset_vs_verboffset_window.png")
