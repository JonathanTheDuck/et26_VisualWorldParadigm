import sys, math
sys.path.insert(0, "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m")
import analysis_pipeline as ap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
INPUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Input"
COL_R = "#2a78d6"; COL_N = "#e34948"; COL_R_D = "#9dbfe0"; COL_N_D = "#eda6a4"

# Same AOI as the rest of the recent work: angular v2 (equal +/-25deg wedges)
# + the subject box measured off the real figure (150,300 half-extents).
CANVAS_W, CANVAS_H = 2560, 1440
SCALE = 0.8; SCREEN_W, SCREEN_H = 2560, 1440
X_OFF = (SCREEN_W - CANVAS_W*SCALE)/2; Y_OFF = (SCREEN_H - CANVAS_H*SCALE)/2
def to_img_px(bx, by):
    sx = bx*SCREEN_W; sy = by*SCREEN_H
    return (sx-X_OFF)/SCALE, (sy-Y_OFF)/SCALE

SUBJECT_CENTER = (1200, 400); SUBJECT_HALF = (150, 300); OBJ_HALF = 200
OBJ_CENTERS = {"pos1": (1982.5, 578.6), "pos2": (1542.3, 1025.3), "pos3": (857.7, 1025.3), "pos4": (417.5, 578.6)}
OBJ_ANGLES = {n: math.atan2(cy-SUBJECT_CENTER[1], cx-SUBJECT_CENTER[0]) for n, (cx, cy) in OBJ_CENTERS.items()}
WEDGE_HALF = math.radians(25)

def in_box(px, py, cx, cy, hw, hh=None):
    hh = hw if hh is None else hh
    return abs(px-cx) <= hw and abs(py-cy) <= hh

def angdiff(a, b):
    d = (a-b) % (2*math.pi)
    return min(d, 2*math.pi-d)

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
# 1. Annotation: verb onset (anchor, t=0), verb offset, target
#    onset/offset -- full timing, not just the narrow critical window
# =========================================================
words = ap.load_annotation_words(f"{INPUT_DIR}/annotation_audiov2.csv")
verbs_on = words[words.WordRole == "ROOT"][["id","condition","start_s"]].rename(columns={"start_s":"verb_onset_s"})
verbs_off = words[words.WordRole == "ROOT"][["id","condition","end_s"]].rename(columns={"end_s":"verb_offset_s"})
targets = words[words.WordRole == "dobj"][["id","condition","start_s","end_s"]].rename(columns={"start_s":"target_onset_s","end_s":"target_offset_s"})
ann = verbs_on.merge(verbs_off, on=["id","condition"]).merge(targets, on=["id","condition"])
ann_lookup = {(int(r.id), str(r.condition)): r for r in ann.itertuples()}
print(f"{len(ann)} (item,condition) annotation rows")
print("verb_onset->verb_offset (ms):", ((ann.verb_offset_s-ann.verb_onset_s)*1000).mean())
print("verb_onset->target_onset (ms):", ((ann.target_onset_s-ann.verb_onset_s)*1000).mean())

WIN_START_MS, WIN_END_MS, BIN_MS = -300, 1600, 50
bin_edges = np.arange(WIN_START_MS, WIN_END_MS, BIN_MS)

# =========================================================
# 2. Walk every subject/trial's FULL gaze stream (not cropped to
#    the narrow critical window), time-lock to verb onset, classify
# =========================================================
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
        a = ann_lookup[key]
        onset = seg["audio_onset_s"]
        t0 = onset + a.verb_onset_s
        g = seg["gaze"]
        w = g[(g["TIME"] >= t0 + WIN_START_MS/1000) & (g["TIME"] < t0 + WIN_END_MS/1000)]
        if "BPOGV" in w.columns:
            w = w[w["BPOGV"] == 1]
        if len(w) == 0:
            continue
        target_position = int(t_row["target_position"])
        tgt_pos = f"pos{target_position}"
        for r in w.itertuples():
            px, py = to_img_px(r.BPOGX, r.BPOGY)
            cls = classify_angular_v2(px, py)
            rel_ms = (r.TIME - t0) * 1000
            rows.append({
                "subject_nr": subj_nr, "trial": i, "sentence_id": audio_id, "condition": condition,
                "rel_ms": rel_ms, "cls": cls, "target_position": tgt_pos,
            })

samp = pd.DataFrame(rows)
print(f"\n{len(samp)} raw samples, {samp.groupby(['subject_nr','trial']).ngroups} trials")
samp["is_target"] = samp["cls"] == samp["target_position"]
samp["is_distractor"] = (samp["cls"].isin(["pos1","pos2","pos3","pos4"])) & (~samp["is_target"])
samp["bin_start_ms"] = ((samp["rel_ms"] - WIN_START_MS) // BIN_MS) * BIN_MS + WIN_START_MS
samp = samp[(samp["bin_start_ms"] >= WIN_START_MS) & (samp["bin_start_ms"] < WIN_END_MS)]

trial_bin = (samp.groupby(["subject_nr","trial","condition","bin_start_ms"])
             .agg(n=("cls","size"), n_target=("is_target","sum"), n_distr=("is_distractor","sum"))
             .reset_index())
trial_bin["prop_target"] = trial_bin.n_target / trial_bin.n
trial_bin["prop_distractor_mean"] = (trial_bin.n_distr / trial_bin.n) / 3  # per-distractor average (3 distractors)

curve = (trial_bin.groupby(["condition","bin_start_ms"])
         .agg(target_mean=("prop_target","mean"), target_sem=("prop_target","sem"),
              distr_mean=("prop_distractor_mean","mean"), distr_sem=("prop_distractor_mean","sem"),
              n_trials=("trial","size"))
         .reset_index())
curve.to_csv(f"{OUT_DIR}/full_timecourse_by_condition.csv", index=False)

# item coverage per bin (audio total duration = target_onset + tail to sentence end)
last_word_end = words.groupby(["id","condition"])["end_s"].max().reset_index().rename(columns={"end_s":"sent_end_s"})
ann2 = ann.merge(last_word_end, on=["id","condition"])
ann2["avail_ms"] = (ann2.sent_end_s - ann2.verb_onset_s) * 1000
coverage_by_bin = {b: int((ann2["avail_ms"] >= b + BIN_MS).sum()) for b in bin_edges}

# =========================================================
# 3. Chart: original-paper style growth curve — target vs.
#    mean-distractor, by condition, full time range, verb onset
#    to well past target onset, with mean verb-offset/target-onset
#    markers
# =========================================================
mean_verb_offset = (ann.verb_offset_s - ann.verb_onset_s).mean() * 1000
mean_target_onset = (ann.target_onset_s - ann.verb_onset_s).mean() * 1000

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150, height_ratios=[3.2, 1], sharex=True)
for cond, ct, cd, label in [("restrictive", COL_R, COL_R_D, "restrictive"), ("non-restrictive", COL_N, COL_N_D, "non-restrictive")]:
    d = curve[curve.condition == cond].sort_values("bin_start_ms")
    x = d.bin_start_ms + BIN_MS/2
    ax.plot(x, d.target_mean, color=ct, lw=2.2, label=f"{label} — target")
    ax.fill_between(x, d.target_mean-d.target_sem, d.target_mean+d.target_sem, color=ct, alpha=0.15)
    ax.plot(x, d.distr_mean, color=cd, lw=1.6, ls="--", label=f"{label} — mean distractor")

ax.axvline(0, color="#555", lw=1, ls="-")
ax.text(0, ax.get_ylim()[1] if False else 0.02, "verb\nonset", fontsize=8.5, ha="center", color="#555")
ax.axvline(mean_verb_offset, color="#888", lw=1, ls=":")
ax.text(mean_verb_offset, 0.02, "mean verb\noffset", fontsize=8.5, ha="center", color="#888")
ax.axvline(mean_target_onset, color="#888", lw=1, ls=":")
ax.text(mean_target_onset, 0.02, "mean target\nonset", fontsize=8.5, ha="center", color="#888")
ax.axhline(0.25, color="#bbb", lw=1, ls=":")

ax.set_ylabel("Proportion of gaze samples\n(angular AOI v2, measured subject box)", fontsize=10.5)
ax.set_title("Full time course, verb onset → well past target onset (original-paper style)\n"
             "target vs. mean-distractor, by condition — n=5 pilot subjects", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.spines[["top","right"]].set_visible(False)
ax.grid(color="#e1e0d9", lw=0.7)
ax.set_ylim(0, max(0.5, curve.target_mean.max()*1.25))

cov = pd.Series({b: coverage_by_bin.get(b, 0) for b in bin_edges}) / 99
ax2.bar(np.array(bin_edges)+BIN_MS/2, cov.reindex(bin_edges).values, width=BIN_MS*0.85, color="#b7c4cf")
ax2.set_ylabel("% items with\naudio this long", fontsize=8.5)
ax2.set_xlabel("Time from verb onset (ms)", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", color="#e1e0d9", lw=0.7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/23_full_timecourse_originalstyle.png")
plt.close(fig)
print("saved 23_full_timecourse_originalstyle.png")
