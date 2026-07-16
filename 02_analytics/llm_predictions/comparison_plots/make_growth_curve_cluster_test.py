import sys, math, itertools
sys.path.insert(0, "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m")
import analysis_pipeline as ap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/llm_predictions/comparison_plots"
INPUT_DIR = "/Users/ladidadida2025/et26_VisualWorldParadigm/02_analytics/Analysis_pipeline_m/Input"
COL_R = "#2a78d6"; COL_N = "#e34948"

# =========================================================
# Geometry: angular AOI v2 (equal +/-25 deg wedges) + the
# subject box measured off the real figure (see v3 script) --
# our current best/most-defensible AOI.
# =========================================================
CANVAS_W, CANVAS_H = 2400, 1400
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
# 1. Annotation lookup: (id, condition) -> verb_offset_s, target_onset_s
# =========================================================
ann = ap.load_annotation(f"{INPUT_DIR}/annotation_audiov2.csv")
ann_lookup = {(int(r.id), str(r.condition)): (float(r.verb_offset_s), float(r.target_onset_s))
              for r in ann.itertuples()}

BIN_MS = 50.0
MAX_BIN_MS = 350.0
bin_edges = np.arange(0, MAX_BIN_MS, BIN_MS)

# =========================================================
# 2. Walk every subject/trial, pull raw samples in
#    [verb_offset, target_onset], bin by ms-from-verb-offset,
#    classify with angular AOI v2 + measured subject box
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
        verb_offset_s, target_onset_s = ann_lookup[key]
        onset = seg["audio_onset_s"]
        win_start, win_end = onset + verb_offset_s, onset + target_onset_s
        g = seg["gaze"]
        w = g[(g["TIME"] >= win_start) & (g["TIME"] < win_end)]
        if "BPOGV" in w.columns:
            w = w[w["BPOGV"] == 1]
        if len(w) == 0:
            continue
        target_position = int(t_row["target_position"])
        tgt_pos = f"pos{target_position}"
        for r in w.itertuples():
            px, py = to_img_px(r.BPOGX, r.BPOGY)
            cls = classify_angular_v2(px, py)
            rel_ms = (r.TIME - win_start) * 1000
            rows.append({
                "subject_nr": subj_nr, "trial": i, "sentence_id": audio_id, "condition": condition,
                "rel_ms": rel_ms, "is_target": cls == tgt_pos,
                "is_object": cls in ("pos1","pos2","pos3","pos4"),
            })

samp = pd.DataFrame(rows)
print(f"{len(samp)} raw samples across {samp.groupby(['subject_nr','trial']).ngroups} trials")

samp["bin_start_ms"] = (samp["rel_ms"] // BIN_MS) * BIN_MS
samp = samp[(samp["bin_start_ms"] >= 0) & (samp["bin_start_ms"] < MAX_BIN_MS)]

# item-level window coverage per bin (how many of the 99 item-conditions
# actually have a window long enough to reach this bin at all)
ann["window_ms"] = (ann.target_onset_s - ann.verb_offset_s) * 1000
coverage_by_bin = {b: int((ann["window_ms"] >= b + BIN_MS).sum()) for b in bin_edges}

# =========================================================
# 3. Trial-level -> subject-level proportion-target-of-object-gaze per bin
#    (proportion of ON-OBJECT samples, i.e. target vs distractor share,
#    to isolate "which object" rather than "on/off subject")
# =========================================================
trial_bin = (samp.groupby(["subject_nr","trial","sentence_id","condition","bin_start_ms"])
             .agg(n_samples=("is_target","size"), n_target=("is_target","sum"), n_object=("is_object","sum"))
             .reset_index())
trial_bin["prop_target_of_object"] = np.where(trial_bin.n_object > 0, trial_bin.n_target / trial_bin.n_object.replace(0, np.nan), np.nan)
trial_bin["prop_target_of_all"] = trial_bin.n_target / trial_bin.n_samples

subj_bin = (trial_bin.groupby(["subject_nr","condition","bin_start_ms"])
            .agg(prop_target_of_all=("prop_target_of_all","mean"),
                 n_trials=("trial","nunique"))
            .reset_index())

pivot = subj_bin.pivot_table(index=["subject_nr","bin_start_ms"], columns="condition", values="prop_target_of_all")
pivot = pivot.dropna()  # keep only subject x bin cells with BOTH conditions present
print(f"\nsubject x bin cells with both conditions present: {len(pivot)} / {5*len(bin_edges)} possible")

# =========================================================
# 4. Cluster-based permutation test (paired, by subject)
#    Exchangeability unit = subject (swap restrictive<->non-restrictive
#    labels together across all bins for that subject). With n=5
#    subjects there are 2^5=32 possible label assignments, so the
#    smallest achievable p-value is 1/32 ~= 0.031 -- noted explicitly.
# =========================================================
bins_with_data = sorted(pivot.index.get_level_values("bin_start_ms").unique())
subjects = sorted(pivot.index.get_level_values("subject_nr").unique())
print(f"bins with >=1 subject having both conditions: {bins_with_data}")
print(f"subjects contributing: {subjects}")

# build a subject x bin matrix of (restrictive - non-restrictive), NaN if missing
diff_mat = pd.DataFrame(index=subjects, columns=bins_with_data, dtype=float)
for (subj, b), row in pivot.iterrows():
    if "restrictive" in row and "non-restrictive" in row:
        diff_mat.loc[subj, b] = row["restrictive"] - row["non-restrictive"]

# only keep bins where ALL 5 subjects have a value (fully balanced design
# needed for a clean paired permutation test)
full_bins = [b for b in bins_with_data if diff_mat[b].notna().all()]
print(f"bins with all {len(subjects)} subjects present: {full_bins}")
diff_full = diff_mat[full_bins].values  # shape (n_subj, n_bins)

def paired_t_per_bin(mat):
    t_vals, p_vals = [], []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        t, p = stats.ttest_1samp(col, 0.0)
        t_vals.append(t); p_vals.append(p)
    return np.array(t_vals), np.array(p_vals)

obs_t, obs_p = paired_t_per_bin(diff_full)

def find_clusters(t_vals, p_vals, alpha=0.05):
    sig = p_vals < alpha
    clusters = []
    i = 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j < len(sig) and sig[j] and np.sign(t_vals[j]) == np.sign(t_vals[i]):
                j += 1
            clusters.append((i, j-1, np.sum(t_vals[i:j])))
            i = j
        else:
            i += 1
    return clusters

obs_clusters = find_clusters(obs_t, obs_p)
print(f"\nobserved candidate clusters (uncorrected p<0.05): {obs_clusters}")

# permutation: flip sign of ALL bins for a subject (equivalent to swapping
# which condition is "restrictive" for that subject), all 2^5 combinations
n_subj = diff_full.shape[0]
null_max_mass = []
for flips in itertools.product([1, -1], repeat=n_subj):
    flips = np.array(flips).reshape(-1, 1)
    permuted = diff_full * flips
    t_p, p_p = paired_t_per_bin(permuted)
    clusters_p = find_clusters(t_p, p_p)
    max_mass = max([abs(c[2]) for c in clusters_p], default=0.0)
    null_max_mass.append(max_mass)
null_max_mass = np.array(null_max_mass)

print(f"\n{len(null_max_mass)} permutations (2^{n_subj}), null max |cluster mass| distribution:")
print(f"  mean={null_max_mass.mean():.2f}, 95th pct={np.percentile(null_max_mass,95):.2f}, max={null_max_mass.max():.2f}")

results = []
for (i0, i1, mass) in obs_clusters:
    p_perm = (null_max_mass >= abs(mass)).mean()
    results.append({
        "bin_start_ms": full_bins[i0], "bin_end_ms": full_bins[i1] + BIN_MS,
        "cluster_mass": mass, "p_permutation": p_perm,
        "min_possible_p": 1/len(null_max_mass),
    })
res_df = pd.DataFrame(results)
print("\n=== cluster-level results ===")
print(res_df.to_string(index=False) if len(res_df) else "no candidate clusters found at uncorrected p<0.05")
res_df.to_csv(f"{OUT_DIR}/growth_curve_cluster_test_results.csv", index=False)

# =========================================================
# 5. Chart: fine-grained growth curve (target-of-object share)
#    by condition, with per-bin item coverage (n) and any
#    significant cluster shaded
# =========================================================
curve = subj_bin.groupby(["condition","bin_start_ms"])["prop_target_of_all"].agg(["mean","sem","count"]).reset_index()

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=150, height_ratios=[3,1], sharex=True)
for cond, color, label in [("restrictive", COL_R, "restrictive"), ("non-restrictive", COL_N, "non-restrictive")]:
    d = curve[curve.condition == cond].sort_values("bin_start_ms")
    ax.plot(d.bin_start_ms + BIN_MS/2, d["mean"], color=color, lw=2, marker="o", ms=5, label=label)
    ax.fill_between(d.bin_start_ms + BIN_MS/2, d["mean"]-d["sem"], d["mean"]+d["sem"], color=color, alpha=0.18)
ax.axhline(0.25, color="#999", ls=":", lw=1.1, label="chance (25%)")
for _, r in res_df.iterrows():
    ax.axvspan(r.bin_start_ms, r.bin_end_ms, color="#f4d35e", alpha=0.35, zorder=0)
    ax.text((r.bin_start_ms+r.bin_end_ms)/2, 0.02, f"p={r.p_permutation:.3f}", ha="center", fontsize=8.5, color="#7a5c00")
ax.set_ylabel("Proportion of samples on target\n(angular AOI v2, measured subject box)", fontsize=10.5)
ax.set_title(f"Fine-grained time course, verb offset → target onset ({BIN_MS:.0f}ms bins)\n"
             f"cluster-based permutation test, n=5 subjects (2^5=32 perms, min p={1/32:.3f})",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
ax.grid(color="#e1e0d9", lw=0.7)
ax.set_ylim(0, max(0.6, curve["mean"].max()*1.2))

n_by_bin = curve.groupby("bin_start_ms")["count"].sum()
cov_frac = pd.Series({b: coverage_by_bin.get(b, 0) for b in bin_edges}) / 99
ax2.bar(np.array(bin_edges)+BIN_MS/2, cov_frac.reindex(bin_edges).values, width=BIN_MS*0.85, color="#b7c4cf")
ax2.set_ylabel("% items with\nwindow this long", fontsize=9)
ax2.set_xlabel("Time from verb offset (ms)", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", color="#e1e0d9", lw=0.7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/22_growth_curve_finebin_clustertest.png")
plt.close(fig)
print("\nsaved 22_growth_curve_finebin_clustertest.png")

subj_bin.to_csv(f"{OUT_DIR}/growth_curve_finebin_by_subject.csv", index=False)
curve.to_csv(f"{OUT_DIR}/growth_curve_finebin_pooled.csv", index=False)
