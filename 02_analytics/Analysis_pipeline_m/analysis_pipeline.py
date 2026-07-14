"""
Eye-Tracking Visual World Paradigm — Analysis Pipeline
=======================================================
Inputs
------
  j.tsv               GazePoint raw sample stream (one participant)
  j.csv               OpenSesame trial log (one participant)
  annotation_audiov1.csv  Forced-alignment word timing per sentence

Outputs (saved to OUTPUT_DIR)
------
  fixation_proportions.csv       Per-trial 4-way fixation proportions + metrics
  fixation_critical_window.csv   Every fixation event inside the critical window
  growth_curves.csv              Fixation proportions in 50 ms bins (growth curves)
  raw_gaze_critical_window.csv   Every raw (per-sample) gaze point inside the critical window

  plots/growth_curve.png              Target vs. distractor fixation curves over time, by condition
  plots/fixation_proportions.png      Mean 4-way AOI fixation proportions, by condition
  plots/target_advantage.png          Target-vs-distractor advantage, by condition (box + points)
  plots/aoi_fixation_qc.png           Fixation-level events plotted over the AOI boxes (sanity check)
  plots/raw_gaze_points.png           Raw per-sample gaze point density over the AOI boxes

Design notes
------------
* All timing stays on the GazePoint clock (seconds) throughout.
  The audio onset marker written to the TSV USER column is used directly,
  so no cross-clock conversion is needed. TIME and FPOGS are on the same
  clock (verified: FPOGS is always <= TIME, since a fixation's reported
  start time precedes the current sample's TIME by construction).
* Fixation *timing* (FPOGS/FPOGD, and the FPOGID grouping itself) still
  comes from GazePoint's onboard fixation algorithm. Fixation *position*
  is computed by this script as the mean BPOGX/BPOGY ("Best Point of
  Gaze" — GazePoint's fused left+right-eye raw sample) across the valid
  (BPOGV==1) samples inside each FPOGID group, rather than trusting
  GazePoint's own pre-averaged FPOGX/FPOGY.
* The critical window per trial is [verb_offset, target_onset] relative to
  audio onset, both sourced from the forced-alignment annotation.
* AOI bounding boxes were measured from 1_pos1_sub.png (2400×1400 px) and
  converted to the GazePoint normalized (0-1) coordinate space assuming
  OpenSesame "scale to fit, preserve aspect ratio" display mode on 1920×1080.

  Slot layout (consistent across all stimulus images):
    pos1 = top-right   pos2 = top-left
    pos3 = bottom-left pos4 = bottom-right
    (subject always in center — not an AOI of interest)

Distractor object names
-----------------------
The 4-way output uses slot labels (pos1–pos4). To add specific object names
for the non-target slots, provide a stimulus list CSV with columns:
    sentence_id, pos1_object, pos2_object, pos3_object, pos4_object
and call add_object_names(props_df, stimulus_list_path) at the end of main().

Portability
-----------
This script is designed to be run straight after cloning the repo — no
path editing required. All input/output paths are resolved relative to
this script's own location:

    <repo>/02_analytics/Analysis_pipeline_m/
        ├── analysis_pipeline.py   <- this file
        ├── Input/                 <- put j.tsv, j.csv, annotation_audiov1.csv here
        └── Output/                <- results are written here (created automatically)

If you need to point at different files/folders, either:
  - drop your files into Input/ using the same filenames, or
  - override via command-line arguments (see `python analysis_pipeline.py --help`), or
  - override via environment variables VWP_GAZE_TSV / VWP_TRIAL_CSV /
    VWP_ANNOTATION_CSV / VWP_OUTPUT_DIR / VWP_INPUT_DIR.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe backend; plots are saved to file, not shown interactively
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────
# CONFIGURATION — relative to this script's location in the repo
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR  = Path(os.environ.get("VWP_INPUT_DIR", SCRIPT_DIR / "Input"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("VWP_OUTPUT_DIR", SCRIPT_DIR / "Output"))

DEFAULT_GAZE_TSV       = Path(os.environ.get("VWP_GAZE_TSV", DEFAULT_INPUT_DIR / "j.tsv"))
DEFAULT_TRIAL_CSV      = Path(os.environ.get("VWP_TRIAL_CSV", DEFAULT_INPUT_DIR / "j.csv"))
DEFAULT_ANNOTATION_CSV = Path(os.environ.get("VWP_ANNOTATION_CSV", DEFAULT_INPUT_DIR / "annotation_audiov1.csv"))

GROWTH_CURVE_BIN_MS = 50   # time-bin width for growth curves

# ──────────────────────────────────────────────────────────────
# AOI GEOMETRY
# ──────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080
IMG_W,    IMG_H    = 2400, 1400

# Scale-to-fit: height-constrained, pillar-boxed (black bars on left/right)
_SCALE    = SCREEN_H / IMG_H                       # 0.77143
_DISP_W   = IMG_W * _SCALE                         # 1851.43 px
_X_OFFSET = (SCREEN_W - _DISP_W) / 2              # 34.29 px  (each side)


def _to_norm(x_img, y_img):
    """Image pixel → GazePoint normalized (FPOGX/FPOGY in 0-1 range)."""
    x_scr = x_img * _SCALE + _X_OFFSET
    y_scr = y_img * _SCALE
    return x_scr / SCREEN_W, y_scr / SCREEN_H


# AOI bounding boxes in image pixels, measured from 1_pos1_sub.png
# Positions are numbered right-to-left by x-coordinate:
#   pos1 = rightmost (top-right)   pos2 = second from right (bottom-right)
#   pos3 = third from right (bottom-left)  pos4 = leftmost (top-left)
_AOI_IMG_PX = {
    "pos1":    (1836, 468, 2140, 702),   # top-right    (rightmost)
    "pos2":    (1384, 945, 1681, 1134),  # bottom-right (second from right)
    "pos3":    (718,  903, 1009, 1144),  # bottom-left  (third from right)
    "pos4":    (301,  455, 530,  706),   # top-left     (leftmost)
    "subject": (1111, 111, 1289, 688),   # center — for validation only
}

# Convert to normalized screen coords once at import time
AOI_NORM = {}
for _name, (_x1, _y1, _x2, _y2) in _AOI_IMG_PX.items():
    _nx1, _ny1 = _to_norm(_x1, _y1)
    _nx2, _ny2 = _to_norm(_x2, _y2)
    AOI_NORM[_name] = (_nx1, _ny1, _nx2, _ny2)


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_gaze(path):
    """Load GazePoint TSV. Returns full DataFrame with cleaned USER column."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["USER"] = df["USER"].fillna("").astype(str).str.strip()
    return df


def load_trials(path):
    """
    Load OpenSesame CSV. Keep only actual experimental trial rows
    (rows that have a real audio_file value).
    Sort by count_trial_loop so row order matches TSV trial order.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[
        df["audio_file"].notna() &
        (df["audio_file"].astype(str) != "undefined")
    ].copy()
    df = df.sort_values("count_trial_loop").reset_index(drop=True)
    return df


def load_annotation(path, time_unit="auto"):
    """
    Load forced-alignment CSV.

    time_unit : "auto" | "seconds" | "ms"
        Forced-alignment tools differ in whether 'start'/'end' are reported
        in seconds (e.g. 1.26) or milliseconds (e.g. 1260). This function
        can auto-detect it: real sentences in this paradigm run well under
        20 seconds, so if the median 'end' value is > 20 we assume the
        column is in milliseconds and convert to seconds. Pass "seconds" or
        "ms" explicitly to skip the heuristic if you already know the unit.

    Returns one row per (sentence_id, condition) with columns:
        id, condition, verb_offset_s, target_onset_s, target_word
    All returned times are in seconds, regardless of the input unit.
    """
    df = pd.read_csv(path)

    if time_unit == "auto":
        median_end = df["end"].median()
        detected = "ms" if median_end > 20 else "seconds"
        print(f"  [annotation] time unit auto-detected as '{detected}' "
              f"(median 'end' value = {median_end:.3g})")
        time_unit = detected

    if time_unit == "ms":
        df = df.copy()
        df["start"] = df["start"] / 1000.0
        df["end"]   = df["end"]   / 1000.0
    elif time_unit != "seconds":
        raise ValueError(f"Unknown time_unit '{time_unit}' — must be 'auto', 'seconds', or 'ms'")

    verbs = (df[df["WordRole"] == "ROOT"]
             [["id", "SentenceRole", "end"]]
             .rename(columns={"end": "verb_offset_s",
                               "SentenceRole": "condition"}))
    targets = (df[df["WordRole"] == "dobj"]
               [["id", "SentenceRole", "start", "word"]]
               .rename(columns={"start": "target_onset_s",
                                 "word":  "target_word",
                                 "SentenceRole": "condition"}))
    return verbs.merge(targets, on=["id", "condition"])


# ──────────────────────────────────────────────────────────────
# TRIAL SEGMENTATION (GazePoint clock)
# ──────────────────────────────────────────────────────────────

def segment_trials(gaze_df):
    """
    Split the full gaze stream into per-trial segments.

    Uses START_TRIAL markers as boundaries: trial i spans from
    START_TRIAL[i] up to (but not including) START_TRIAL[i+1], or to the end
    of the file for the last trial.  This is robust to STOP_TRIAL being absent
    or under-counted (a known quirk in some OpenSesame/GazePoint setups).

    AUDIO_FILE_ONSET_LOG must match START_TRIAL count 1-to-1.

    Returns a list of dicts (one per trial, in chronological order):
        trial_idx      : 0-based integer
        audio_onset_s  : GazePoint TIME (s) when audio started
        gaze           : DataFrame slice for this trial
    """
    events = gaze_df[gaze_df["USER"] != ""][["TIME", "USER"]].copy()
    starts = events[events["USER"] == "START_TRIAL"]["TIME"].values
    audio  = events[events["USER"] == "AUDIO_FILE_ONSET_LOG"]["TIME"].values

    if len(starts) == 0:
        raise RuntimeError(
            "No START_TRIAL markers found in the USER column. "
            "Check that j.tsv is the correct file."
        )
    if len(audio) != len(starts):
        raise RuntimeError(
            f"START_TRIAL count ({len(starts)}) ≠ AUDIO_FILE_ONSET_LOG count "
            f"({len(audio)}). Files may be mismatched."
        )

    segments = []
    for i in range(len(starts)):
        t_start = starts[i]
        t_end   = starts[i + 1] if i + 1 < len(starts) else gaze_df["TIME"].max()
        mask    = (gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)
        segments.append({
            "trial_idx":     i,
            "audio_onset_s": audio[i],
            "gaze":          gaze_df[mask].copy(),
        })
    return segments


# ──────────────────────────────────────────────────────────────
# FIXATION EXTRACTION
# ──────────────────────────────────────────────────────────────

def collapse_to_fixations(gaze_seg):
    """
    Collapse a per-trial gaze segment into one row per fixation.
    Uses GazePoint's onboard fixation detection (FPOGID groups) to define
    *which samples belong to which fixation*, but derives the fixation's
    X/Y position by averaging the raw per-sample BPOGX/BPOGY ("Best Point
    Of Gaze" — GazePoint's fused left+right-eye estimate) rather than
    trusting GazePoint's own pre-smoothed FPOGX/FPOGY. Validity is checked
    on BPOGV (not FPOGV) since BPOGV is what governs BPOGX/BPOGY.

    FPOGS/FPOGD (fixation start time and duration) still come from
    GazePoint's own fixation algorithm — BPOG is a raw per-sample stream
    and has no notion of fixation duration on its own.

    Returns DataFrame with columns: FPOGID, FPOGX, FPOGY, FPOGS, FPOGD
        FPOGX, FPOGY = mean BPOGX, BPOGY across valid samples in the fixation
        FPOGS = fixation start (GazePoint seconds)
        FPOGD = fixation duration (GazePoint seconds, taken from last sample
                in each group — GazePoint continuously updates FPOGD)
    """
    valid = gaze_seg[gaze_seg["BPOGV"] == 1]
    if valid.empty:
        return pd.DataFrame(columns=["FPOGID", "FPOGX", "FPOGY", "FPOGS", "FPOGD"])

    fix = (
        valid.groupby("FPOGID", sort=False)
        .agg(
            FPOGX=("BPOGX", "mean"),
            FPOGY=("BPOGY", "mean"),
            FPOGS=("FPOGS", "first"),
            FPOGD=("FPOGD", "last"),    # last = fully accumulated duration
        )
        .reset_index()
    )
    return fix


# ──────────────────────────────────────────────────────────────
# AOI CLASSIFICATION
# ──────────────────────────────────────────────────────────────

def classify_aoi(fpogx, fpogy):
    """
    Return the AOI name if the gaze point falls inside a bounding box.
    Checks pos1–pos4 and subject (center figure).
    Returns 'elsewhere' if outside all named boxes.
    """
    for name in ("pos1", "pos2", "pos3", "pos4", "subject"):
        x1, y1, x2, y2 = AOI_NORM[name]
        if x1 <= fpogx <= x2 and y1 <= fpogy <= y2:
            return name
    return "elsewhere"


# ──────────────────────────────────────────────────────────────
# CRITICAL WINDOW EXTRACTION
# ──────────────────────────────────────────────────────────────

def extract_critical_window(fix_df, audio_onset_s, verb_offset_s, target_onset_s):
    """
    Keep only fixations that overlap the critical window
        [audio_onset_s + verb_offset_s,  audio_onset_s + target_onset_s]
    (all times in GazePoint seconds).

    Clips each fixation to the window and records clipped duration in ms.
    Adds an 'aoi' column via classify_aoi().

    Returns a DataFrame (may be empty if no fixations fall in the window).
    """
    win_start = audio_onset_s + verb_offset_s
    win_end   = audio_onset_s + target_onset_s

    if win_end <= win_start:
        return pd.DataFrame()

    df = fix_df.copy()
    df["fix_end_s"] = df["FPOGS"] + df["FPOGD"]

    # Keep fixations that overlap [win_start, win_end]
    overlap = (df["FPOGS"] < win_end) & (df["fix_end_s"] > win_start)
    df = df[overlap].copy()
    if df.empty:
        return df

    # Clip to window
    df["clip_start_s"] = df["FPOGS"].clip(lower=win_start)
    df["clip_end_s"]   = df["fix_end_s"].clip(upper=win_end)
    df["clip_dur_ms"]  = (df["clip_end_s"] - df["clip_start_s"]) * 1000
    df = df[df["clip_dur_ms"] > 0].copy()

    # Classify AOI
    df["aoi"] = [classify_aoi(x, y) for x, y in zip(df["FPOGX"], df["FPOGY"])]
    return df


def extract_critical_window_raw(gaze_seg, audio_onset_s, verb_offset_s, target_onset_s):
    """
    Like extract_critical_window, but keeps every individual raw gaze
    sample (BPOGX/BPOGY, one row per ~16.7 ms sample at 60 Hz) inside the
    critical window, rather than collapsing to fixations first.

    Used for raw-gaze-point visualizations (density/scatter of literally
    where the eye was on every sample), as opposed to the fixation-level
    plots which show GazePoint's own fixation clusters.

    Returns a DataFrame with columns: TIME, BPOGX, BPOGY, aoi
    (may be empty if no valid samples fall in the window).
    """
    win_start = audio_onset_s + verb_offset_s
    win_end   = audio_onset_s + target_onset_s
    if win_end <= win_start:
        return pd.DataFrame(columns=["TIME", "BPOGX", "BPOGY", "aoi"])

    mask = (
        (gaze_seg["TIME"] >= win_start) & (gaze_seg["TIME"] < win_end) &
        (gaze_seg["BPOGV"] == 1)
    )
    df = gaze_seg.loc[mask, ["TIME", "BPOGX", "BPOGY"]].copy()
    if df.empty:
        return df

    df["aoi"] = [classify_aoi(x, y) for x, y in zip(df["BPOGX"], df["BPOGY"])]
    return df


# ──────────────────────────────────────────────────────────────
# METRICS PER TRIAL
# ──────────────────────────────────────────────────────────────

def _entropy(probs):
    """Shannon entropy (bits) of a list of probabilities."""
    val = -sum(p * np.log2(p) for p in probs if p > 0)
    return float(abs(val))   # abs() prevents -0.0 display artefact


def compute_trial_metrics(crit_fix, trial_idx, subject_nr,
                           condition, sentence, target_word, target_position,
                           window_ms):
    """
    From critical-window fixations for one trial compute:

    4-way gaze distribution (prop_pos1–4)
      Normalised over fixation time spent on the 4 object slots only.
      These four values sum to 1.0 and are directly comparable to the
      LLM's renormalised 4-way surprisal distribution.
      Trials where the participant never looked at any object slot during
      the window get prop_pos* = 0.0 and are flagged with no_object_gaze=True.

    prop_subject (informational)
      Proportion of time on the subject figure, normalised over all 5 named
      AOIs.  Not included in the 4-way sum.

    target_advantage : P(target) - mean(P(distractor_1..3))
    fixation_entropy : Shannon entropy of the 4-way distribution (bits; max=2)
    """
    obj_keys = ["pos1", "pos2", "pos3", "pos4"]
    raw_obj  = {k: 0.0 for k in obj_keys}
    raw_subj = 0.0

    if not crit_fix.empty:
        for k in obj_keys:
            raw_obj[k] = crit_fix[crit_fix["aoi"] == k]["clip_dur_ms"].sum()
        raw_subj = crit_fix[crit_fix["aoi"] == "subject"]["clip_dur_ms"].sum()

    obj_total  = sum(raw_obj.values())
    all_total  = obj_total + raw_subj

    # 4-way distribution: normalised over object-slot time only
    props = {k: (v / obj_total if obj_total > 0 else 0.0)
             for k, v in raw_obj.items()}

    # Subject proportion: out of all named AOI time
    prop_subj = raw_subj / all_total if all_total > 0 else 0.0

    tgt_key       = f"pos{target_position}"
    prop_target   = props[tgt_key]
    dist_props    = [props[k] for k in obj_keys if k != tgt_key]
    tgt_advantage = prop_target - float(np.mean(dist_props))

    return {
        "subject_nr":       subject_nr,
        "trial":            trial_idx,
        "condition":        condition,
        "sentence":         sentence,
        "target_word":      target_word,
        "target_position":  target_position,
        # ── 4-way gaze distribution (sums to 1 over object slots) ──
        "prop_pos1":        round(props["pos1"], 6),
        "prop_pos2":        round(props["pos2"], 6),
        "prop_pos3":        round(props["pos3"], 6),
        "prop_pos4":        round(props["pos4"], 6),
        # ── subject (informational, not part of the 4-way sum) ──
        "prop_subject":     round(prop_subj, 6),
        # ── derived metrics ──
        "prop_target":      round(prop_target,   6),
        "target_advantage": round(tgt_advantage, 6),
        "fixation_entropy": round(_entropy(list(props.values())), 6),
        "no_object_gaze":   obj_total == 0,
        "total_window_ms":  round(window_ms, 2),
        "obj_gaze_ms":      round(obj_total, 2),
    }


# ──────────────────────────────────────────────────────────────
# GROWTH CURVES
# ──────────────────────────────────────────────────────────────

def compute_growth_curve(crit_fix, audio_onset_s, verb_offset_s,
                          target_onset_s, bin_ms=50.0):
    """
    Compute proportion of each bin spent fixating each AOI.
    Bins are aligned to verb offset (bin 0 = [verb_offset, verb_offset+50ms]).
    Returns DataFrame with columns: bin_start_ms, aoi, prop_fixating
    """
    win_start = audio_onset_s + verb_offset_s
    win_end   = audio_onset_s + target_onset_s
    win_len   = (win_end - win_start) * 1000

    if win_len <= 0 or crit_fix.empty:
        return pd.DataFrame(columns=["bin_start_ms", "aoi", "prop_fixating"])

    bins = np.arange(0, win_len, bin_ms)
    rows = []
    for b in bins:
        abs_b_start = win_start + b / 1000
        abs_b_end   = win_start + (b + bin_ms) / 1000
        if abs_b_end > win_end:
            abs_b_end = win_end
        actual_bin_ms = (abs_b_end - abs_b_start) * 1000
        if actual_bin_ms <= 0:
            continue
        for pos in list(range(1, 5)) + ["subject"]:
            aoi_label = f"pos{pos}" if pos != "subject" else "subject"
            seg     = crit_fix[crit_fix["aoi"] == aoi_label]
            overlap = 0.0
            for _, fix in seg.iterrows():
                o = (min(fix["clip_end_s"], abs_b_end) -
                     max(fix["clip_start_s"], abs_b_start))
                if o > 0:
                    overlap += o * 1000
            rows.append({
                "bin_start_ms":  b,
                "aoi":           aoi_label,
                "prop_fixating": round(overlap / actual_bin_ms, 6),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# OPTIONAL: ADD OBJECT NAMES FROM STIMULUS LIST
# ──────────────────────────────────────────────────────────────

def add_object_names(props_df, stimulus_list_path):
    """
    Merge distractor object names into props_df from a stimulus list CSV.

    The stimulus list CSV must have columns:
        sentence_id   (integer, matching annotation id)
        pos1_object, pos2_object, pos3_object, pos4_object

    Returns an enriched copy of props_df with those columns added.
    This function is a placeholder — call it once you have the stimulus list.
    """
    stim = pd.read_csv(stimulus_list_path)
    # Derive sentence_id from audio_file if not already in props_df
    # (requires that props_df was built with 'audio_id' column — add if needed)
    return props_df.merge(stim, left_on="audio_id", right_on="sentence_id", how="left")


# ──────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ──────────────────────────────────────────────────────────────

_CONDITION_COLORS = {"restrictive": "#1f77b4", "non-restrictive": "#d62728"}
_AOI_COLORS = {"pos1": "#1f77b4", "pos2": "#ff7f0e", "pos3": "#2ca02c",
               "pos4": "#d62728", "subject": "#7f7f7f"}


def _condition_color(cond, fallback_cycle=iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])):
    return _CONDITION_COLORS.get(cond, next(fallback_cycle))


def plot_growth_curve(growth_df, out_path):
    """
    Classic visual-world-paradigm growth curve: proportion of fixation time
    on the target vs. the average distractor, over time bins from verb
    offset to target onset, plotted separately per condition.
    """
    if growth_df.empty:
        print("  [plot] skipped growth curve — no growth-curve data")
        return

    df = growth_df.copy()
    df["role"] = np.where(
        df["aoi"] == "subject", "subject",
        np.where(df["aoi"] == ("pos" + df["target_pos"].astype(str)), "target", "distractor")
    )

    agg = (df.groupby(["condition", "bin_start_ms", "role"])["prop_fixating"]
             .mean().reset_index())

    fig, axes = plt.subplots(1, agg["condition"].nunique(), figsize=(6 * agg["condition"].nunique(), 4.5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    role_style = {"target": dict(color="#1f77b4", linestyle="-", linewidth=2.2, label="Target"),
                  "distractor": dict(color="#d62728", linestyle="--", linewidth=2.2, label="Distractor (mean)"),
                  "subject": dict(color="#7f7f7f", linestyle=":", linewidth=1.6, label="Subject")}

    for ax, cond in zip(axes, sorted(agg["condition"].unique())):
        sub = agg[agg["condition"] == cond]
        for role in ("target", "distractor", "subject"):
            r = sub[sub["role"] == role].sort_values("bin_start_ms")
            if r.empty:
                continue
            ax.plot(r["bin_start_ms"], r["prop_fixating"], **role_style[role])
        ax.set_title(cond)
        ax.set_xlabel("Time from verb offset (ms)")
        ax.axhline(0.25, color="black", linewidth=0.7, linestyle=":", alpha=0.4)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Proportion of time fixating")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Fixation growth curves: verb offset → target onset", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_fixation_proportions(props_df, out_path):
    """
    Bar chart of mean 4-way AOI fixation proportions (pos1-4) by condition,
    matching the CONDITION SUMMARY table printed to console.
    """
    if props_df.empty:
        print("  [plot] skipped fixation proportions — no trial data")
        return

    pos_cols = ["prop_pos1", "prop_pos2", "prop_pos3", "prop_pos4"]
    summary = props_df.groupby("condition")[pos_cols].mean()

    conditions = summary.index.tolist()
    n_cond = len(conditions)
    x = np.arange(len(pos_cols))
    width = 0.8 / n_cond

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, cond in enumerate(conditions):
        ax.bar(x + i * width - (0.8 - width) / 2, summary.loc[cond, pos_cols].values,
               width=width, label=cond, color=_condition_color(cond))

    ax.set_xticks(x)
    ax.set_xticklabels(["pos1\n(top-right)", "pos2\n(top-left)",
                         "pos3\n(bottom-left)", "pos4\n(bottom-right)"])
    ax.set_ylabel("Mean proportion of fixation time")
    ax.set_title("4-way AOI fixation distribution by condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_target_advantage(props_df, out_path):
    """
    Box plot (with individual sentence points overlaid) of target_advantage
    = P(target) - mean(P(distractors)), by condition.
    """
    if props_df.empty:
        print("  [plot] skipped target advantage — no trial data")
        return

    conditions = sorted(props_df["condition"].unique())
    data = [props_df.loc[props_df["condition"] == c, "target_advantage"].values for c in conditions]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    try:
        bp = ax.boxplot(data, tick_labels=conditions, patch_artist=True, showfliers=False, widths=0.5)
    except TypeError:  # matplotlib < 3.9 doesn't know tick_labels
        bp = ax.boxplot(data, labels=conditions, patch_artist=True, showfliers=False, widths=0.5)
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(_condition_color(cond))
        patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for i, (cond, vals) in enumerate(zip(conditions, data), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=_condition_color(cond),
                   s=22, alpha=0.75, zorder=3, edgecolor="white", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Target advantage  (P(target) − mean P(distractor))")
    ax.set_title("Target advantage by condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_aoi_qc(crit_df, out_path):
    """
    Quality-control scatter plot: every raw critical-window fixation
    (FPOGX, FPOGY in normalized screen space) plotted over the AOI
    bounding boxes, colored by assigned AOI. Useful for sanity-checking
    that AOI_NORM boxes actually line up with where people looked.
    """
    if crit_df.empty:
        print("  [plot] skipped AOI QC plot — no critical-window fixations")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.3))

    for name, (x1, y1, x2, y2) in AOI_NORM.items():
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   facecolor=_AOI_COLORS.get(name, "#999999"),
                                   alpha=0.15, edgecolor=_AOI_COLORS.get(name, "#999999"),
                                   linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x1, y1 - 0.015, name, fontsize=8, color=_AOI_COLORS.get(name, "#999999"))

    for aoi, sub in crit_df.groupby("aoi"):
        ax.scatter(sub["FPOGX"], sub["FPOGY"], s=10, alpha=0.6,
                   color=_AOI_COLORS.get(aoi, "#000000"), label=aoi, edgecolor="none")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)   # y=0 at top, matching screen coordinates
    ax.set_xlabel("FPOGX (normalized screen space)")
    ax.set_ylabel("FPOGY (normalized screen space)")
    ax.set_title("QC: critical-window fixations vs. AOI boxes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_raw_gaze_points(raw_df, out_path):
    """
    Raw gaze point visualization: every individual valid BPOGX/BPOGY
    sample captured during the critical window (verb offset -> target
    onset), across all trials, plotted as a semi-transparent scatter over
    the AOI boxes. Unlike plot_aoi_qc (which shows GazePoint's fixation
    clusters, one point per fixation), this shows literally every ~16.7 ms
    sample, so density/color intensity reflects how much raw dwell time
    accumulated in each region — closer to a gaze heatmap.
    """
    if raw_df.empty:
        print("  [plot] skipped raw gaze points — no raw samples in critical windows")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.3))

    # Density as a 2D histogram (heatmap), AOI boxes drawn on top for reference
    hb = ax.hexbin(raw_df["BPOGX"], raw_df["BPOGY"], gridsize=60, extent=(0, 1, 0, 1),
                    cmap="viridis", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="Raw sample count (log scale)")

    for name, (x1, y1, x2, y2) in AOI_NORM.items():
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   facecolor="none", edgecolor="white",
                                   linewidth=1.3, linestyle="--")
        ax.add_patch(rect)
        ax.text(x1, y1 - 0.015, name, fontsize=8, color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)   # y=0 at top, matching screen coordinates
    ax.set_xlabel("BPOGX (normalized screen space)")
    ax.set_ylabel("BPOGY (normalized screen space)")
    ax.set_title("Raw gaze point density: critical window, all trials (BPOG samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def generate_visualizations(props_df, crit_df, growth_df, raw_gaze_df, plots_dir):
    """Generate and save all plots to plots_dir. Never raises — a failed
    plot is logged and skipped so it can't take down the rest of the pipeline."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating visualizations...")
    for fn, args, filename in [
        (plot_growth_curve,        (growth_df,),   "growth_curve.png"),
        (plot_fixation_proportions,(props_df,),    "fixation_proportions.png"),
        (plot_target_advantage,    (props_df,),    "target_advantage.png"),
        (plot_aoi_qc,               (crit_df,),    "aoi_fixation_qc.png"),
        (plot_raw_gaze_points,      (raw_gaze_df,),"raw_gaze_points.png"),
    ]:
        try:
            fn(*args, plots_dir / filename)
        except Exception as e:
            print(f"  [plot] FAILED ({filename}): {e}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Eye-Tracking Visual World Paradigm analysis pipeline. "
                     "By default reads j.tsv, j.csv and annotation_audiov1.csv "
                     "from the Input/ folder next to this script, and writes "
                     "results to the Output/ folder next to this script."
    )
    p.add_argument("--gaze-tsv", type=Path, default=DEFAULT_GAZE_TSV,
                    help=f"Path to GazePoint TSV (default: {DEFAULT_GAZE_TSV})")
    p.add_argument("--trial-csv", type=Path, default=DEFAULT_TRIAL_CSV,
                    help=f"Path to OpenSesame trial CSV (default: {DEFAULT_TRIAL_CSV})")
    p.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION_CSV,
                    help=f"Path to forced-alignment annotation CSV (default: {DEFAULT_ANNOTATION_CSV})")
    p.add_argument("--annotation-time-unit", choices=["auto", "seconds", "ms"], default="auto",
                    help="Unit of the 'start'/'end' columns in the annotation CSV. "
                         "'auto' (default) detects seconds vs. milliseconds automatically.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Directory to write results to (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--skip-plots", action="store_true",
                    help="Skip generating PNG visualizations (only write CSV outputs)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    gaze_tsv       = args.gaze_tsv
    trial_csv      = args.trial_csv
    annotation_csv = args.annotation_csv
    out            = args.output_dir

    for f, label in [(gaze_tsv, "gaze TSV"), (trial_csv, "trial CSV"), (annotation_csv, "annotation CSV")]:
        if not f.exists():
            sys.exit(
                f"ERROR: {label} not found at {f}\n"
                f"Place your input files in {DEFAULT_INPUT_DIR} "
                f"(using the default filenames j.tsv, j.csv, annotation_audiov1.csv), "
                f"or pass explicit paths via command-line flags. Run with --help for options."
            )

    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    print("Loading data...")
    gaze_df  = load_gaze(gaze_tsv)
    trial_df = load_trials(trial_csv)
    ann_df   = load_annotation(annotation_csv, time_unit=args.annotation_time_unit)

    subject_nr = int(trial_df["subject_nr"].iloc[0])
    print(f"  Subject {subject_nr}  |  {len(trial_df)} trials in CSV")

    # ── Segment gaze stream ───────────────────────────────────
    print("Segmenting gaze stream into trials...")
    segments = segment_trials(gaze_df)
    print(f"  {len(segments)} trial segments found in TSV")

    if len(segments) != len(trial_df):
        print(
            f"  WARNING: segment count ({len(segments)}) ≠ trial CSV rows "
            f"({len(trial_df)}). Using first {min(len(segments), len(trial_df))}."
        )

    n_trials = min(len(segments), len(trial_df))

    # ── Build annotation lookup ───────────────────────────────
    # Key: (sentence_id_int, condition_str)  →  (verb_offset_s, target_onset_s, target_word)
    ann_lookup = {}
    for _, row in ann_df.iterrows():
        ann_lookup[(int(row["id"]), str(row["condition"]))] = (
            float(row["verb_offset_s"]),
            float(row["target_onset_s"]),
            str(row["target_word"]),
        )

    # ── Per-trial processing ──────────────────────────────────
    print("Processing trials...")
    all_metrics   = []
    all_crit_fix  = []
    all_growth    = []
    all_raw_gaze  = []

    for i in range(n_trials):
        seg    = segments[i]
        t_row  = trial_df.iloc[i]

        # Parse sentence id and condition from audio filename
        # e.g. "2_r.wav" → audio_id=2, condition from t_row
        audio_file = str(t_row["audio_file"])
        try:
            audio_id = int(audio_file.split("_")[0])
        except ValueError:
            print(f"  [skip] Trial {i}: cannot parse audio_file '{audio_file}'")
            continue

        condition      = str(t_row["condition"])
        sentence       = str(t_row["sentence"])
        target_position = int(t_row["target_position"])

        ann_key = (audio_id, condition)
        if ann_key not in ann_lookup:
            print(f"  [skip] Trial {i}: no annotation for {ann_key}")
            continue

        verb_offset_s, target_onset_s, target_word = ann_lookup[ann_key]
        window_ms = (target_onset_s - verb_offset_s) * 1000

        # Fixations for this trial
        fix_df = collapse_to_fixations(seg["gaze"])

        # Critical window
        crit = extract_critical_window(
            fix_df, seg["audio_onset_s"], verb_offset_s, target_onset_s
        )

        # Raw (per-sample) gaze in the critical window, for gaze-point plots
        raw = extract_critical_window_raw(
            seg["gaze"], seg["audio_onset_s"], verb_offset_s, target_onset_s
        )
        if not raw.empty:
            raw = raw.copy()
            raw["trial"]     = i
            raw["condition"] = condition
            all_raw_gaze.append(raw)

        if crit.empty:
            print(f"  [warn] Trial {i} ({condition}, '{target_word}'): "
                  f"no fixations in critical window ({window_ms:.0f} ms)")

        # Metrics
        metrics = compute_trial_metrics(
            crit, i, subject_nr,
            condition, sentence, target_word, target_position, window_ms
        )
        metrics["sentence_id"] = audio_id   # 0-indexed sentence ID from annotation
        metrics["audio_id"]    = audio_id   # keep for later stimulus-list merge
        all_metrics.append(metrics)

        if not crit.empty:
            crit = crit.copy()
            crit["trial"]        = i
            crit["condition"]    = condition
            crit["subject_nr"]   = subject_nr
            crit["target_word"]  = target_word
            crit["target_position"] = target_position
            all_crit_fix.append(crit)

            gc = compute_growth_curve(
                crit, seg["audio_onset_s"], verb_offset_s, target_onset_s,
                bin_ms=GROWTH_CURVE_BIN_MS
            )
            if not gc.empty:
                gc["trial"]      = i
                gc["condition"]  = condition
                gc["target_pos"] = target_position
                all_growth.append(gc)

    # ── Assemble DataFrames ───────────────────────────────────
    props_df  = pd.DataFrame(all_metrics)
    crit_df   = (pd.concat(all_crit_fix, ignore_index=True)
                 if all_crit_fix else pd.DataFrame())
    growth_df = (pd.concat(all_growth, ignore_index=True)
                 if all_growth else pd.DataFrame())
    raw_gaze_df = (pd.concat(all_raw_gaze, ignore_index=True)
                   if all_raw_gaze else pd.DataFrame())

    # ── Print summary ─────────────────────────────────────────
    # Sort by sentence_id so results are sentence-by-sentence
    props_df = props_df.sort_values("sentence_id").reset_index(drop=True)

    print("\n" + "="*105)
    print("GAZE DISTRIBUTION PER SENTENCE  (critical window: verb offset → target onset)")
    print("prop_pos1–4 sum to 1.0 per sentence  |  pos1=rightmost … pos4=leftmost  |  prop_subject informational")
    print("="*105)
    display_cols = ["sentence_id", "condition", "sentence", "target_word", "target_position",
                    "prop_pos1", "prop_pos2", "prop_pos3", "prop_pos4",
                    "prop_subject", "no_object_gaze", "total_window_ms"]
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 45)
    print(props_df[display_cols].to_string(index=False))

    print("\n" + "="*90)
    print("CONDITION SUMMARY  (mean across sentences — sentences with no object gaze included as 0)")
    print("="*90)
    summary = (props_df
               .groupby("condition")[["prop_pos1","prop_pos2","prop_pos3","prop_pos4",
                                      "prop_subject","prop_target","target_advantage",
                                      "fixation_entropy"]]
               .mean().round(3))
    print(summary.to_string())

    # ── Save outputs ──────────────────────────────────────────
    props_path  = out / "fixation_proportions.csv"
    crit_path   = out / "fixation_critical_window.csv"
    growth_path = out / "growth_curves.csv"

    props_df.to_csv(props_path, index=False)
    print(f"\nSaved: {props_path}  ({len(props_df)} trial rows)")

    if not crit_df.empty:
        crit_df.to_csv(crit_path, index=False)
        print(f"Saved: {crit_path}  ({len(crit_df)} fixation events)")

    if not growth_df.empty:
        growth_df.to_csv(growth_path, index=False)
        print(f"Saved: {growth_path}  ({len(growth_df)} time-bin rows)")

    if not raw_gaze_df.empty:
        raw_gaze_path = out / "raw_gaze_critical_window.csv"
        raw_gaze_df.to_csv(raw_gaze_path, index=False)
        print(f"Saved: {raw_gaze_path}  ({len(raw_gaze_df)} raw gaze samples)")

    # ── Plots ─────────────────────────────────────────────────
    if not args.skip_plots:
        generate_visualizations(props_df, crit_df, growth_df, raw_gaze_df, out / "plots")

    return props_df, crit_df, growth_df


if __name__ == "__main__":
    props_df, crit_df, growth_df = main()