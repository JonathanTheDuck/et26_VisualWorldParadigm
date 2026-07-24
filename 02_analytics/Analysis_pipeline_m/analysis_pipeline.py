"""
Eye-Tracking Visual World Paradigm — Analysis Pipeline
=======================================================
Inputs
------
  {subject}.tsv, {subject}.csv   One pair per participant, e.g.:
                                    subject-1.tsv + subject-1.csv
                                    subject-2.tsv + subject-2.csv
                                  (legacy single-subject naming j.tsv + j.csv
                                  also works — it's just one more matching pair)
  annotation_audiov1.csv         Forced-alignment word timing per sentence
                                  (shared across all subjects/stimuli)

Outputs (saved to OUTPUT_DIR)
------
  fixation_proportions.csv       Per-trial 4-way fixation proportions + metrics, all subjects pooled
  fixation_critical_window.csv   Every fixation event inside the critical window, all subjects
  growth_curves.csv              Fixation proportions in 50 ms bins (growth curves), all subjects
  raw_gaze_critical_window.csv   Every raw (per-sample) gaze point inside the critical window
  word_probabilities.csv         Per-word 4-way gaze probability distribution (raw-gaze average
                                  over EVERY word's span, not just the critical window) — the
                                  human-gaze analogue of an LLM's per-word predicted probability

  plots/growth_curve.png              Target vs. distractor fixation curves over time, by condition
  plots/fixation_proportions.png      Mean 4-way AOI fixation proportions, by condition
  plots/target_advantage.png          Target-vs-distractor advantage, by condition (box + points)
  plots/aoi_fixation_qc.png           Fixation-level events plotted over the AOI boxes (sanity check)
  plots/raw_gaze_points.png           Raw per-sample gaze point density over the AOI boxes
  plots/word_probability_curve.png    Word-by-word target vs. distractor gaze probability

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
        ├── Input/                 <- put {subject}.tsv/{subject}.csv pairs +
        │                             annotation_audiov1.csv here
        └── Output/                <- results are written here (created automatically)

By default the script auto-discovers every {subject}.tsv + {subject}.csv
pair in Input/ (e.g. subject-1.tsv/subject-1.csv, subject-2.tsv/subject-2.csv,
...) and processes all of them, pooling results with a subject_nr column.

If you need to point at different files/folders, either:
  - drop your files into Input/ using the {subject}.tsv/{subject}.csv pattern,
  - override via command-line arguments (see `python analysis_pipeline.py --help`) —
    e.g. --input-dir for a different folder, or --gaze-tsv/--trial-csv together
    for a single explicit file pair, or
  - override via environment variables VWP_INPUT_DIR / VWP_OUTPUT_DIR /
    VWP_ANNOTATION_CSV.
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

DEFAULT_ANNOTATION_CSV = Path(os.environ.get("VWP_ANNOTATION_CSV", DEFAULT_INPUT_DIR / "annotation_audiov2.csv"))

GROWTH_CURVE_BIN_MS = 50   # time-bin width for growth curves

# ──────────────────────────────────────────────────────────────
# AOI GEOMETRY
# ──────────────────────────────────────────────────────────────
# AOI positions below are adopted from the teammate's human_pipeline.ipynb
# (verified visually against real gaze scatter — "image and AOI match
# perfectly"), NOT re-derived from the image-pixel scaling formula this
# script used previously. That earlier formula-derived version is kept
# only as a comment for reference; positions_teammate is the source of truth.
#
# The teammate's positions/box size were partly hand-adjusted (not a pure
# uniform scale of the 400x400 image-space boxes), so they are hard-coded
# here in SCREEN pixel space (1920x1080) exactly as verified, then
# converted to normalized (0-1) coordinates to match BPOGX/BPOGY.
#
# The "subject" box is NOT in the teammate's notebook (they didn't define
# one) — it's measured directly from the real stimulus images by diffing
# 1_pos1_sub.png against 1_pos1_nosub.png (pixel-difference bounding box),
# then passed through the same image->screen scale-to-fit transform.
SCREEN_W, SCREEN_H = 2560, 1440
IMG_W,    IMG_H    = 2560, 1440

_TEAMMATE_POS_PX = {   # top-left corner, screen pixel space (1920x1080)
    "pos1": (1426, 292),
    "pos2": (1074, 636),
    "pos3": (526,  636),
    "pos4": (174,  292),
}
_TEAMMATE_BOX_W, _TEAMMATE_BOX_H = 320, 309

# Subject bbox measured in IMAGE pixel space (2400x1400) via image diff,
# then converted using the same scale-to-fit transform as the objects.
_SCALE    = min(SCREEN_W / IMG_W, SCREEN_H / IMG_H)      # 0.77143 (height-constrained)
_X_OFFSET = (SCREEN_W - IMG_W * _SCALE) / 2              # 34.29 px pillarbox each side
_Y_OFFSET = (SCREEN_H - IMG_H * _SCALE) / 2              # 0 px (height-constrained)
_SUBJECT_IMG_PX = (1111, 109, 1289, 687)                 # measured via sub/nosub image diff

_sx1, _sy1, _sx2, _sy2 = _SUBJECT_IMG_PX
_SUBJECT_SCREEN_PX = (
    _sx1 * _SCALE + _X_OFFSET, _sy1 * _SCALE + _Y_OFFSET,
    _sx2 * _SCALE + _X_OFFSET, _sy2 * _SCALE + _Y_OFFSET,
)

# Assemble all 5 AOI boxes in SCREEN pixel space, then normalize to 0-1
_AOI_SCREEN_PX = {
    name: (x, y, x + _TEAMMATE_BOX_W, y + _TEAMMATE_BOX_H)
    for name, (x, y) in _TEAMMATE_POS_PX.items()
}
_AOI_SCREEN_PX["subject"] = _SUBJECT_SCREEN_PX

AOI_NORM = {
    name: (x1 / SCREEN_W, y1 / SCREEN_H, x2 / SCREEN_W, y2 / SCREEN_H)
    for name, (x1, y1, x2, y2) in _AOI_SCREEN_PX.items()
}


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


def load_annotation_words(path, time_unit="auto"):
    """
    Load the forced-alignment CSV and return EVERY word's timing (not just
    the verb and target), one row per (sentence id, condition, word).

    time_unit : "auto" | "seconds" | "ms"
        Forced-alignment tools differ in whether 'start'/'end' are reported
        in seconds (e.g. 1.26) or milliseconds (e.g. 1260). This function
        can auto-detect it: real sentences in this paradigm run well under
        20 seconds, so if the median 'end' value is > 20 we assume the
        column is in milliseconds and convert to seconds. Pass "seconds" or
        "ms" explicitly to skip the heuristic if you already know the unit.

    Returns columns: id, condition, WordRole, word, start_s, end_s, sentence
    sorted by (id, condition, start_s). All times are in seconds regardless
    of the input unit.
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

    df = df.rename(columns={"SentenceRole": "condition", "start": "start_s", "end": "end_s"})
    cols = ["id", "condition", "WordRole", "word", "start_s", "end_s"]
    if "sentence" in df.columns:
        cols.append("sentence")
    return (df[cols]
            .sort_values(["id", "condition", "start_s"])
            .reset_index(drop=True))


def load_annotation(path, time_unit="auto"):
    """
    Load forced-alignment CSV, reduced to just the verb-offset / target-onset
    pair used for the critical-window analysis (built on load_annotation_words).

    Returns one row per (sentence_id, condition) with columns:
        id, condition, verb_offset_s, target_onset_s, target_word
    All returned times are in seconds, regardless of the input unit.
    """
    words_df = load_annotation_words(path, time_unit=time_unit)

    verbs = (words_df[words_df["WordRole"] == "ROOT"]
             [["id", "condition", "end_s"]]
             .rename(columns={"end_s": "verb_offset_s"}))
    targets = (words_df[words_df["WordRole"] == "dobj"]
               [["id", "condition", "start_s", "word"]]
               .rename(columns={"start_s": "target_onset_s",
                                 "word":    "target_word"}))
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
# PER-WORD PROBABILITY DISTRIBUTIONS (raw gaze average)
# ──────────────────────────────────────────────────────────────

def _aoi_distribution(aoi_values):
    """
    Shared core: given an iterable of AOI labels (one per gaze
    sample/fixation, unweighted — i.e. every sample counts equally), return
    the 4-way object distribution (normalised over object-slot samples only)
    and the subject proportion (normalised over object + subject samples).

    Returns (props: dict[pos1..4 -> float], prop_subject: float,
             obj_total: int, all_total: int)
    """
    obj_keys = ["pos1", "pos2", "pos3", "pos4"]
    aoi_series = pd.Series(list(aoi_values))

    raw_obj  = {k: int((aoi_series == k).sum()) for k in obj_keys}
    raw_subj = int((aoi_series == "subject").sum())

    obj_total = sum(raw_obj.values())
    all_total = obj_total + raw_subj

    props = {k: (v / obj_total if obj_total > 0 else 0.0) for k, v in raw_obj.items()}
    prop_subj = raw_subj / all_total if all_total > 0 else 0.0

    return props, prop_subj, obj_total, all_total


def compute_word_probabilities(gaze_seg, audio_onset_s, word_rows,
                                trial_idx, subject_nr, condition,
                                sentence_id, target_position):
    """
    For every word in the sentence (word_rows: rows from load_annotation_words
    for this trial's sentence_id + condition, sorted by start_s), average the
    raw valid BPOGX/BPOGY samples that fall within that word's
    [audio_onset_s + start_s, audio_onset_s + end_s) span, and turn that into
    a 4-way object probability distribution — the human-gaze analogue of an
    LLM's per-word predicted probability over the 4 candidate objects.

    Unlike the critical-window metrics (which use GazePoint's fixation
    detection), this works directly on the raw per-sample gaze stream, since
    individual word windows are often too short to contain a full fixation.

    Returns a DataFrame with one row per word:
        word_index, WordRole, word, start_s, end_s, window_ms,
        prop_pos1..4, prop_target, prop_subject, no_object_gaze, n_samples
    """
    rows = []
    for word_index, w in enumerate(word_rows.itertuples(index=False)):
        win_start = audio_onset_s + w.start_s
        win_end   = audio_onset_s + w.end_s
        window_ms = (win_end - win_start) * 1000

        if win_end <= win_start:
            wdf = pd.DataFrame(columns=["BPOGX", "BPOGY"])
        else:
            mask = (
                (gaze_seg["TIME"] >= win_start) & (gaze_seg["TIME"] < win_end) &
                (gaze_seg["BPOGV"] == 1)
            )
            wdf = gaze_seg.loc[mask, ["BPOGX", "BPOGY"]]

        if wdf.empty:
            aoi_vals = []
        else:
            aoi_vals = [classify_aoi(x, y) for x, y in zip(wdf["BPOGX"], wdf["BPOGY"])]

        props, prop_subj, obj_total, all_total = _aoi_distribution(aoi_vals)
        tgt_key = f"pos{target_position}"

        rows.append({
            "subject_nr":      subject_nr,
            "trial":           trial_idx,
            "condition":       condition,
            "sentence_id":     sentence_id,
            "word_index":      word_index,
            "WordRole":        w.WordRole,
            "word":            w.word,
            "start_s":         round(w.start_s, 4),
            "end_s":           round(w.end_s, 4),
            "window_ms":       round(window_ms, 2),
            "prop_pos1":       round(props["pos1"], 6),
            "prop_pos2":       round(props["pos2"], 6),
            "prop_pos3":       round(props["pos3"], 6),
            "prop_pos4":       round(props["pos4"], 6),
            "prop_target":     round(props[tgt_key], 6),
            "prop_subject":    round(prop_subj, 6),
            "no_object_gaze":  obj_total == 0,
            "n_samples":       len(aoi_vals),
        })
    return pd.DataFrame(rows)


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


import itertools
_FALLBACK_COLOR_CYCLE = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
_condition_color_cache = {}


def _condition_color(cond):
    """
    Color for a condition label. Known conditions (restrictive/
    non-restrictive) use fixed colors; anything else gets a color from the
    matplotlib cycle, assigned once and cached (so the same unknown label
    always gets the same color within a run).

    Note: previously this used dict.get(cond, next(fallback_cycle)), but
    dict.get() evaluates its default argument eagerly on every call
    (Python doesn't short-circuit it), so next() fired even when `cond`
    was already a known key — silently draining a 10-color iterator until
    an unrelated later call crashed with StopIteration. Explicit
    if/else avoids evaluating next() unless it's actually needed.
    """
    if cond in _CONDITION_COLORS:
        return _CONDITION_COLORS[cond]
    if cond not in _condition_color_cache:
        _condition_color_cache[cond] = next(_FALLBACK_COLOR_CYCLE)
    return _condition_color_cache[cond]


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


def plot_word_probability_curve(word_prob_df, out_path):
    """
    Human gaze-derived probability of the target vs. mean distractor,
    plotted across word position in the sentence (word_index), averaged
    over all trials/subjects, one line per condition. This is the
    word-by-word analogue of plot_growth_curve, built from raw gaze
    averaging per word rather than the 50 ms-binned critical window —
    intended to be compared directly against an LLM's per-word predicted
    probability of each object.
    """
    if word_prob_df.empty:
        print("  [plot] skipped word probability curve — no word-level data")
        return

    df = word_prob_df.copy()
    obj_cols = ["prop_pos1", "prop_pos2", "prop_pos3", "prop_pos4"]

    # distractor mean = (sum of all 4 object proportions - target) / 3
    df["prop_distractor_mean"] = (df[obj_cols].sum(axis=1) - df["prop_target"]) / 3

    agg = (df.groupby(["condition", "word_index"])[["prop_target", "prop_distractor_mean"]]
             .mean().reset_index())
    # use the most common WordRole label at each word_index for x-tick labels
    role_labels = (df.groupby("word_index")["WordRole"]
                     .agg(lambda s: s.value_counts().idxmax()))

    conditions = sorted(agg["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.5),
                              sharey=True, squeeze=False)
    axes = axes[0]

    for ax, cond in zip(axes, conditions):
        sub = agg[agg["condition"] == cond].sort_values("word_index")
        ax.plot(sub["word_index"], sub["prop_target"], color="#1f77b4",
                marker="o", linewidth=2.2, label="Target")
        ax.plot(sub["word_index"], sub["prop_distractor_mean"], color="#d62728",
                marker="o", linestyle="--", linewidth=2.2, label="Distractor (mean)")
        ax.axhline(0.25, color="black", linewidth=0.7, linestyle=":", alpha=0.4)
        ax.set_xticks(sub["word_index"])
        ax.set_xticklabels([role_labels.get(wi, "") for wi in sub["word_index"]], fontsize=8)
        ax.set_title(cond)
        ax.set_xlabel("Word (by role in sentence)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean gaze-derived probability")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Word-by-word gaze probability (raw-gaze average per word)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_calibration_accuracy(calib_df, out_path, threshold_deg=1.0):
    """
    Calibration accuracy per subject (from the pygaze calibration report),
    one grouped bar cluster per subject: LX, LY, RX, RY in degrees of
    visual angle. A horizontal line marks the QC threshold (default 1.0
    degree — matching the "< 1 degree acceptable" check).

    calib_df columns expected: subject_id, LX, LY, RX, RY [, group]
    """
    if calib_df.empty:
        print("  [plot] skipped calibration accuracy — no calibration reports found")
        return

    axes_cols = ["LX", "LY", "RX", "RY"]
    subjects = calib_df["subject_id"].tolist()
    x = np.arange(len(subjects))
    width = 0.8 / len(axes_cols)
    colors = ["#1f77b4", "#4fa3d1", "#d62728", "#e8807f"]

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(subjects) + 2), 4.5))
    for i, col in enumerate(axes_cols):
        ax.bar(x + i * width - (0.8 - width) / 2, calib_df[col].values,
               width=width, label=col, color=colors[i])

    ax.axhline(threshold_deg, color="black", linewidth=1.0, linestyle="--",
               alpha=0.6, label=f"QC threshold ({threshold_deg}°)")
    ax.set_xticks(x)
    xtick_labels = (calib_df["subject_id"] + "\n(" + calib_df["group"].astype(str) + ")"
                    if "group" in calib_df.columns else calib_df["subject_id"])
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("Calibration accuracy (degrees of visual angle)")
    ax.set_title("Calibration accuracy per participant (lower = better)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_valid_data_percentage(valid_df, out_path, threshold_pct=90.0):
    """
    % of valid gaze samples (BPOGV==1) per participant, sorted ascending
    so the worst participants are immediately visible. A horizontal line
    marks a QC threshold (default 90% valid).

    valid_df columns expected: subject_id, pct_valid_overall [, group]
    """
    if valid_df.empty:
        print("  [plot] skipped valid data percentage — no gaze data found")
        return

    df = valid_df.sort_values("pct_valid_overall").reset_index(drop=True)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df) + 2), 4.5))
    if "group" in df.columns:
        groups = sorted(df["group"].unique())
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
        colors = [group_color[g] for g in df["group"]]
        for g in groups:
            ax.bar([], [], color=group_color[g], label=f"Group {g}")  # legend handles
        ax.legend(frameon=False, fontsize=8, title="Group")
    else:
        colors = "#1f77b4"

    ax.bar(x, df["pct_valid_overall"], color=colors)
    ax.axhline(threshold_pct, color="black", linewidth=1.0, linestyle="--",
               alpha=0.6, label=f"QC threshold ({threshold_pct}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject_id"], fontsize=9, rotation=0)
    ax.set_ylabel("Valid gaze samples (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Valid data captured per participant (BPOGV == 1)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_gaze_bias(raw_gaze_df, out_path):
    """
    Milestone-4 check ③: "Any systematic left/right gaze bias?"
    Histogram/density of BPOGX across all valid samples (critical-window
    raw gaze), one distribution per subject. Mean should sit close to 0.5
    (screen center) if there's no directional bias.
    """
    if raw_gaze_df.empty:
        print("  [plot] skipped gaze bias — no raw gaze data")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    subjects = sorted(raw_gaze_df["subject_nr"].unique()) if "subject_nr" in raw_gaze_df.columns else [None]
    for subj in subjects:
        sub = raw_gaze_df[raw_gaze_df["subject_nr"] == subj] if subj is not None else raw_gaze_df
        ax.hist(sub["BPOGX"], bins=40, alpha=0.5, density=True, label=f"Subject {subj}" if subj is not None else "All")

    ax.axvline(0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label="Screen center (0.5)")
    ax.set_xlabel("BPOGX (normalized screen space)")
    ax.set_ylabel("Density")
    ax.set_title("Left/right gaze bias check: BPOGX distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_target_metric_distribution(props_df, out_path, metric="prop_target"):
    """
    Assignment-notebook-style histogram/density (their cell 31: accuracy by
    eyetracker), adapted to this study: distribution of a gaze-derived
    target metric (default prop_target) split by condition instead of
    by eyetracker.
    """
    if props_df.empty or metric not in props_df.columns:
        print(f"  [plot] skipped target metric distribution — no '{metric}' data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"{metric}: restrictive vs. non-restrictive (assignment-style comparison)")

    axes[0].set_title("Histogram")
    for cond in sorted(props_df["condition"].unique()):
        vals = props_df.loc[props_df["condition"] == cond, metric]
        axes[0].hist(vals, bins=15, alpha=0.5, label=cond, color=_condition_color(cond))
    axes[0].set_xlabel(metric)
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Density (KDE)")
    for cond in sorted(props_df["condition"].unique()):
        vals = props_df.loc[props_df["condition"] == cond, metric].dropna()
        if len(vals) > 1 and vals.std() > 0:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.min(), vals.max(), 200)
            axes[1].plot(xs, kde(xs), color=_condition_color(cond), label=cond, linewidth=2)
            axes[1].fill_between(xs, kde(xs), alpha=0.2, color=_condition_color(cond))
    axes[1].set_xlabel(metric)
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_target_metric_by_trial_block(props_df, out_path, metric="prop_target", trials_per_block=10):
    """
    Assignment-notebook-style point/box plot (their cells 35-36: accuracy
    by subject x block x eyetracker), adapted here: since this study
    doesn't have explicit experimental blocks, trials are grouped into
    pseudo-blocks of `trials_per_block` consecutive trials (by 'trial'
    order) per subject, as an approximation — flagged clearly on the plot.
    """
    if props_df.empty or metric not in props_df.columns:
        print(f"  [plot] skipped target metric by block — no '{metric}' data")
        return

    df = props_df.copy()
    df["pseudo_block"] = (df["trial"] // trials_per_block) + 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    blocks = sorted(df["pseudo_block"].unique())
    conditions = sorted(df["condition"].unique())
    width = 0.8 / len(conditions)
    x = np.arange(len(blocks))

    for i, cond in enumerate(conditions):
        means, sems = [], []
        for b in blocks:
            vals = df.loc[(df["pseudo_block"] == b) & (df["condition"] == cond), metric]
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem() if len(vals) > 1 else 0)
        ax.errorbar(x + i * width - (0.8 - width) / 2, means, yerr=sems, fmt="o-",
                    color=_condition_color(cond), label=cond, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {b}\n(trials {(b-1)*trials_per_block}-{b*trials_per_block-1})" for b in blocks],
                        fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by pseudo-block and condition\n"
                 f"(pseudo-blocks = {trials_per_block} consecutive trials — no real block variable in this design)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def generate_visualizations(props_df, crit_df, growth_df, raw_gaze_df, word_prob_df,
                             calib_df, valid_df, plots_dir):
    """Generate and save all plots to plots_dir. Never raises — a failed
    plot is logged and skipped so it can't take down the rest of the pipeline."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating visualizations...")
    for fn, args, filename in [
        (plot_growth_curve,          (growth_df,),    "growth_curve.png"),
        (plot_fixation_proportions,  (props_df,),     "fixation_proportions.png"),
        (plot_target_advantage,      (props_df,),     "target_advantage.png"),
        (plot_aoi_qc,                 (crit_df,),     "aoi_fixation_qc.png"),
        (plot_raw_gaze_points,        (raw_gaze_df,), "raw_gaze_points.png"),
        (plot_word_probability_curve, (word_prob_df,),"word_probability_curve.png"),
        (plot_calibration_accuracy,   (calib_df,),    "calibration_accuracy.png"),
        (plot_valid_data_percentage,  (valid_df,),    "valid_data_percentage.png"),
        (plot_gaze_bias,              (raw_gaze_df,), "gaze_bias.png"),
        (plot_target_metric_distribution,   (props_df,), "target_metric_distribution.png"),
        (plot_target_metric_by_trial_block, (props_df,), "target_metric_by_block.png"),
    ]:
        try:
            fn(*args, plots_dir / filename)
        except Exception as e:
            print(f"  [plot] FAILED ({filename}): {e}")


def discover_subjects(input_dir):
    """
    Scan input_dir for per-subject gaze/trial file pairs, e.g.:
        subject-1.tsv + subject-1.csv (+ optional subject-1_log.txt)
        subject-2.tsv + subject-2.csv (+ optional subject-2_log.txt)
        j.tsv + j.csv                 (single-subject / legacy naming)

    A subject is identified by matching stem: for every *.tsv file, look
    for a *.csv file with the exact same stem. Files ending in "_log" are
    never treated as a gaze/trial file. If a matching {stem}_log.txt exists
    (the pygaze init/calibration report), its path is attached too — used
    for the calibration-accuracy QC plot.

    Returns a sorted list of dicts:
        {"subject_id": str, "tsv": Path, "csv": Path, "log": Path or None}
    """
    input_dir = Path(input_dir)
    subjects = []
    for tsv_path in sorted(input_dir.glob("*.tsv")):
        stem = tsv_path.stem
        if stem.endswith("_log"):
            continue
        csv_path = input_dir / f"{stem}.csv"
        if csv_path.exists():
            log_path = input_dir / f"{stem}_log.txt"
            subjects.append({
                "subject_id": stem,
                "tsv": tsv_path,
                "csv": csv_path,
                "log": log_path if log_path.exists() else None,
            })
        else:
            print(f"  [discover] skipping {tsv_path.name}: no matching {stem}.csv found")
    return subjects


# ──────────────────────────────────────────────────────────────
# CALIBRATION & DATA-VALIDITY QC
# ──────────────────────────────────────────────────────────────

def parse_calibration_report(log_path):
    """
    Parse a pygaze init/calibration report (.txt) for the calibration
    accuracy line, e.g.:
        accuracy (degrees): LX=0.49, LY=0.46, RX=0.24, RY=0.78

    Returns a dict {"LX": float, "LY": float, "RX": float, "RY": float},
    or None if the line/file can't be parsed.
    """
    import re
    try:
        text = Path(log_path).read_text()
    except (OSError, FileNotFoundError):
        return None

    m = re.search(
        r"accuracy \(degrees\):\s*LX=([\d.]+),\s*LY=([\d.]+),\s*RX=([\d.]+),\s*RY=([\d.]+)",
        text
    )
    if not m:
        return None
    return {
        "LX": float(m.group(1)),
        "LY": float(m.group(2)),
        "RX": float(m.group(3)),
        "RY": float(m.group(4)),
    }


def compute_valid_data_percentage(gaze_df, segments=None):
    """
    Compute the percentage of valid gaze samples (BPOGV==1) for one subject.

    Returns a dict:
        pct_valid_overall : float, % valid across the whole recording
        pct_valid_per_trial : list[float], % valid within each trial segment
                              (only populated if `segments` is given —
                              matches the "per-trial invalid rate" framing)
    """
    pct_valid_overall = 100.0 * gaze_df["BPOGV"].mean()
    pct_valid_per_trial = []
    if segments is not None:
        for seg in segments:
            g = seg["gaze"]
            if len(g) > 0:
                pct_valid_per_trial.append(100.0 * g["BPOGV"].mean())
    return {"pct_valid_overall": pct_valid_overall, "pct_valid_per_trial": pct_valid_per_trial}


def load_group_assignments(path):
    """
    Load an optional participant-group file (e.g. participant_groupX.csv /
    overview.csv) mapping each subject to a group label.

    Tries to auto-detect the subject-id column (any of: subject_id, subject,
    subject_nr, participant, participant_id — case-insensitive) and the
    group column (any of: group, group_number, group_nr, condition_group
    — case-insensitive). Returns a dict {subject_id_str: group_label_str},
    or an empty dict if the file/columns can't be found.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"  [group] group file not found at {path} — proceeding without group labels")
        return {}

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    subject_col = next((cols_lower[c] for c in
                         ["subject_id", "subject", "subject_nr", "participant", "participant_id"]
                         if c in cols_lower), None)
    group_col = next((cols_lower[c] for c in
                       ["group", "group_number", "group_nr", "condition_group"]
                       if c in cols_lower), None)

    if subject_col is None or group_col is None:
        print(f"  [group] could not find subject/group columns in {path.name} "
              f"(columns present: {list(df.columns)}) — proceeding without group labels")
        return {}

    return {str(row[subject_col]): str(row[group_col]) for _, row in df.iterrows()}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Eye-Tracking Visual World Paradigm analysis pipeline. "
                     "By default, discovers every {subject}.tsv + {subject}.csv "
                     "pair in the Input/ folder next to this script (e.g. "
                     "subject-1.tsv/subject-1.csv, subject-2.tsv/subject-2.csv, ...) "
                     "and processes all of them, pooling results with a subject_nr "
                     "column. Pass --gaze-tsv/--trial-csv to instead process a single "
                     "explicit file pair. Results are written to the Output/ folder "
                     "next to this script."
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                    help=f"Folder to scan for {{subject}}.tsv + {{subject}}.csv pairs "
                         f"(default: {DEFAULT_INPUT_DIR}). Ignored if --gaze-tsv/--trial-csv given.")
    p.add_argument("--gaze-tsv", type=Path, default=None,
                    help="Path to a single GazePoint TSV (overrides --input-dir discovery; "
                         "must be paired with --trial-csv)")
    p.add_argument("--trial-csv", type=Path, default=None,
                    help="Path to a single OpenSesame trial CSV (overrides --input-dir discovery; "
                         "must be paired with --gaze-tsv)")
    p.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION_CSV,
                    help=f"Path to forced-alignment annotation CSV, shared across all subjects "
                         f"(default: {DEFAULT_ANNOTATION_CSV})")
    p.add_argument("--annotation-time-unit", choices=["auto", "seconds", "ms"], default="auto",
                    help="Unit of the 'start'/'end' columns in the annotation CSV. "
                         "'auto' (default) detects seconds vs. milliseconds automatically.")
    p.add_argument("--group-csv", type=Path, default=None,
                    help="Optional CSV mapping subjects to a group (e.g. participant_groupX.csv "
                         "/ overview.csv). Auto-detects a subject-id column (subject_id/subject/"
                         "subject_nr/participant/participant_id) and a group column (group/"
                         "group_number/group_nr/condition_group). Used to color the QC plots.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help=f"Directory to write results to (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--skip-plots", action="store_true",
                    help="Skip generating PNG visualizations (only write CSV outputs)")
    args = p.parse_args()

    if bool(args.gaze_tsv) != bool(args.trial_csv):
        p.error("--gaze-tsv and --trial-csv must be given together (or neither, to use --input-dir discovery)")
    return args


# ──────────────────────────────────────────────────────────────
# PER-SUBJECT PROCESSING
# ──────────────────────────────────────────────────────────────

def process_subject(gaze_tsv, trial_csv, words_df, ann_lookup, log_path=None):
    """
    Run the full per-trial pipeline for one subject's gaze TSV + trial CSV,
    against the shared (stimulus-level) words_df / ann_lookup.

    If log_path is given, also parses that subject's pygaze calibration
    report for the QC plots.

    Returns (props_df, crit_df, growth_df, raw_gaze_df, word_prob_df,
             calib_row, valid_row) for this subject alone. The DataFrames
    may be empty if nothing could be processed (but never None); calib_row
    is None if no/unparsable log file was given; valid_row is always a dict.
    """
    print("  Loading gaze/trial data...")
    gaze_df  = load_gaze(gaze_tsv)
    trial_df = load_trials(trial_csv)

    subject_nr = int(trial_df["subject_nr"].iloc[0])
    print(f"  Subject {subject_nr}  |  {len(trial_df)} trials in CSV")

    print("  Segmenting gaze stream into trials...")
    segments = segment_trials(gaze_df)
    print(f"    {len(segments)} trial segments found in TSV")

    # ── QC: calibration accuracy + valid-data percentage ────────
    calib_row = None
    if log_path is not None:
        calib = parse_calibration_report(log_path)
        if calib is not None:
            calib_row = {"subject_nr": subject_nr, **calib}
        else:
            print(f"    [qc] could not parse calibration accuracy from {log_path}")

    valid = compute_valid_data_percentage(gaze_df, segments=segments)
    valid_row = {"subject_nr": subject_nr, "pct_valid_overall": round(valid["pct_valid_overall"], 2)}
    print(f"    [qc] valid gaze samples: {valid_row['pct_valid_overall']:.1f}% overall")

    if len(segments) != len(trial_df):
        print(
            f"    WARNING: segment count ({len(segments)}) ≠ trial CSV rows "
            f"({len(trial_df)}). Using first {min(len(segments), len(trial_df))}."
        )

    n_trials = min(len(segments), len(trial_df))

    all_metrics, all_crit_fix, all_growth, all_raw_gaze, all_word_probs = [], [], [], [], []

    for i in range(n_trials):
        seg   = segments[i]
        t_row = trial_df.iloc[i]

        # Parse sentence id from audio filename, e.g. "2_r.wav" → audio_id=2
        audio_file = str(t_row["audio_file"])
        try:
            audio_id = int(audio_file.split("_")[0])
        except ValueError:
            print(f"    [skip] Trial {i}: cannot parse audio_file '{audio_file}'")
            continue

        condition        = str(t_row["condition"])
        sentence         = str(t_row["sentence"])
        target_position  = int(t_row["target_position"])

        ann_key = (audio_id, condition)
        if ann_key not in ann_lookup:
            print(f"    [skip] Trial {i}: no annotation for {ann_key}")
            continue

        verb_offset_s, target_onset_s, target_word = ann_lookup[ann_key]
        window_ms = (target_onset_s - verb_offset_s) * 1000

        # Fixations for this trial (position derived from BPOGX/BPOGY)
        fix_df = collapse_to_fixations(seg["gaze"])

        # Critical window (fixation-level)
        crit = extract_critical_window(
            fix_df, seg["audio_onset_s"], verb_offset_s, target_onset_s
        )

        # Critical window (raw per-sample, for gaze-point plots)
        raw = extract_critical_window_raw(
            seg["gaze"], seg["audio_onset_s"], verb_offset_s, target_onset_s
        )
        if not raw.empty:
            raw = raw.copy()
            raw["trial"]      = i
            raw["condition"]  = condition
            raw["subject_nr"] = subject_nr
            all_raw_gaze.append(raw)

        if crit.empty:
            print(f"    [warn] Trial {i} ({condition}, '{target_word}'): "
                  f"no fixations in critical window ({window_ms:.0f} ms)")

        # Trial-level metrics (critical window only)
        metrics = compute_trial_metrics(
            crit, i, subject_nr,
            condition, sentence, target_word, target_position, window_ms
        )
        metrics["sentence_id"] = audio_id
        metrics["audio_id"]    = audio_id
        all_metrics.append(metrics)

        if not crit.empty:
            crit = crit.copy()
            crit["trial"]            = i
            crit["condition"]        = condition
            crit["subject_nr"]       = subject_nr
            crit["target_word"]      = target_word
            crit["target_position"]  = target_position
            all_crit_fix.append(crit)

            gc = compute_growth_curve(
                crit, seg["audio_onset_s"], verb_offset_s, target_onset_s,
                bin_ms=GROWTH_CURVE_BIN_MS
            )
            if not gc.empty:
                gc["trial"]      = i
                gc["condition"]  = condition
                gc["target_pos"] = target_position
                gc["subject_nr"] = subject_nr
                all_growth.append(gc)

        # Per-word probability distributions: every word in the sentence,
        # raw-gaze-averaged — the human analogue of an LLM's per-word
        # predicted probability over the 4 candidate objects.
        word_rows = words_df[(words_df["id"] == audio_id) & (words_df["condition"] == condition)]
        if not word_rows.empty:
            wp = compute_word_probabilities(
                seg["gaze"], seg["audio_onset_s"], word_rows,
                i, subject_nr, condition, audio_id, target_position
            )
            if not wp.empty:
                all_word_probs.append(wp)

    props_df     = pd.DataFrame(all_metrics)
    crit_df      = pd.concat(all_crit_fix,  ignore_index=True) if all_crit_fix  else pd.DataFrame()
    growth_df    = pd.concat(all_growth,    ignore_index=True) if all_growth    else pd.DataFrame()
    raw_gaze_df  = pd.concat(all_raw_gaze,  ignore_index=True) if all_raw_gaze  else pd.DataFrame()
    word_prob_df = pd.concat(all_word_probs,ignore_index=True) if all_word_probs else pd.DataFrame()

    return props_df, crit_df, growth_df, raw_gaze_df, word_prob_df, calib_row, valid_row


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    annotation_csv = args.annotation_csv
    out            = args.output_dir

    if not annotation_csv.exists():
        sys.exit(
            f"ERROR: annotation CSV not found at {annotation_csv}\n"
            f"Place it in {DEFAULT_INPUT_DIR} (default name annotation_audiov1.csv), "
            f"or pass --annotation-csv explicitly."
        )

    out.mkdir(parents=True, exist_ok=True)

    # ── Load shared (stimulus-level) annotation once ───────────
    print("Loading annotation (shared across all subjects)...")
    words_df = load_annotation_words(annotation_csv, time_unit=args.annotation_time_unit)

    verbs = (words_df[words_df["WordRole"] == "ROOT"]
             [["id", "condition", "end_s"]]
             .rename(columns={"end_s": "verb_offset_s"}))
    targets = (words_df[words_df["WordRole"] == "dobj"]
               [["id", "condition", "start_s", "word"]]
               .rename(columns={"start_s": "target_onset_s", "word": "target_word"}))
    ann_df = verbs.merge(targets, on=["id", "condition"])

    ann_lookup = {}
    for _, row in ann_df.iterrows():
        ann_lookup[(int(row["id"]), str(row["condition"]))] = (
            float(row["verb_offset_s"]),
            float(row["target_onset_s"]),
            str(row["target_word"]),
        )

    # ── Determine which subject(s) to process ───────────────────
    if args.gaze_tsv and args.trial_csv:
        for f, label in [(args.gaze_tsv, "gaze TSV"), (args.trial_csv, "trial CSV")]:
            if not f.exists():
                sys.exit(f"ERROR: {label} not found at {f}")
        log_path = args.trial_csv.parent / f"{args.trial_csv.stem}_log.txt"
        subjects = [{"subject_id": args.trial_csv.stem, "tsv": args.gaze_tsv, "csv": args.trial_csv,
                     "log": log_path if log_path.exists() else None}]
    else:
        subjects = discover_subjects(args.input_dir)
        if not subjects:
            sys.exit(
                f"ERROR: no {{subject}}.tsv + {{subject}}.csv file pairs found in {args.input_dir}\n"
                f"Expected e.g. subject-1.tsv + subject-1.csv (per participant), or j.tsv + j.csv. "
                f"Pass --gaze-tsv/--trial-csv for a single explicit pair, or check --input-dir."
            )

    print(f"\nFound {len(subjects)} subject(s): {', '.join(s['subject_id'] for s in subjects)}")

    # ── Process each subject ────────────────────────────────────
    all_props, all_crit, all_growth, all_raw, all_wordprob = [], [], [], [], []
    all_calib, all_valid = [], []
    for subj in subjects:
        print(f"\n{'='*70}\nSubject: {subj['subject_id']}\n{'='*70}")
        try:
            p_df, c_df, g_df, r_df, w_df, calib_row, valid_row = process_subject(
                subj["tsv"], subj["csv"], words_df, ann_lookup, log_path=subj.get("log")
            )
        except Exception as e:
            print(f"  [subject FAILED] {subj['subject_id']}: {e}")
            continue
        if not p_df.empty: all_props.append(p_df)
        if not c_df.empty: all_crit.append(c_df)
        if not g_df.empty: all_growth.append(g_df)
        if not r_df.empty: all_raw.append(r_df)
        if not w_df.empty: all_wordprob.append(w_df)
        if calib_row is not None:
            calib_row["subject_id"] = subj["subject_id"]
            all_calib.append(calib_row)
        valid_row["subject_id"] = subj["subject_id"]
        all_valid.append(valid_row)

    if not all_props:
        sys.exit("ERROR: no trials were successfully processed for any subject.")

    # ── Pool across subjects ─────────────────────────────────────
    props_df     = pd.concat(all_props,    ignore_index=True)
    crit_df      = pd.concat(all_crit,     ignore_index=True) if all_crit     else pd.DataFrame()
    growth_df    = pd.concat(all_growth,   ignore_index=True) if all_growth   else pd.DataFrame()
    raw_gaze_df  = pd.concat(all_raw,      ignore_index=True) if all_raw      else pd.DataFrame()
    word_prob_df = pd.concat(all_wordprob, ignore_index=True) if all_wordprob else pd.DataFrame()
    calib_df     = pd.DataFrame(all_calib) if all_calib else pd.DataFrame()
    valid_df     = pd.DataFrame(all_valid) if all_valid else pd.DataFrame()

    props_df = props_df.sort_values(["subject_nr", "sentence_id"]).reset_index(drop=True)

    # ── Attach group labels, if a group file was given ───────────
    group_map = load_group_assignments(args.group_csv)
    if group_map:
        for df in (calib_df, valid_df):
            if not df.empty:
                df["group"] = df["subject_id"].map(group_map).fillna("unassigned")

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "="*105)
    print(f"GAZE DISTRIBUTION PER TRIAL  ({props_df['subject_nr'].nunique()} subject(s), "
          f"{len(props_df)} trials total; critical window: verb offset → target onset)")
    print("prop_pos1–4 sum to 1.0 per trial  |  pos1=rightmost … pos4=leftmost  |  prop_subject informational")
    print("="*105)
    display_cols = ["subject_nr", "sentence_id", "condition", "sentence", "target_word", "target_position",
                    "prop_pos1", "prop_pos2", "prop_pos3", "prop_pos4",
                    "prop_subject", "no_object_gaze", "total_window_ms"]
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 45)
    print(props_df[display_cols].to_string(index=False))

    print("\n" + "="*90)
    print("CONDITION SUMMARY  (mean across all subjects/sentences — no object gaze trials included as 0)")
    print("="*90)
    summary = (props_df
               .groupby("condition")[["prop_pos1","prop_pos2","prop_pos3","prop_pos4",
                                      "prop_subject","prop_target","target_advantage",
                                      "fixation_entropy"]]
               .mean().round(3))
    print(summary.to_string())

    if not calib_df.empty or not valid_df.empty:
        print("\n" + "="*90)
        print("DATA-QUALITY SUMMARY  (per participant)")
        print("="*90)
        if not calib_df.empty:
            print("\nCalibration accuracy (degrees):")
            print(calib_df.to_string(index=False))
        if not valid_df.empty:
            print("\nValid gaze samples (%):")
            print(valid_df.to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────
    props_path      = out / "fixation_proportions.csv"
    crit_path       = out / "fixation_critical_window.csv"
    growth_path     = out / "growth_curves.csv"
    raw_gaze_path   = out / "raw_gaze_critical_window.csv"
    word_prob_path  = out / "word_probabilities.csv"
    calib_path      = out / "calibration_accuracy.csv"
    valid_path      = out / "valid_data_percentage.csv"

    props_df.to_csv(props_path, index=False)
    print(f"\nSaved: {props_path}  ({len(props_df)} trial rows)")

    if not crit_df.empty:
        crit_df.to_csv(crit_path, index=False)
        print(f"Saved: {crit_path}  ({len(crit_df)} fixation events)")

    if not growth_df.empty:
        growth_df.to_csv(growth_path, index=False)
        print(f"Saved: {growth_path}  ({len(growth_df)} time-bin rows)")

    if not raw_gaze_df.empty:
        raw_gaze_df.to_csv(raw_gaze_path, index=False)
        print(f"Saved: {raw_gaze_path}  ({len(raw_gaze_df)} raw gaze samples)")

    if not word_prob_df.empty:
        word_prob_df.to_csv(word_prob_path, index=False)
        print(f"Saved: {word_prob_path}  ({len(word_prob_df)} word-level rows)")

    if not calib_df.empty:
        calib_df.to_csv(calib_path, index=False)
        print(f"Saved: {calib_path}  ({len(calib_df)} subject(s))")

    if not valid_df.empty:
        valid_df.to_csv(valid_path, index=False)
        print(f"Saved: {valid_path}  ({len(valid_df)} subject(s))")

    # ── Plots ─────────────────────────────────────────────────
    if not args.skip_plots:
        generate_visualizations(props_df, crit_df, growth_df, raw_gaze_df, word_prob_df,
                                 calib_df, valid_df, out / "plots")

    return props_df, crit_df, growth_df, raw_gaze_df, word_prob_df, calib_df, valid_df


if __name__ == "__main__":
    props_df, crit_df, growth_df, raw_gaze_df, word_prob_df, calib_df, valid_df = main()
