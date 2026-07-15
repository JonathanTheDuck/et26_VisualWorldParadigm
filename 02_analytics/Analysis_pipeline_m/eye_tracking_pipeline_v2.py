"""
Eye-Tracking Visual World Paradigm — Analysis Pipeline (v2)
=============================================================
Inputs
------
  j.tsv                     GazePoint raw sample stream (one participant)
  j.csv                     OpenSesame trial log (one participant)
  annotation_audiov1.csv    Forced-alignment word timing per sentence
  stimulus_list.csv         (NEW) sentence_id -> pos1_object..pos4_object
                             names, one row per image *version*
                             (4 versions per image, one per target slot)

Outputs (saved to OUTPUT_DIR)
------
  gaze_proportions.csv        Per-trial 4-way gaze-COUNT proportions + metrics
  gaze_critical_window.csv    Every raw gaze sample inside the critical window
  growth_curves.csv           Gaze-count proportions in 50 ms bins
  object_proportions.csv      (NEW) per-trial proportions re-labelled by
                               OBJECT NAME instead of slot position
  object_aggregate.csv        (NEW) proportions aggregated over the 4
                               position-versions of each image, one row per
                               object name (this is what you compare to the
                               LLM's probability distribution)

======================================================================
WHAT CHANGED FROM v1  (see accompanying CHANGES.md for the long version)
======================================================================
1. RAW GAZE SAMPLES, NOT FIXATIONS
   GazePoint's own fixation detector (FPOGID) is no longer used to decide
   what counts as "looking at" an AOI. Instead every valid raw sample
   (FPOGV == 1) in the critical window is classified into an AOI. This
   avoids the "0 fixations -> 0 probability" problem: a participant can
   glance briefly at an object without GazePoint registering a full
   fixation event there, but the raw samples still land inside the AOI
   and are now counted.

2. OBJECT NAMES, NOT SLOT POSITIONS
   The 4-way output is still computed per position (pos1-pos4) internally
   (since AOI geometry is defined by position), but a new step maps
   pos1-pos4 -> real object names via stimulus_list.csv, then aggregates
   across the 4 position-counterbalanced versions of each image so you
   get ONE distribution per object identity, comparable to the LLM output.

3. AOI WINDOW = VERB ONSET -> TARGET ONSET
   Previously the window was [verb_offset, target_onset]. It is now
   [verb_onset, target_onset], i.e. it starts as soon as the verb begins
   being spoken, not when it finishes.

4. COUNT-BASED PROBABILITY, NOT DURATION-BASED
   Proportions are now computed as (# raw gaze samples landing on object)
   / (total # raw gaze samples landing on any of the 4 objects) within the
   critical window - NOT time-weighted by fixation/sample duration. Each
   raw sample is one "gaze" event, so this is effectively a per-sample
   (near-saccade-rate) count rather than a dwell-time measure. NOTE:
   GazePoint's raw stream does not ship an explicit saccade-id column, so
   "count of saccades" is operationalised here as "count of raw samples in
   the AOI" (one row per sample = one gaze position for that time-step).
   If your GazePoint export DOES contain a saccade-id column, tell me and
   I'll switch the counting unit to collapsed saccades instead of raw rows.

Design notes carried over from v1
----------------------------------
* All timing stays on the GazePoint clock (seconds) throughout.
* AOI bounding boxes were measured from 1_pos1_sub.png (2400x1400 px) and
  converted to the GazePoint normalized (0-1) coordinate space assuming
  OpenSesame "scale to fit, preserve aspect ratio" display mode on 1920x1080.

  Slot layout (consistent across all stimulus images):
    pos1 = top-right   pos2 = top-left
    pos3 = bottom-left pos4 = bottom-right
    (subject always in center - not an AOI of interest)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
GAZE_TSV           = r"01_experiment/recordings/validParticipants/subject-3.tsv"
TRIAL_CSV          = r"01_experiment/recordings/validParticipants/subject-3.csv"
ANNOTATION_CSV      = r"01_experiment/stimuli/annotation_audiov2.csv"
#STIMULUS_LIST_CSV   = r"C:\Users\mmudali\Downloads\stimulus_list.csv"  # NEW - object names per position, per image version
OUTPUT_DIR          = r""#empty for now

GROWTH_CURVE_BIN_MS = 50   # time-bin width for growth curves

# ──────────────────────────────────────────────────────────────
# AOI GEOMETRY  (unchanged from v1)
# ──────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080
IMG_W,    IMG_H    = 2400, 1400

_SCALE    = SCREEN_H / IMG_H
_DISP_W   = IMG_W * _SCALE
_X_OFFSET = (SCREEN_W - _DISP_W) / 2


def _to_norm(x_img, y_img):
    """Image pixel -> GazePoint normalized (FPOGX/FPOGY in 0-1 range)."""
    x_scr = x_img * _SCALE + _X_OFFSET
    y_scr = y_img * _SCALE
    return x_scr / SCREEN_W, y_scr / SCREEN_H


_AOI_IMG_PX = {
    "pos1":    (1836, 468, 2140, 702),
    "pos2":    (1384, 945, 1681, 1134),
    "pos3":    (718,  903, 1009, 1144),
    "pos4":    (301,  455, 530,  706),
    "subject": (1111, 111, 1289, 688),
}

AOI_NORM = {}
for _name, (_x1, _y1, _x2, _y2) in _AOI_IMG_PX.items():
    _nx1, _ny1 = _to_norm(_x1, _y1)
    _nx2, _ny2 = _to_norm(_x2, _y2)
    AOI_NORM[_name] = (_nx1, _ny1, _nx2, _ny2)


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_gaze(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["USER"] = df["USER"].fillna("").astype(str).str.strip()
    return df


def load_trials(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[
        df["audio_file"].notna() &
        (df["audio_file"].astype(str) != "undefined")
    ].copy()
    df = df.sort_values("count_trial_loop").reset_index(drop=True)
    return df


def load_annotation(path):
    """
    Load forced-alignment CSV.
    Returns one row per (sentence_id, condition) with columns:
        id, condition, verb_onset_s, verb_offset_s, target_onset_s, target_word

    CHANGED (v2): now also pulls the verb's *start* time (verb_onset_s),
    since the critical window is now [verb_onset, target_onset] instead
    of [verb_offset, target_onset]. verb_offset_s is still returned for
    reference/QA but no longer used to define the window.
    """
    df = pd.read_csv(path)
    verbs = (df[df["WordRole"] == "ROOT"]
             [["id", "SentenceRole", "start", "end"]]
             .rename(columns={"start": "verb_onset_s",
                               "end": "verb_offset_s",
                               "SentenceRole": "condition"}))
    targets = (df[df["WordRole"] == "dobj"]
               [["id", "SentenceRole", "start", "word"]]
               .rename(columns={"start": "target_onset_s",
                                 "word":  "target_word",
                                 "SentenceRole": "condition"}))
    return verbs.merge(targets, on=["id", "condition"])


# ──────────────────────────────────────────────────────────────
# TRIAL SEGMENTATION (GazePoint clock) — unchanged
# ──────────────────────────────────────────────────────────────

def segment_trials(gaze_df):
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
            f"START_TRIAL count ({len(starts)}) != AUDIO_FILE_ONSET_LOG count "
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
# AOI CLASSIFICATION — unchanged
# ──────────────────────────────────────────────────────────────

def classify_aoi(fpogx, fpogy):
    for name in ("pos1", "pos2", "pos3", "pos4", "subject"):
        x1, y1, x2, y2 = AOI_NORM[name]
        if x1 <= fpogx <= x2 and y1 <= fpogy <= y2:
            return name
    return "elsewhere"


# ──────────────────────────────────────────────────────────────
# RAW GAZE SAMPLE EXTRACTION IN CRITICAL WINDOW  (NEW — replaces
# collapse_to_fixations + extract_critical_window)
# ──────────────────────────────────────────────────────────────

def extract_critical_window_gaze(gaze_seg, audio_onset_s,
                                  verb_onset_s, target_onset_s):
    """
    Keep every VALID raw gaze sample (FPOGV == 1) whose TIME falls inside
        [audio_onset_s + verb_onset_s,  audio_onset_s + target_onset_s]

    This replaces the old fixation-based extraction. Each row of the
    returned DataFrame is one raw sample = one discrete "gaze" event.
    Adds an 'aoi' column via classify_aoi().

    Returns a DataFrame (may be empty if no samples fall in the window).
    """
    win_start = audio_onset_s + verb_onset_s
    win_end   = audio_onset_s + target_onset_s

    if win_end <= win_start:
        return pd.DataFrame()

    df = gaze_seg[gaze_seg["FPOGV"] == 1].copy()
    df = df[(df["TIME"] >= win_start) & (df["TIME"] <= win_end)].copy()
    if df.empty:
        return df

    df["aoi"] = [classify_aoi(x, y) for x, y in zip(df["FPOGX"], df["FPOGY"])]
    return df


# ──────────────────────────────────────────────────────────────
# METRICS PER TRIAL — NOW COUNT-BASED, NOT DURATION-BASED
# ──────────────────────────────────────────────────────────────

def _entropy(probs):
    val = -sum(p * np.log2(p) for p in probs if p > 0)
    return float(abs(val))


def compute_trial_metrics(crit_gaze, trial_idx, subject_nr,
                           condition, sentence, target_word, target_position,
                           window_ms):
    """
    From critical-window RAW GAZE SAMPLES for one trial compute:

    4-way gaze distribution (prop_pos1-4)
      Normalised over the NUMBER OF RAW GAZE SAMPLES landing on the 4
      object slots only (count-based, not duration-weighted). These four
      values sum to 1.0 and are directly comparable to the LLM's
      renormalised 4-way surprisal distribution.
      Trials where the participant's gaze never lands on any object slot
      during the window get prop_pos* = 0.0 and are flagged with
      no_object_gaze=True.

    prop_subject (informational)
      Proportion of samples on the subject figure, normalised over all 5
      named AOIs' sample counts. Not included in the 4-way sum.

    target_advantage : P(target) - mean(P(distractor_1..3))
    gaze_entropy      : Shannon entropy of the 4-way distribution (bits)
    """
    obj_keys = ["pos1", "pos2", "pos3", "pos4"]
    raw_obj  = {k: 0 for k in obj_keys}
    raw_subj = 0

    if not crit_gaze.empty:
        counts = crit_gaze["aoi"].value_counts()
        for k in obj_keys:
            raw_obj[k] = int(counts.get(k, 0))
        raw_subj = int(counts.get("subject", 0))

    obj_total = sum(raw_obj.values())
    all_total = obj_total + raw_subj

    # 4-way distribution: normalised over object-slot SAMPLE COUNTS only
    props = {k: (v / obj_total if obj_total > 0 else 0.0)
             for k, v in raw_obj.items()}

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
        "gaze_entropy":     round(_entropy(list(props.values())), 6),
        "no_object_gaze":   obj_total == 0,
        "total_window_ms":  round(window_ms, 2),
        # raw counts, kept for transparency / debugging
        "n_gaze_pos1":      raw_obj["pos1"],
        "n_gaze_pos2":      raw_obj["pos2"],
        "n_gaze_pos3":      raw_obj["pos3"],
        "n_gaze_pos4":      raw_obj["pos4"],
        "n_gaze_subject":   raw_subj,
        "n_gaze_object_total": obj_total,
    }


# ──────────────────────────────────────────────────────────────
# GROWTH CURVES — NOW COUNT-BASED (proportion of samples per bin)
# ──────────────────────────────────────────────────────────────

def compute_growth_curve(crit_gaze, audio_onset_s, verb_onset_s,
                          target_onset_s, bin_ms=50.0):
    """
    Compute, per time bin, the proportion of that bin's gaze SAMPLES
    landing on each AOI (count-based; replaces the old fixation-overlap
    duration calculation). Bins are aligned to verb onset (bin 0 =
    [verb_onset, verb_onset+50ms]).
    Returns DataFrame with columns: bin_start_ms, aoi, prop_fixating,
    n_samples_in_bin
    """
    win_start = audio_onset_s + verb_onset_s
    win_end   = audio_onset_s + target_onset_s
    win_len   = (win_end - win_start) * 1000

    if win_len <= 0 or crit_gaze.empty:
        return pd.DataFrame(columns=["bin_start_ms", "aoi", "prop_fixating"])

    bins = np.arange(0, win_len, bin_ms)
    rows = []
    for b in bins:
        abs_b_start = win_start + b / 1000
        abs_b_end   = win_start + (b + bin_ms) / 1000
        if abs_b_end > win_end:
            abs_b_end = win_end
        if abs_b_end <= abs_b_start:
            continue

        bin_samples = crit_gaze[
            (crit_gaze["TIME"] >= abs_b_start) & (crit_gaze["TIME"] < abs_b_end)
        ]
        n_bin = len(bin_samples)

        for pos in list(range(1, 5)) + ["subject"]:
            aoi_label = f"pos{pos}" if pos != "subject" else "subject"
            n_aoi = int((bin_samples["aoi"] == aoi_label).sum())
            rows.append({
                "bin_start_ms":    b,
                "aoi":             aoi_label,
                "prop_fixating":   round(n_aoi / n_bin, 6) if n_bin > 0 else 0.0,
                "n_samples_in_bin": n_bin,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# OBJECT NAME MAPPING + AGGREGATION ACROSS POSITION-VERSIONS  (NEW)
# ──────────────────────────────────────────────────────────────

def add_object_names(props_df, stimulus_list_path):
    """
    Merge object names into props_df from the stimulus list CSV and
    relabel the 4-way position distribution by OBJECT NAME.

    stimulus_list.csv must have columns:
        sentence_id, pos1_object, pos2_object, pos3_object, pos4_object

    IMPORTANT: sentence_id here identifies one *version* (one specific
    position-arrangement) of an image, matching props_df["sentence_id"]
    exactly as produced by main(). Each of the 4 versions of the same
    base image should share a common "base_image" identifier — provide
    that column too if you want automatic aggregation via
    aggregate_by_object() below. If a "base_image" column is not present,
    it is inferred by stripping a trailing "_posN" style suffix from
    sentence_id, if present; otherwise you'll need to supply it.

    Returns a long-format DataFrame with one row per (trial, object):
        subject_nr, trial, sentence_id, condition, target_word,
        object_name, prop, is_target
    """
    stim = pd.read_csv(stimulus_list_path)
    merged = props_df.merge(stim, left_on="sentence_id", right_on="sentence_id",
                             how="left")

    missing = merged["pos1_object"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} trial rows had no matching stimulus_list "
              f"entry (sentence_id not found) — object names will be NaN.")

    long_rows = []
    for _, row in merged.iterrows():
        for pos in range(1, 5):
            obj_name = row.get(f"pos{pos}_object", np.nan)
            long_rows.append({
                "subject_nr":   row["subject_nr"],
                "trial":        row["trial"],
                "sentence_id":  row["sentence_id"],
                "base_image":   row.get("base_image", _infer_base_image(row["sentence_id"])),
                "condition":    row["condition"],
                "target_word":  row["target_word"],
                "object_name":  obj_name,
                "prop":         row[f"prop_pos{pos}"],
                "n_gaze":       row[f"n_gaze_pos{pos}"],
                "is_target":    (pos == row["target_position"]),
            })
    return pd.DataFrame(long_rows)


def _infer_base_image(sentence_id):
    """
    Best-effort fallback for grouping the 4 position-counterbalanced
    versions of the same image together when no explicit 'base_image'
    column is supplied in stimulus_list.csv.
    Strips a trailing '_posN' / '-posN' suffix from a string sentence_id.
    If sentence_id is a bare integer (as in this pipeline, matched from
    the audio filename), this cannot infer grouping — supply a
    'base_image' column in stimulus_list.csv instead.
    """
    s = str(sentence_id)
    import re
    m = re.match(r"^(.*?)[_\-]?pos\d$", s, flags=re.IGNORECASE)
    return m.group(1) if m else s


def aggregate_by_object(object_props_df):
    """
    Aggregate the long-format per-trial, per-object proportions across the
    4 position-versions of each base image, giving ONE probability
    distribution over object identities per base image — directly
    comparable to the LLM's per-object probability distribution.

    Aggregation method: for each (base_image, object_name), sum the raw
    gaze-sample counts (n_gaze) across all trials/versions/participants
    that included that object, then renormalise within each base_image so
    the object-level proportions sum to 1.0. This is the count-consistent
    way to combine counts-based proportions (summing raw counts before
    renormalising avoids double-weighting trials with few total samples).

    Returns DataFrame with columns:
        base_image, object_name, n_gaze_total, prop_object, n_trials,
        n_trials_as_target
    """
    grouped = (object_props_df
               .groupby(["base_image", "object_name"])
               .agg(n_gaze_total=("n_gaze", "sum"),
                    n_trials=("trial", "nunique"),
                    n_trials_as_target=("is_target", "sum"))
               .reset_index())

    grouped["prop_object"] = (
        grouped.groupby("base_image")["n_gaze_total"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0.0)
    )

    return grouped.sort_values(["base_image", "prop_object"], ascending=[True, False])


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    print("Loading data...")
    gaze_df  = load_gaze(GAZE_TSV)
    trial_df = load_trials(TRIAL_CSV)
    ann_df   = load_annotation(ANNOTATION_CSV)

    subject_nr = int(trial_df["subject_nr"].iloc[0])
    print(f"  Subject {subject_nr}  |  {len(trial_df)} trials in CSV")

    # ── Segment gaze stream ───────────────────────────────────
    print("Segmenting gaze stream into trials...")
    segments = segment_trials(gaze_df)
    print(f"  {len(segments)} trial segments found in TSV")

    if len(segments) != len(trial_df):
        print(
            f"  WARNING: segment count ({len(segments)}) != trial CSV rows "
            f"({len(trial_df)}). Using first {min(len(segments), len(trial_df))}."
        )

    n_trials = min(len(segments), len(trial_df))

    # ── Build annotation lookup ───────────────────────────────
    # Key: (sentence_id_int, condition_str) ->
    #      (verb_onset_s, target_onset_s, target_word)
    ann_lookup = {}
    for _, row in ann_df.iterrows():
        ann_lookup[(int(row["id"]), str(row["condition"]))] = (
            float(row["verb_onset_s"]),
            float(row["target_onset_s"]),
            str(row["target_word"]),
        )

    # ── Per-trial processing ──────────────────────────────────
    print("Processing trials...")
    all_metrics  = []
    all_crit_gaze = []
    all_growth   = []

    for i in range(n_trials):
        seg    = segments[i]
        t_row  = trial_df.iloc[i]

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

        verb_onset_s, target_onset_s, target_word = ann_lookup[ann_key]
        window_ms = (target_onset_s - verb_onset_s) * 1000

        # Raw gaze samples in the critical window (verb onset -> target onset)
        crit = extract_critical_window_gaze(
            seg["gaze"], seg["audio_onset_s"], verb_onset_s, target_onset_s
        )

        if crit.empty:
            print(f"  [warn] Trial {i} ({condition}, '{target_word}'): "
                  f"no valid gaze samples in critical window ({window_ms:.0f} ms)")

        # Metrics
        metrics = compute_trial_metrics(
            crit, i, subject_nr,
            condition, sentence, target_word, target_position, window_ms
        )
        metrics["sentence_id"] = audio_id
        metrics["audio_id"]    = audio_id
        all_metrics.append(metrics)

        if not crit.empty:
            crit = crit.copy()
            crit["trial"]        = i
            crit["condition"]    = condition
            crit["subject_nr"]   = subject_nr
            crit["target_word"]  = target_word
            crit["target_position"] = target_position
            all_crit_gaze.append(crit)

            gc = compute_growth_curve(
                crit, seg["audio_onset_s"], verb_onset_s, target_onset_s,
                bin_ms=GROWTH_CURVE_BIN_MS
            )
            if not gc.empty:
                gc["trial"]      = i
                gc["condition"]  = condition
                gc["target_pos"] = target_position
                all_growth.append(gc)

    # ── Assemble DataFrames ───────────────────────────────────
    props_df  = pd.DataFrame(all_metrics)
    crit_df   = (pd.concat(all_crit_gaze, ignore_index=True)
                 if all_crit_gaze else pd.DataFrame())
    growth_df = (pd.concat(all_growth, ignore_index=True)
                 if all_growth else pd.DataFrame())

    props_df = props_df.sort_values("sentence_id").reset_index(drop=True)

    print("\n" + "="*105)
    print("GAZE DISTRIBUTION PER SENTENCE  (critical window: verb ONSET -> target onset, count-based)")
    print("prop_pos1-4 sum to 1.0 per sentence  |  pos1=rightmost ... pos4=leftmost  |  prop_subject informational")
    print("="*105)
    display_cols = ["sentence_id", "condition", "target_word", "target_position",
                    "prop_pos1", "prop_pos2", "prop_pos3", "prop_pos4",
                    "prop_subject", "no_object_gaze", "total_window_ms",
                    "n_gaze_object_total"]
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 45)
    print(props_df[display_cols].to_string(index=False))

    print("\n" + "="*90)
    print("CONDITION SUMMARY  (mean across sentences)")
    print("="*90)
    summary = (props_df
               .groupby("condition")[["prop_pos1","prop_pos2","prop_pos3","prop_pos4",
                                      "prop_subject","prop_target","target_advantage",
                                      "gaze_entropy"]]
               .mean().round(3))
    print(summary.to_string())

    # ── Save core outputs ─────────────────────────────────────
    props_path  = out / "gaze_proportions.csv"
    crit_path   = out / "gaze_critical_window.csv"
    growth_path = out / "growth_curves.csv"

    props_df.to_csv(props_path, index=False)
    print(f"\nSaved: {props_path}  ({len(props_df)} trial rows)")

    if not crit_df.empty:
        crit_df.to_csv(crit_path, index=False)
        print(f"Saved: {crit_path}  ({len(crit_df)} raw gaze-sample events)")

    if not growth_df.empty:
        growth_df.to_csv(growth_path, index=False)
        print(f"Saved: {growth_path}  ({len(growth_df)} time-bin rows)")

    # ── Object-name mapping + aggregation (NEW) ───────────────
    object_props_df = None
    object_agg_df   = None
    #stim_path = Path(STIMULUS_LIST_CSV)
    if stim_path.exists():
        print("\nMapping positions to object names via stimulus list...")
        #object_props_df = add_object_names(props_df, STIMULUS_LIST_CSV)
        obj_props_path = out / "object_proportions.csv"
        object_props_df.to_csv(obj_props_path, index=False)
        print(f"Saved: {obj_props_path}  ({len(object_props_df)} rows)")

        if object_props_df["base_image"].isna().all() or \
           (object_props_df["base_image"] == object_props_df["sentence_id"].astype(str)).all():
            print("  NOTE: no 'base_image' column found in stimulus_list.csv and "
                  "sentence_id does not look like it encodes a base image + position "
                  "suffix. Add a 'base_image' column to stimulus_list.csv (same value "
                  "for all 4 position-versions of a given image) so the 4 versions can "
                  "be aggregated together. Skipping aggregation for now.")
        else:
            object_agg_df = aggregate_by_object(object_props_df)
            obj_agg_path = out / "object_aggregate.csv"
            object_agg_df.to_csv(obj_agg_path, index=False)
            print(f"Saved: {obj_agg_path}  ({len(object_agg_df)} object rows across "
                  f"{object_agg_df['base_image'].nunique()} base images)")

            print("\n" + "="*90)
            print("OBJECT-LEVEL AGGREGATE PROBABILITY DISTRIBUTION (per base image)")
            print("="*90)
            print(object_agg_df.to_string(index=False))
    else:
       # print(f"\nNOTE: stimulus_list.csv not found at {STIMULUS_LIST_CSV} — "
        #      f"skipping object-name mapping/aggregation step. Provide that file "
       #       f"(columns: sentence_id, base_image, pos1_object..pos4_object) and "
       #       f"re-run, or call add_object_names()/aggregate_by_object() manually.")

    return props_df, crit_df, growth_df, object_props_df, object_agg_df


if __name__ == "__main__":
    props_df, crit_df, growth_df, object_props_df, object_agg_df = main()
