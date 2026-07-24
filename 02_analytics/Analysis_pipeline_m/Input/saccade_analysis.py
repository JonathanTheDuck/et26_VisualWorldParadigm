"""
First-Saccade-After-Verb-Onset Analysis
=========================================
A genuinely different measure from the rest of this project's QC pipeline:
everything else in quality_control.py / gaze_trace_video.py classifies RAW
GAZE SAMPLES inside the critical window (no fixation/saccade detection at
all — confirmed explicitly in an earlier conversation). This script adds
the complementary, more traditional VWP measure: detect actual saccades
from the raw gaze stream, find the FIRST one that starts after the verb
begins, and ask where it lands.

Saccade detection
------------------
Velocity-threshold algorithm using the pygaze session's OWN reported
criteria (from j_log.txt): speed threshold = 35 deg/s, acceleration
threshold = 9500 deg/s**2. Unlike the log's own PIXEL versions of these
thresholds (which were computed assuming an unconfirmed screen resolution
— see project history), this script converts raw BPOGX/BPOGY displacement
to degrees of visual angle directly from the session's physical geometry
(distance-to-screen and screen size in cm, also from the log), so it does
not depend on which pixel resolution assumption turned out to be right.

  angle (deg) = degrees(atan2(screen_distance_cm * tan(...))) -- see
  _px_to_deg() for the exact formula (small-angle-safe, not an
  approximation).

A saccade is a run of >=2 consecutive VALID samples where instantaneous
velocity exceeds the speed threshold; it ends at the first sample where
velocity drops back below threshold. Runs across invalid (BPOGV==0) gaps
are never bridged.

"Verb onset" here means the ROOT word's START time (annotation column
'start' where WordRole=='ROOT') -- deliberately NOT verb_offset (the
'end' time used elsewhere in this project for the critical-window start).
This is a different reference point on purpose, per this analysis's remit.

Comparison to the existing critical-window measure
-----------------------------------------------------
For every trial with both a valid first-saccade landing AND a valid
critical-window prop_target (from the raw-sample method), this script
reports:
  - per-trial agreement: does the first saccade land on the target AOI,
    and is prop_target > chance (0.25), and do the two methods agree?
  - a correlation between "first saccade lands on target" (binary) and
    prop_target (continuous)
  - side-by-side condition summaries from both methods

Run: python saccade_analysis.py
"""

import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE
ANNOTATION_CSV = HERE / "annotation_audiov2.csv"
OUT_DIR = HERE / "saccade_output"
PLOTS_DIR = OUT_DIR / "plots"
CSV_DIR = OUT_DIR / "csv"
for d in (OUT_DIR, PLOTS_DIR, CSV_DIR):
    d.mkdir(exist_ok=True)

# Group/arrangement CSVs (participant_group1.csv ... participant_group8.csv)
# -- this is where the actual OBJECT NAMES per position live (the subject's
# own trial CSV only has position NUMBERS, never object names). Update this
# path to match your machine:
GROUP_CSV_DIR = Path(r"C:\Users\ramas\Documents\GitHub\et26_VisualWorldParadigm"
                      r"\et26_VisualWorldParadigm\01_experiment\stimuli"
                      r"\creatingDataStructure\participant_groups_8")

SUBJECT_GROUP_CSV_OVERRIDE = {
    # "subject-4": "participant_group5.csv",
}


def subject_id_to_group_number(subject_id):
    """group = (subject_number % 8) + 1 -- confirmed formula (e.g. subject-4 -> group 5)."""
    matches = re.findall(r"\d+", subject_id)
    if not matches:
        return None
    return (int(matches[-1]) % 8) + 1


def find_group_csv_for_subject(subject_id):
    if subject_id in SUBJECT_GROUP_CSV_OVERRIDE:
        p = GROUP_CSV_DIR / SUBJECT_GROUP_CSV_OVERRIDE[subject_id]
        return p if p.exists() else None
    group_num = subject_id_to_group_number(subject_id)
    if group_num is None:
        return None
    p = GROUP_CSV_DIR / f"participant_group{group_num}.csv"
    return p if p.exists() else None

# ── Saccade detection parameters (from the session's own pygaze log) ────
SPEED_THRESHOLD_DEG_S = 35.0
ACCEL_THRESHOLD_DEG_S2 = 9500.0
MIN_SACCADE_SAMPLES = 3          # consecutive above-threshold transitions required
MIN_SACCADE_AMPLITUDE_DEG = 1.0  # minimum total displacement -- rejects noise-driven
                                  # threshold crossings that don't add up to real movement
SMOOTHING_WINDOW = 5             # rolling median window (samples) applied to BPOGX/BPOGY
                                  # before differentiating -- this tracker's RMS noise
                                  # (21.99px X / 37.57px Y, from its own calibration report)
                                  # exceeds the 35 deg/s threshold on RAW adjacent samples
                                  # (57% of raw frame-to-frame transitions were spuriously
                                  # above threshold; verified empirically before choosing
                                  # this window+amplitude combination)

# ── Blink detection (no dedicated blink column in this TSV -- inferred from
#    BPOGV dropout duration). Checked empirically on real data: invalid-run
#    durations show a clear cluster at 100-400ms (the classic human blink
#    range), separate from brief single-sample noise (<60ms) and long
#    tracking-loss/off-screen periods (>1000ms). ──
BLINK_MIN_DUR_S = 0.06
BLINK_MAX_DUR_S = 0.50
BLINK_EDGE_BUFFER_SAMPLES = 3   # also exclude this many samples immediately
                                # before/after a detected blink -- eyelid
                                # closing/opening artifacts often corrupt the
                                # nominally "valid" samples flanking a blink

# ── AOI geometry (deterministic, from split_and_compose.py's own constants
#    + the confirmed OpenSesame scale=0.8 / 2560x1440 display transform) ──
IMG_W, IMG_H = 2400, 1400
SCALE = 0.8
SCREEN_W, SCREEN_H = 2560, 1440
_DISP_W, _DISP_H = IMG_W * SCALE, IMG_H * SCALE
_X_OFFSET = (SCREEN_W - _DISP_W) / 2
_Y_OFFSET = (SCREEN_H - _DISP_H) / 2
_SUBJECT_SIZE = (420 * 2, 300 * 2)
_SUBJECT_Y = 50 * 2
_OPTION_SIZE = (200 * 2, 200 * 2)
_SEMICIRCLE_GAP = 95 * 2
_SEMICIRCLE_ANGLES = [15, 65, 115, 165]


def _img_box_to_norm(x1, y1, x2, y2):
    sx1, sy1 = x1 * SCALE + _X_OFFSET, y1 * SCALE + _Y_OFFSET
    sx2, sy2 = x2 * SCALE + _X_OFFSET, y2 * SCALE + _Y_OFFSET
    return (sx1 / SCREEN_W, sy1 / SCREEN_H, sx2 / SCREEN_W, sy2 / SCREEN_H)


def compute_default_aoi_norm():
    center_x = IMG_W // 2
    center_y = _SUBJECT_Y + _SUBJECT_SIZE[1] // 2
    a = _SUBJECT_SIZE[0] / 2 + _SEMICIRCLE_GAP + _OPTION_SIZE[0] / 2
    b = _SUBJECT_SIZE[1] / 2 + _SEMICIRCLE_GAP + _OPTION_SIZE[1] / 2
    boxes = {}
    for k in range(1, 5):
        angle = math.radians(_SEMICIRCLE_ANGLES[k - 1])
        ox = center_x + a * math.cos(angle)
        oy = center_y + b * math.sin(angle)
        boxes[f"pos{k}"] = (ox - _OPTION_SIZE[0] / 2, oy - _OPTION_SIZE[1] / 2,
                             ox + _OPTION_SIZE[0] / 2, oy + _OPTION_SIZE[1] / 2)
    sx1 = (IMG_W - _SUBJECT_SIZE[0]) // 2
    boxes["subject"] = (sx1, _SUBJECT_Y, sx1 + _SUBJECT_SIZE[0], _SUBJECT_Y + _SUBJECT_SIZE[1])
    return {name: _img_box_to_norm(*box) for name, box in boxes.items()}


AOI_NORM = compute_default_aoi_norm()


def classify_aoi(x, y):
    for name, (x1, y1, x2, y2) in AOI_NORM.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "elsewhere"


# ──────────────────────────────────────────────────────────────
# GEOMETRY: normalized screen displacement -> degrees of visual angle
# ──────────────────────────────────────────────────────────────

def _load_session_geometry(log_text):
    """Pulls distance-to-screen and physical screen size (cm) straight out
    of the session's own pygaze log — no assumed/hardcoded values."""
    m_dist = re.search(r"distance between participant and display:\s*([\d.]+)\s*cm", log_text)
    m_size = re.search(r"display size in cm:\s*([\d.]+)x([\d.]+)", log_text)
    distance_cm = float(m_dist.group(1)) if m_dist else 57.0
    if m_size:
        screen_w_cm, screen_h_cm = float(m_size.group(1)), float(m_size.group(2))
    else:
        screen_w_cm, screen_h_cm = 33.8, 27.1
    return distance_cm, screen_w_cm, screen_h_cm


def _norm_dist_to_deg(dx_norm, dy_norm, distance_cm, screen_w_cm, screen_h_cm):
    """
    Convert a displacement in normalized screen units (BPOGX/BPOGY, 0-1) to
    degrees of visual angle, using the actual viewing geometry — exact
    formula (angle between two viewing rays from the eye), not a small-angle
    linear approximation, since some saccades in this data span a large
    fraction of the screen.
    """
    dx_cm = dx_norm * screen_w_cm
    dy_cm = dy_norm * screen_h_cm
    dist_cm = math.hypot(dx_cm, dy_cm)
    return math.degrees(2 * math.atan2(dist_cm / 2, distance_cm))


# ──────────────────────────────────────────────────────────────
# SACCADE DETECTION
# ──────────────────────────────────────────────────────────────

def detect_and_mask_blinks(gaze_df):
    """
    Infers blinks from BPOGV dropout runs lasting BLINK_MIN_DUR_S to
    BLINK_MAX_DUR_S (this TSV has no dedicated blink column). Returns
    (gaze_df_with_blink_mask, blink_events) where gaze_df has a new
    'IS_BLINK_ADJACENT' column (True for samples inside a blink OR within
    BLINK_EDGE_BUFFER_SAMPLES of one -- these are excluded from saccade
    detection), and blink_events is a list of {onset_time, offset_time,
    duration_s} for QC reporting.
    """
    v = gaze_df["BPOGV"].values
    t = gaze_df["TIME"].values
    n = len(v)
    is_blink_adjacent = np.zeros(n, dtype=bool)
    blink_events = []

    i = 0
    while i < n:
        if v[i] == 0:
            j = i
            while j < n and v[j] == 0:
                j += 1
            dur = t[min(j, n - 1)] - t[i]
            if BLINK_MIN_DUR_S <= dur <= BLINK_MAX_DUR_S:
                blink_events.append({"onset_time": t[i], "offset_time": t[min(j, n - 1)], "duration_s": dur})
                lo = max(0, i - BLINK_EDGE_BUFFER_SAMPLES)
                hi = min(n, j + BLINK_EDGE_BUFFER_SAMPLES)
                is_blink_adjacent[lo:hi] = True
            i = j
        else:
            i += 1

    gaze_df = gaze_df.copy()
    gaze_df["IS_BLINK_ADJACENT"] = is_blink_adjacent
    return gaze_df, blink_events


def detect_saccades(seg, distance_cm, screen_w_cm, screen_h_cm):
    """
    Detects saccades in one trial's raw gaze segment.

    Two noise-control steps were added after an empirical check on this
    dataset (see module docstring / project notes): a naive velocity
    threshold on raw adjacent samples flagged 200+ "saccades" per trial,
    with median latencies under 20ms -- physically impossible (real saccade
    latency has a floor around 150ms). Diagnosis: this tracker's own
    calibration report states RMS noise of 21.99px (X) / 37.57px (Y), and
    on a real chunk of this data, 57% of RAW adjacent-sample transitions
    already exceeded 35 deg/s from noise alone, before any real eye
    movement. Fix:
      1. Smooth BPOGX/BPOGY with a rolling median (SMOOTHING_WINDOW
         samples) before differentiating -- reduces but does not eliminate
         noise-driven spikes.
      2. Require a MINIMUM AMPLITUDE (total angular displacement across the
         candidate saccade), not just sustained velocity -- real saccades
         cover real distance; noise-driven threshold crossings mostly
         don't accumulate net displacement even when locally fast.

    Only BPOGV==1 samples are used; velocity/smoothing never bridges an
    invalid (dropout) gap -- each valid run is processed independently.
    """
    if "IS_BLINK_ADJACENT" in seg.columns:
        not_blink = ~seg["IS_BLINK_ADJACENT"]
    else:
        not_blink = pd.Series(True, index=seg.index)
    valid = seg[(seg["BPOGV"] == 1) & not_blink][["TIME", "BPOGX", "BPOGY"]].reset_index(drop=True)
    if len(valid) < MIN_SACCADE_SAMPLES + 1:
        return []

    # identify contiguous runs of valid samples (no large time gaps within a run)
    t_all = valid["TIME"].values
    gap_breaks = np.where(np.diff(t_all) > 0.05)[0] + 1  # >50ms gap = new run
    run_bounds = [0] + list(gap_breaks) + [len(t_all)]

    saccades = []
    for ri in range(len(run_bounds) - 1):
        r0, r1 = run_bounds[ri], run_bounds[ri + 1]
        if r1 - r0 < MIN_SACCADE_SAMPLES + 1:
            continue
        run = valid.iloc[r0:r1].reset_index(drop=True)

        x_smooth = run["BPOGX"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).median().values
        y_smooth = run["BPOGY"].rolling(SMOOTHING_WINDOW, center=True, min_periods=1).median().values
        t = run["TIME"].values

        dt = np.diff(t)
        dx = np.diff(x_smooth)
        dy = np.diff(y_smooth)
        vel_deg_s = np.array([
            _norm_dist_to_deg(dxi, dyi, distance_cm, screen_w_cm, screen_h_cm) / dti
            if 0 < dti < 0.1 else 0.0
            for dxi, dyi, dti in zip(dx, dy, dt)
        ])

        above = vel_deg_s > SPEED_THRESHOLD_DEG_S
        i = 0
        n = len(above)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                onset_idx, offset_idx = i, min(j, n - 1)
                end_sample_idx = min(offset_idx + 1, len(t) - 1)
                if (offset_idx - onset_idx) >= MIN_SACCADE_SAMPLES - 1:
                    amp_deg = _norm_dist_to_deg(
                        x_smooth[end_sample_idx] - x_smooth[onset_idx],
                        y_smooth[end_sample_idx] - y_smooth[onset_idx],
                        distance_cm, screen_w_cm, screen_h_cm)
                    if amp_deg >= MIN_SACCADE_AMPLITUDE_DEG:
                        saccades.append({
                            "onset_time": t[onset_idx], "offset_time": t[end_sample_idx],
                            "start_x": run["BPOGX"].values[onset_idx], "start_y": run["BPOGY"].values[onset_idx],
                            "end_x": run["BPOGX"].values[end_sample_idx], "end_y": run["BPOGY"].values[end_sample_idx],
                            "peak_velocity_deg_s": float(vel_deg_s[onset_idx:offset_idx + 1].max()),
                            "amplitude_deg": round(amp_deg, 2),
                            "duration_s": t[end_sample_idx] - t[onset_idx],
                        })
                i = j + 1
            else:
                i += 1
    saccades.sort(key=lambda s: s["onset_time"])
    return saccades


# ──────────────────────────────────────────────────────────────
# ANNOTATION / TRIAL HELPERS
# ──────────────────────────────────────────────────────────────

def build_ann_lookup_full(ann_df):
    """
    Returns {(id, condition): {verb_onset_s, verb_offset_s, target_onset_s, target_word}}.
    verb_onset_s = ROOT 'start' (NOT 'end' -- deliberately different from
    the rest of this project, which uses verb OFFSET for the critical
    window start).
    """
    verbs = ann_df[ann_df.WordRole == "ROOT"][["id", "SentenceRole", "start", "end"]].rename(
        columns={"start": "verb_onset_s", "end": "verb_offset_s", "SentenceRole": "condition"})
    targets = ann_df[ann_df.WordRole == "dobj"][["id", "SentenceRole", "start", "word"]].rename(
        columns={"start": "target_onset_s", "word": "target_word", "SentenceRole": "condition"})
    merged = verbs.merge(targets, on=["id", "condition"])
    lookup = {}
    for _, r in merged.iterrows():
        lookup[(int(r["id"]), str(r["condition"]))] = {
            "verb_onset_s": float(r["verb_onset_s"]), "verb_offset_s": float(r["verb_offset_s"]),
            "target_onset_s": float(r["target_onset_s"]), "target_word": str(r["target_word"]),
        }
    return lookup


def compute_critical_window_probabilities(seg, audio_onset, verb_offset_s, target_onset_s, target_position):
    """
    Raw-sample duration-weighted probability distribution over the 4
    object AOIs (prop_pos1..prop_pos4, summing to 1), reproducing the
    existing quality_control.py measure -- now also excluding blink-
    adjacent samples for a fair comparison against the (blink-filtered)
    saccade-based probabilities computed below.
    Returns (props: dict[pos1..4 -> float] or None, no_object_gaze: bool).
    """
    win_start, win_end = audio_onset + verb_offset_s, audio_onset + target_onset_s
    if "IS_BLINK_ADJACENT" in seg.columns:
        not_blink = ~seg["IS_BLINK_ADJACENT"]
    else:
        not_blink = pd.Series(True, index=seg.index)
    mask = (seg["TIME"] >= win_start) & (seg["TIME"] < win_end) & (seg["BPOGV"] == 1) & not_blink
    w = seg.loc[mask, ["TIME", "BPOGX", "BPOGY"]].copy()
    if w.empty:
        return None, True
    w["aoi"] = [classify_aoi(x, y) for x, y in zip(w["BPOGX"], w["BPOGY"])]
    t_vals = w["TIME"].values
    if len(t_vals) > 1:
        gaps = np.diff(t_vals, append=t_vals[-1] + np.median(np.diff(t_vals)))
        dur = np.clip(gaps, 0, np.median(np.diff(t_vals)) * 3)
    else:
        dur = np.array([0.0] * len(t_vals))
    obj_keys = ["pos1", "pos2", "pos3", "pos4"]
    durs = {k: float(dur[w["aoi"].values == k].sum()) for k in obj_keys}
    total = sum(durs.values())
    if total == 0:
        return None, True
    props = {k: durs[k] / total for k in obj_keys}
    return props, False


def compute_saccade_probabilities(saccades, audio_onset, verb_offset_s, target_onset_s):
    """
    The saccade-based analogue of the function above: instead of duration-
    weighting raw samples, this counts every saccade LANDING (endpoint)
    inside the critical window and normalizes into a probability
    distribution over the 4 object AOIs. A genuinely different measure --
    discrete saccadic events rather than continuous dwell time.
    Returns (props: dict[pos1..4 -> float] or None, n_saccades_landed: int).
    """
    win_start, win_end = audio_onset + verb_offset_s, audio_onset + target_onset_s
    obj_keys = ["pos1", "pos2", "pos3", "pos4"]
    counts = {k: 0 for k in obj_keys}
    n_total = 0
    for s in saccades:
        if win_start <= s["offset_time"] < win_end:
            aoi = classify_aoi(s["end_x"], s["end_y"])
            if aoi in counts:
                counts[aoi] += 1
            n_total += 1
    total_on_object = sum(counts.values())
    if total_on_object == 0:
        return None, n_total
    props = {k: counts[k] / total_on_object for k in obj_keys}
    return props, n_total


# ──────────────────────────────────────────────────────────────
# PER-SUBJECT PROCESSING
# ──────────────────────────────────────────────────────────────

def discover_subjects(input_dir):
    input_dir = Path(input_dir)
    subjects = []
    for tsv_path in sorted(input_dir.glob("*.tsv")):
        stem = tsv_path.stem
        if stem.endswith("_log"):
            continue
        csv_path = input_dir / f"{stem}.csv"
        if csv_path.exists():
            log_path = input_dir / f"{stem}_log.txt"
            subjects.append({"subject_id": stem, "tsv": tsv_path, "csv": csv_path,
                              "log": log_path if log_path.exists() else None,
                              "group_csv": find_group_csv_for_subject(stem)})
    return subjects


def process_subject(subj, ann_lookup):
    subject_id = subj["subject_id"]
    print(f"\n=== {subject_id} ===")
    gaze_df = pd.read_csv(subj["tsv"], sep="\t", low_memory=False)
    gaze_df["USER"] = gaze_df["USER"].fillna("0").astype(str).str.strip()
    trial_df = pd.read_csv(subj["csv"], low_memory=False)
    trial_df = trial_df[trial_df["audio_file"].notna() & (trial_df["audio_file"].astype(str) != "undefined")]
    trial_df = trial_df.sort_values("count_trial_loop").reset_index(drop=True)

    log_text = subj["log"].read_text() if subj["log"] else ""
    distance_cm, screen_w_cm, screen_h_cm = _load_session_geometry(log_text)
    print(f"  geometry: distance={distance_cm}cm, screen={screen_w_cm}x{screen_h_cm}cm")

    gaze_df, blink_events = detect_and_mask_blinks(gaze_df)
    print(f"  blinks detected: {len(blink_events)} "
          f"(mean duration {1000*np.mean([b['duration_s'] for b in blink_events]):.0f}ms)"
          if blink_events else "  blinks detected: 0")

    group_df = None
    if subj.get("group_csv") is not None and subj["group_csv"].exists():
        group_df = pd.read_csv(subj["group_csv"]).set_index("id")
        print(f"  matched group CSV: {subj['group_csv'].name} (object names available)")
    else:
        print(f"  [warn] no group CSV found for {subject_id} (looked for "
              f"participant_group{subject_id_to_group_number(subject_id)}.csv in {GROUP_CSV_DIR}) "
              f"-- object names will be unavailable, only pos1-4 labels")

    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    starts = events[events.USER == "START_TRIAL"]["TIME"].values

    rows = []
    for i in range(min(len(starts), len(trial_df))):
        t_start = starts[i]
        t_end = starts[i + 1] if i + 1 < len(starts) else gaze_df["TIME"].max()
        seg = gaze_df[(gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)]
        seg_events = events[(events["TIME"] >= t_start) & (events["TIME"] < t_end)]

        t_row = trial_df.iloc[i]
        try:
            audio_id = int(str(t_row["audio_file"]).split("_")[0])
        except ValueError:
            continue
        condition = str(t_row["condition"])
        target_position = int(t_row["target_position"])
        key = (audio_id, condition)
        if key not in ann_lookup:
            continue
        info = ann_lookup[key]

        audio_onset_vals = seg_events.loc[seg_events.USER == "AUDIO_FILE_ONSET_LOG", "TIME"].values
        audio_offset_vals = seg_events.loc[seg_events.USER == "AUDIO_FILE_OFFSET", "TIME"].values
        if len(audio_onset_vals) == 0:
            continue
        audio_onset = audio_onset_vals[0]
        audio_offset = audio_offset_vals[0] if len(audio_offset_vals) else seg["TIME"].max()

        verb_onset_abs = audio_onset + info["verb_onset_s"]

        # detect all saccades in the trial (blink-adjacent samples already
        # excluded inside detect_saccades), then find the first one starting
        # at/after verb onset and before audio offset
        saccades = detect_saccades(seg, distance_cm, screen_w_cm, screen_h_cm)
        candidates = [s for s in saccades if verb_onset_abs <= s["onset_time"] < audio_offset]
        first_sacc = min(candidates, key=lambda s: s["onset_time"]) if candidates else None

        # two probability distributions over the 4 object AOIs, same
        # critical window (verb offset -> target onset) for direct comparison:
        #   raw-sample duration-weighted (existing quality_control.py method)
        #   saccade-landing count-based (new)
        raw_props, no_object_gaze = compute_critical_window_probabilities(
            seg, audio_onset, info["verb_offset_s"], info["target_onset_s"], target_position)
        sacc_props, n_sacc_landed = compute_saccade_probabilities(
            saccades, audio_onset, info["verb_offset_s"], info["target_onset_s"])

        row = {
            "subject_id": subject_id, "trial": i, "sentence_id": audio_id, "condition": condition,
            "sentence": str(t_row.get("sentence", "")), "target_position": target_position,
            "n_saccades_detected": len(saccades), "n_saccades_after_verb_onset": len(candidates),
        }
        if first_sacc is not None:
            landing_aoi = classify_aoi(first_sacc["end_x"], first_sacc["end_y"])
            row.update({
                "first_saccade_latency_s": round(first_sacc["onset_time"] - verb_onset_abs, 4),
                "first_saccade_duration_s": round(first_sacc["duration_s"], 4),
                "first_saccade_peak_vel_deg_s": round(first_sacc["peak_velocity_deg_s"], 1),
                "first_saccade_landing_aoi": landing_aoi,
                "first_saccade_is_target": landing_aoi == f"pos{target_position}",
                "first_saccade_is_object": landing_aoi in ["pos1", "pos2", "pos3", "pos4"],
            })
        else:
            row.update({
                "first_saccade_latency_s": np.nan, "first_saccade_duration_s": np.nan,
                "first_saccade_peak_vel_deg_s": np.nan, "first_saccade_landing_aoi": None,
                "first_saccade_is_target": None, "first_saccade_is_object": None,
            })

        tgt_key = f"pos{target_position}"
        for k in ["pos1", "pos2", "pos3", "pos4"]:
            row[f"rawsample_prop_{k}"] = round(raw_props[k], 4) if raw_props else np.nan
            row[f"saccade_prop_{k}"] = round(sacc_props[k], 4) if sacc_props else np.nan
        row["prop_target_critical_window"] = round(raw_props[tgt_key], 4) if raw_props else np.nan
        row["saccade_prop_target"] = round(sacc_props[tgt_key], 4) if sacc_props else np.nan
        row["no_object_gaze_critical_window"] = no_object_gaze
        row["n_saccades_landed_critical_window"] = n_sacc_landed

        # object names (not just pos1-4 labels), if a group CSV was matched
        if group_df is not None and audio_id in group_df.index:
            g_row = group_df.loc[audio_id]
            row["target_object"] = str(g_row[f"position{target_position}"])
            landing_aoi = row.get("first_saccade_landing_aoi")
            if landing_aoi in ["pos1", "pos2", "pos3", "pos4"]:
                pos_num = int(landing_aoi[-1])
                row["first_saccade_chosen_object"] = str(g_row[f"position{pos_num}"])
            elif landing_aoi is not None:
                row["first_saccade_chosen_object"] = landing_aoi  # "subject" or "elsewhere"
            else:
                row["first_saccade_chosen_object"] = None
        else:
            row["target_object"] = None
            row["first_saccade_chosen_object"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} trials processed, {df['first_saccade_landing_aoi'].notna().sum()} with a "
          f"detected first saccade after verb onset")
    return df, blink_events


# ──────────────────────────────────────────────────────────────
# COMPARISON: first-saccade method vs. raw-sample critical-window method
# ──────────────────────────────────────────────────────────────

def compare_methods(df):
    comparable = df[df["first_saccade_is_target"].notna() & df["prop_target_critical_window"].notna()].copy()
    if comparable.empty:
        print("No trials with both measures available for comparison.")
        return comparable, {}

    comparable["critical_window_favors_target"] = comparable["prop_target_critical_window"] > 0.25
    comparable["methods_agree"] = comparable["first_saccade_is_target"] == comparable["critical_window_favors_target"]

    agreement_pct = 100 * comparable["methods_agree"].mean()
    corr = comparable["first_saccade_is_target"].astype(float).corr(comparable["prop_target_critical_window"])

    summary = {
        "n_comparable_trials": len(comparable),
        "pct_agreement": round(agreement_pct, 1),
        "point_biserial_correlation": round(corr, 3),
        "pct_first_saccade_on_target": round(100 * comparable["first_saccade_is_target"].mean(), 1),
        "mean_prop_target_critical_window": round(comparable["prop_target_critical_window"].mean(), 3),
    }
    return comparable, summary


# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────

def plot_latency_by_condition(df, out_path):
    d = df[df["first_saccade_latency_s"].notna()]
    if d.empty:
        print(f"  [plot] skipped {out_path} -- no first-saccade data")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cond, color in [("restrictive", "#1f77b4"), ("non-restrictive", "#d62728")]:
        vals = d.loc[d["condition"] == cond, "first_saccade_latency_s"]
        if len(vals):
            ax.hist(vals, bins=20, alpha=0.5, label=f"{cond} (n={len(vals)})", color=color)
    ax.axvline(0, color="black", linestyle=":", linewidth=1, label="verb onset")
    ax.set_xlabel("First-saccade latency after verb onset (s)")
    ax.set_ylabel("Trial count")
    ax.set_title("First-Saccade Latency by Condition")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_landing_accuracy_by_condition(df, out_path):
    d = df[df["first_saccade_is_target"].notna()]
    if d.empty:
        print(f"  [plot] skipped {out_path} -- no first-saccade landing data")
        return
    summary = d.groupby("condition")["first_saccade_is_target"].mean() * 100
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(summary.index, summary.values, color=["#1f77b4", "#d62728"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%",
                ha="center", fontsize=9)
    ax.axhline(25, color="#888888", linestyle=":", linewidth=1, label="chance (25%)")
    ax.set_ylabel("First saccades landing on target (%)")
    ax.set_title("First-Saccade Landing Accuracy by Condition")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_method_comparison(comparable, out_path):
    if comparable.empty:
        print(f"  [plot] skipped {out_path} -- no comparable trials")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(0)
    for is_target, color, label in [(True, "#2ca02c", "First saccade -> target"),
                                     (False, "#d62728", "First saccade -> distractor/other")]:
        sub = comparable[comparable["first_saccade_is_target"] == is_target]
        jitter = rng.uniform(-0.08, 0.08, size=len(sub))
        ax.scatter(np.full(len(sub), int(is_target)) + jitter, sub["prop_target_critical_window"],
                   s=25, alpha=0.6, color=color, label=label)
    ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1, label="chance (0.25)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Distractor/other", "Target"])
    ax.set_xlabel("First saccade landing (binary)")
    ax.set_ylabel("prop_target \u2014 raw-sample critical-window method")
    ax.set_title("Do the Two Methods Agree, Per Trial?")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_first_saccade_accuracy_by_participant(df, out_path):
    """Per-participant (not just per-condition) first-saccade landing accuracy."""
    d = df[df["first_saccade_is_target"].notna()]
    if d.empty:
        print(f"  [plot] skipped {out_path} -- no first-saccade landing data")
        return
    summary = d.groupby("subject_id")["first_saccade_is_target"].mean() * 100
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(summary) + 2), 4.5))
    bars = ax.bar(summary.index, summary.values, color="#2ca02c")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.axhline(25, color="#888888", linestyle=":", linewidth=1, label="chance (25%)")
    ax.set_ylabel("First saccades landing on target (%)")
    ax.set_title("First-Saccade Landing Accuracy per Participant")
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_probability_comparison(df, out_path):
    """Compares the raw-sample duration-weighted vs. saccade-landing-count
    probability distributions over the 4 object AOIs, averaged per condition."""
    pos_cols_raw = ["rawsample_prop_pos1", "rawsample_prop_pos2", "rawsample_prop_pos3", "rawsample_prop_pos4"]
    pos_cols_sacc = ["saccade_prop_pos1", "saccade_prop_pos2", "saccade_prop_pos3", "saccade_prop_pos4"]
    conditions = sorted(df["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.5), squeeze=False)
    axes = axes[0]
    x = np.arange(4); width = 0.35
    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        raw_means = sub[pos_cols_raw].mean().values
        sacc_means = sub[pos_cols_sacc].mean().values
        ax.bar(x - width / 2, raw_means, width, label="Raw-sample (duration-weighted)", color="#1f77b4")
        ax.bar(x + width / 2, sacc_means, width, label="Saccade-landing (count-based)", color="#ff7f0e")
        ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1, label="chance (0.25)")
        ax.set_xticks(x); ax.set_xticklabels(["pos1", "pos2", "pos3", "pos4"])
        ax.set_ylabel("Mean probability"); ax.set_title(cond)
        ax.legend(fontsize=7)
    fig.suptitle("Probability Distribution: Raw-Sample vs. Saccade-Landing Method", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    if not ANNOTATION_CSV.exists():
        raise SystemExit(f"ERROR: annotation CSV not found at {ANNOTATION_CSV}")
    ann_df = pd.read_csv(ANNOTATION_CSV)
    ann_lookup = build_ann_lookup_full(ann_df)

    subjects = discover_subjects(INPUT_DIR)
    if not subjects:
        raise SystemExit(f"ERROR: no subject .tsv/.csv pairs found in {INPUT_DIR}")
    print(f"Found {len(subjects)} subject(s): {', '.join(s['subject_id'] for s in subjects)}")

    all_dfs = []
    all_blinks = []
    for subj in subjects:
        try:
            subj_df, blink_events = process_subject(subj, ann_lookup)
            all_dfs.append(subj_df)
            for b in blink_events:
                b["subject_id"] = subj["subject_id"]
            all_blinks.extend(blink_events)
        except Exception as e:
            print(f"  [subject FAILED] {subj['subject_id']}: {e}")

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if df.empty:
        raise SystemExit("No trials processed successfully.")

    df.to_csv(CSV_DIR / "first_saccade_per_trial.csv", index=False)
    print(f"\nSaved: {CSV_DIR / 'first_saccade_per_trial.csv'}  ({len(df)} rows)")

    if all_blinks:
        blink_df = pd.DataFrame(all_blinks)
        blink_df.to_csv(CSV_DIR / "blinks_detected.csv", index=False)
        print(f"Saved: {CSV_DIR / 'blinks_detected.csv'}  ({len(blink_df)} blinks across "
              f"{blink_df['subject_id'].nunique()} subject(s))")

    print("\n" + "=" * 90)
    print("PROBABILITY DISTRIBUTIONS: raw-sample vs. saccade-landing method (critical window)")
    print("=" * 90)
    prob_summary = (df.groupby(["subject_id", "condition"])[
        ["rawsample_prop_pos1", "rawsample_prop_pos2", "rawsample_prop_pos3", "rawsample_prop_pos4",
         "saccade_prop_pos1", "saccade_prop_pos2", "saccade_prop_pos3", "saccade_prop_pos4"]]
        .mean().round(3))
    print(prob_summary.to_string())
    prob_summary.to_csv(CSV_DIR / "probability_distributions_summary.csv")

    print("\n" + "=" * 90)
    print("FIRST-SACCADE SUMMARY BY SUBJECT x CONDITION")
    print("=" * 90)
    sacc_summary = (df[df["first_saccade_is_target"].notna()]
                    .groupby(["subject_id", "condition"])
                    .agg(n_trials=("trial", "count"),
                         pct_on_target=("first_saccade_is_target", lambda s: round(100 * s.mean(), 1)),
                         mean_latency_ms=("first_saccade_latency_s", lambda s: round(1000 * s.mean(), 0)))
                    )
    print(sacc_summary.to_string())
    sacc_summary.to_csv(CSV_DIR / "first_saccade_summary.csv")

    if "target_object" in df.columns and df["target_object"].notna().any():
        print("\n" + "=" * 90)
        print("WHAT OBJECT DID EACH PARTICIPANT CHOOSE? (first saccade after verb onset)")
        print("=" * 90)
        choice_cols = ["subject_id", "trial", "sentence", "condition", "target_object",
                        "first_saccade_chosen_object", "first_saccade_latency_s"] \
            if "sentence" in df.columns else \
            ["subject_id", "trial", "sentence_id", "condition", "target_object",
             "first_saccade_chosen_object", "first_saccade_latency_s"]
        choice_df = df[[c for c in choice_cols if c in df.columns]].copy()
        choice_df["chose_target"] = choice_df["target_object"] == choice_df["first_saccade_chosen_object"]
        choice_df.to_csv(CSV_DIR / "object_choices_per_trial.csv", index=False)
        print(f"Saved: {CSV_DIR / 'object_choices_per_trial.csv'}  ({len(choice_df)} rows)")
        print("\nSample (first 10 rows):")
        print(choice_df.head(10).to_string(index=False))

        print("\nMost frequently chosen object per participant:")
        for sid in sorted(choice_df["subject_id"].unique()):
            sub = choice_df[(choice_df["subject_id"] == sid) & choice_df["first_saccade_chosen_object"].notna()]
            if sub.empty:
                continue
            top = sub["first_saccade_chosen_object"].value_counts().head(3)
            print(f"  {sid}: {dict(top)}")
    else:
        print("\n[note] No group CSV matched for any subject -- object names unavailable, "
              "only pos1-4 labels are in the output CSV. Set GROUP_CSV_DIR / "
              "SUBJECT_GROUP_CSV_OVERRIDE to enable object-name resolution.")

    print("\n" + "=" * 90)
    print("METHOD COMPARISON: first-saccade landing vs. raw-sample critical-window prop_target")
    print("=" * 90)
    comparable, summary = compare_methods(df)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if not comparable.empty:
        comparable.to_csv(CSV_DIR / "method_comparison_per_trial.csv", index=False)
        pd.DataFrame([summary]).to_csv(CSV_DIR / "method_comparison_summary.csv", index=False)

    print("\nGenerating plots...")
    plot_latency_by_condition(df, PLOTS_DIR / "first_saccade_latency_by_condition.png")
    plot_landing_accuracy_by_condition(df, PLOTS_DIR / "first_saccade_accuracy_by_condition.png")
    plot_first_saccade_accuracy_by_participant(df, PLOTS_DIR / "first_saccade_accuracy_by_participant.png")
    plot_method_comparison(comparable, PLOTS_DIR / "method_comparison.png")
    plot_probability_comparison(df, PLOTS_DIR / "probability_comparison.png")

    return df, comparable, summary


if __name__ == "__main__":
    main()