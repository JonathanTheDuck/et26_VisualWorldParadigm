"""
First-Saccade-After-Verb-Onset Analysis -- Engbert & Kliegl (2003) detector
============================================================================
Saccade DETECTOR is the Engbert & Kliegl (2003) algorithm end to end,
following the 7 steps below exactly (matches the lecture slide):

  1. Calculate horizontal and vertical velocities (per axis, in deg/s).
  2. Estimate the SD of velocity per axis with the median estimator:
         x_std = sqrt( median(x_vel**2) - median(x_vel)**2 )
     (same for y_std). This is computed FRESH for every valid run of
     gaze, i.e. it adapts to that specific stretch of data's own noise
     level -- unlike a flat deg/s threshold, which does not.
  3. Normalise: x_vel_norm = x_vel / x_std,  y_vel_norm = y_vel / y_std.
  4. Threshold = factor (lambda, e.g. 6) x the per-axis SD -- already
     folded into step 3's normalisation, so the test in step 5 compares
     against lambda directly, not against a raw deg/s number.
  5. Elliptic (circular, since both axes are already in SD units)
     criterion:  x_vel_norm**2 + y_vel_norm**2 > lambda**2.
  6. Require >= n_samples (e.g. 3) CONSECUTIVE samples above threshold
     for the run to count as a saccade.
  7. Monocular only (this dataset has no second eye to fuse).

No position smoothing and no separate minimum-amplitude gate are used --
E&K's method does not call for either; the per-run adaptive normalisation
in steps 2-3 is what is supposed to make the detector robust to a given
recording's own noise level, instead of smoothing the signal beforehand.

Velocities use E&K's own 5-sample moving-average differentiator, not a
plain adjacent-sample difference:

    v(n) = [x(n+2) + x(n+1) - x(n-1) - x(n-2)] / (6 * dt)

with a simple 2-sample fallback for the two samples nearest each end of
a run (where the 5-point window doesn't fit), and the outermost sample
on each end left undefined (NaN) since no differentiator can cover it --
NaNs are always treated as "below threshold" so they can never seed or
extend a saccade.

Positions are converted from normalized screen coordinates (BPOGX/BPOGY)
into signed degrees of visual angle relative to screen centre, per axis,
using the session's own physical geometry (viewer distance and screen
size in cm, read out of the pygaze log) -- so x_vel and y_vel are true
deg/s time series, exactly what steps 1-2 assume.

SACCADES ONLY (2026-08-27): the raw-sample dwell-time probability method
has been removed entirely -- this script now measures anticipatory gaze
purely from detected saccades. There is a single probability measure
(saccade-landing proportions) and a single critical window: VERB ONSET
to NOUN ONSET, matching the window already used for the first-saccade
latency/landing measure (previously the probability window ran from verb
OFFSET to noun onset -- a different, later window).

THREE VERIFICATION CONCERNS (2026-08-28), addressed below:

  1. CRITICAL WINDOW = verb onset -> noun onset, consistently, for BOTH
     probability measures now. compute_saccade_probabilities() was already
     correctly bounded this way, but the SEPARATE first-saccade candidate
     window (feeding first_saccade_landing_aoi / first_saccade_chosen_object)
     extended all the way to audio_offset -- the end of the whole trial's
     audio, well past noun onset. That gap is now closed: see
     `noun_onset_abs` in process_subject(), which bounds BOTH measures
     identically.

  2. POSITION -> OBJECT MAPPING, verified per subject against real data
     rather than assumed from a formula. Confirmed directly against the
     actual participant_groupN.csv files: the same 4 objects appear at
     every item across all 8 groups (only their position rotates), and
     target_position always points at the correct semantic target object
     regardless of group -- so the mapping logic itself is sound. The
     genuine risk is WHICH group a given subject actually belongs to: the
     'subject_number % 8 + 1' formula was already found, in an earlier
     session, to be wrong for at least one real subject.
       Fix: detect_subject_group() no longer trusts that formula as ground
     truth. Each group CSV independently carries its own condition and
     target_position per item; a subject's own trial CSV independently
     records what condition/target_position they actually experienced.
     Confirmed empirically (against the real 8-group files uploaded to
     this project) that (target_position, condition) is a PERFECT, UNIQUE
     fingerprint -- every group scores 1.0 against its own data and 0.0
     against all 7 others. detect_subject_group() therefore checks a
     subject's trial data against all 8 candidate groups and picks the one
     that actually matches, falling back to the formula (with a loud
     warning) only if detection is inconclusive. SUBJECT_GROUP_CSV_OVERRIDE
     still exists as a manual, highest-priority override if a subject's
     correct group is known by other means.
       Tested against the real group CSVs with synthetic subjects run
     under a DIFFERENT group than their formula guess would predict:
     detection caught and auto-corrected both cases at 100% confidence.
     Quantified what the old formula-trusting code would have done for one
     such case: 50/50 items would have gotten the wrong target_object
     label, silently.

  3. PROBABILITY CALCULATION (downstream, in human_llm_comparison.py, not
     this script) -- with 1 and 2 fixed here, first_saccade_chosen_object
     is now a reliable, correctly-windowed, correctly-object-mapped
     per-trial choice, suitable as the basis for "count of participants
     who chose object X / total participants for that condition."

  ALSO SURFACED (not fixed here -- these live in human_llm_comparison.py,
  flagged so they aren't missed): two real, confirmed mismatches between
  this pipeline's output and the LLM predictions CSV used for comparison:
    - Object names here use HYPHENS ('toy-car', 'paper-plane'); the LLM
      CSV uses SPACES ('toy car', 'paper plane'). Confirmed directly
      against both real files.
    - 'id' in participant_groupN.csv is 0-INDEXED (items 0-49); the LLM
      CSV's 'Item' column is 1-INDEXED (1-50) for the same 50 items.
      Confirmed directly: group CSV id=0 ('the boy will eat the cake') is
      the same item as LLM CSV Item=1.

Run: python saccade_analysis_ek.py
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

# ── Repo-relative paths ──────────────────────────────────────────────
# This script lives at 02_analytics/Analysis_pipeline_m/Input/ inside the
# repo, so three levels up from HERE is the repo root
# (et26_VisualWorldParadigm/et26_VisualWorldParadigm/). Everything below is
# derived from that -- nothing is hardcoded to a specific machine/username,
# so `git clone` + run works for anyone regardless of where they cloned it.
REPO_ROOT = (HERE / ".." / ".." ).resolve()
EXPERIMENT_DIR = REPO_ROOT / "01_experiment"
STIMULI_DIR = EXPERIMENT_DIR / "stimuli"

# Raw eye-tracker recordings (.tsv/.csv/_log.txt per subject, possibly in
# per-subject subfolders -- discover_subjects() below searches recursively)
INPUT_DIR = EXPERIMENT_DIR / "recordings" / "SUBJECTS (FINAL)"

# Sentence/word timing annotations
ANNOTATION_CSV = STIMULI_DIR / "annotation_audiov2.csv"

# Composited stimulus images -- not read directly by this script, but this
# is the source of truth AOI_TOPLEFT_PX (above) was pixel-measured from;
# kept here so anyone re-verifying or re-deriving AOI geometry knows where
# to look rather than guessing.
IMG_COMPOSITION_DIR = STIMULI_DIR / "img_composition"

# Group/arrangement CSVs (participant_group1.csv ... participant_group8.csv)
# -- this is where the actual OBJECT NAMES per position live (the subject's
# own trial CSV only has position NUMBERS, never object names).
GROUP_CSV_DIR = STIMULI_DIR / "creatingDataStructure" / "participant_groups_8"

OUT_DIR = HERE / "saccade_output_ek"
PLOTS_DIR = OUT_DIR / "plots"
CSV_DIR = OUT_DIR / "csv"
for d in (OUT_DIR, PLOTS_DIR, CSV_DIR):
    d.mkdir(exist_ok=True)

SUBJECT_GROUP_CSV_OVERRIDE = {
    # "subject-4": "participant_group5.csv",
}


def subject_id_to_group_number(subject_id):
    """group = (subject_number % 8) + 1 -- the FORMULA'S guess. Kept only as
    a fallback and as a point of comparison for detect_subject_group() below
    -- this formula was already found, in an earlier session, to NOT match
    the actual arrangement a subject was run with. It is no longer trusted
    as ground truth by itself; see detect_subject_group()."""
    matches = re.findall(r"\d+", subject_id)
    if not matches:
        return None
    return (int(matches[-1]) % 8) + 1


def normalize_condition_local(raw):
    """Local copy of the condition-label normalizer (lowercase, 'non*' ->
    non-restrictive) so this section doesn't depend on anything defined
    later in the file."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()
    return "non-restrictive" if s.startswith("non") else "restrictive"


def load_all_groups(group_dir):
    """Loads all 8 participant_groupN.csv files, indexed by 'id' (the
    numeric prefix of audio_file, e.g. '0_r.wav' -> 0 -- confirmed against
    the real files to be 0-indexed, items 0-49). Returns {group_num: df}."""
    groups = {}
    for g in range(1, 9):
        p = group_dir / f"participant_group{g}.csv"
        if p.exists():
            groups[g] = pd.read_csv(p).set_index("id")
        else:
            print(f"  [warn] missing participant_group{g}.csv in {group_dir}")
    return groups


def detect_subject_group(subject_id, trial_df, groups):
    """
    CONCERN 2 (position->object mapping correctness), resolved by DATA, not
    by the formula: each participant_groupN.csv independently carries its
    own 'condition' and 'target_position' per item, and a subject's own
    trial CSV independently records what condition and target_position
    that subject actually experienced on each item. Since (target_position,
    condition) together form a UNIQUE fingerprint per group -- confirmed
    empirically against the real 8 group files, where every group scores a
    perfect 1.0 match against itself and 0.0 against all 7 others -- the
    correct group for a subject can be detected directly from their own
    trial data instead of assumed from the 'subject_number % 8 + 1' formula
    (which was already found to be wrong for at least one real subject).

    Returns (chosen_group_num, source, scores) where source is one of:
      "override"        -- SUBJECT_GROUP_CSV_OVERRIDE took precedence
      "detected"         -- a confident, unambiguous match was found
      "formula_fallback" -- detection was inconclusive; fell back to the
                             formula's guess (with a loud warning)
    `scores` is {group_num: match_rate} for every group checked, for
    transparency in the console output.
    """
    if subject_id in SUBJECT_GROUP_CSV_OVERRIDE:
        override_name = SUBJECT_GROUP_CSV_OVERRIDE[subject_id]
        override_num = int(re.findall(r"\d+", override_name)[0])
        return override_num, "override", {}

    subj_lookup = {}
    for _, t_row in trial_df.iterrows():
        try:
            audio_id = int(str(t_row["audio_file"]).split("_")[0])
        except (ValueError, KeyError):
            continue
        cond = normalize_condition_local(t_row.get("condition"))
        pos = t_row.get("target_position")
        if cond is None or pd.isna(pos):
            continue
        subj_lookup[audio_id] = (int(pos), cond)

    formula_guess = subject_id_to_group_number(subject_id)

    if not subj_lookup:
        print(f"  [warn] {subject_id}: trial CSV had no usable condition/target_position "
              f"data -- cannot auto-detect group, falling back to formula guess "
              f"(group {formula_guess})")
        return formula_guess, "formula_fallback", {}

    scores = {}
    for gnum, gdf in groups.items():
        matches, total = 0, 0
        for item_id, (tp, cond) in subj_lookup.items():
            if item_id not in gdf.index:
                continue
            total += 1
            g_tp = int(gdf.loc[item_id, "target_position"])
            g_cond = normalize_condition_local(gdf.loc[item_id, "condition"])
            if g_tp == tp and g_cond == cond:
                matches += 1
        scores[gnum] = (matches / total) if total else 0.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_num, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    confident = best_score >= 0.9 and (best_score - runner_up_score) >= 0.5
    if confident:
        agrees = "matches" if best_num == formula_guess else "DIFFERS FROM"
        print(f"  group detected: participant_group{best_num}.csv "
              f"(match={best_score:.0%}, runner-up={runner_up_score:.0%}) -- "
              f"{agrees} the formula's guess (group {formula_guess})")
        return best_num, "detected", scores
    else:
        print(f"  [CRITICAL] {subject_id}: group detection was INCONCLUSIVE "
              f"(best={best_num} at {best_score:.0%}, runner-up={runner_up_score:.0%}) "
              f"-- falling back to formula guess (group {formula_guess}), but this "
              f"subject's object-level results should be checked manually. Consider "
              f"adding an explicit entry to SUBJECT_GROUP_CSV_OVERRIDE once the "
              f"correct group is confirmed by other means.")
        return formula_guess, "formula_fallback", scores


# ── Engbert & Kliegl (2003) detector parameters ─────────────────────────
EK_LAMBDA = 6.0            # threshold factor (multiples of the median-based SD)
EK_MIN_SAMPLES = 3         # consecutive above-threshold samples required (step 6)

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

# ── AOI geometry: pixel-verified positions (Jonathan's top-left-corner
#    measurements off the real composited stimuli, cross-checked to within
#    a few px of an independent pixel-measurement of the same images) ──
AOI_TOPLEFT_PX = {
    "pos1": (1891, 429),
    "pos2": (1423, 915),
    "pos3": (697, 915),
    "pos4": (229, 429),
}
AOI_BOX = 440
SCREEN_W, SCREEN_H = 2560, 1440
_AOI_PX = {name: (x, y, x + AOI_BOX, y + AOI_BOX) for name, (x, y) in AOI_TOPLEFT_PX.items()}
_AOI_PX["subject"] = (1042, 111, 1512, 791)  # unchanged, from corrected_aoi.py
AOI_NORM = {name: (x1 / SCREEN_W, y1 / SCREEN_H, x2 / SCREEN_W, y2 / SCREEN_H)
            for name, (x1, y1, x2, y2) in _AOI_PX.items()}


def classify_aoi(x, y):
    for name, (x1, y1, x2, y2) in AOI_NORM.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "elsewhere"


# ──────────────────────────────────────────────────────────────
# GEOMETRY: normalized screen coordinates -> signed degrees of visual angle
# ──────────────────────────────────────────────────────────────

def _load_session_geometry(log_text):
    """Pulls distance-to-screen and physical screen size (cm) straight out
    of the session's own pygaze log -- no assumed/hardcoded values."""
    m_dist = re.search(r"distance between participant and display:\s*([\d.]+)\s*cm", log_text)
    m_size = re.search(r"display size in cm:\s*([\d.]+)x([\d.]+)", log_text)
    distance_cm = float(m_dist.group(1)) if m_dist else 57.0
    if m_size:
        screen_w_cm, screen_h_cm = float(m_size.group(1)), float(m_size.group(2))
    else:
        screen_w_cm, screen_h_cm = 33.8, 27.1
    return distance_cm, screen_w_cm, screen_h_cm


def _norm_xy_to_deg(x_norm, y_norm, distance_cm, screen_w_cm, screen_h_cm):
    """
    Signed per-axis degrees of visual angle, relative to SCREEN CENTRE
    (x_norm=0.5, y_norm=0.5), using the actual viewing geometry. E&K's
    algorithm operates on x(t) and y(t) as independent position time
    series in degrees, so unlike a pure-displacement magnitude formula,
    this needs to preserve sign and be usable as an absolute position.
    """
    dx_cm = (x_norm - 0.5) * screen_w_cm
    dy_cm = (y_norm - 0.5) * screen_h_cm
    deg_x = math.degrees(math.atan2(dx_cm, distance_cm))
    deg_y = math.degrees(math.atan2(dy_cm, distance_cm))
    return deg_x, deg_y


def _norm_dist_to_deg(dx_norm, dy_norm, distance_cm, screen_w_cm, screen_h_cm):
    """Displacement magnitude (not position) in degrees -- used only for
    reporting saccade amplitude, exact formula (not small-angle)."""
    dx_cm = dx_norm * screen_w_cm
    dy_cm = dy_norm * screen_h_cm
    dist_cm = math.hypot(dx_cm, dy_cm)
    return math.degrees(2 * math.atan2(dist_cm / 2, distance_cm))


# ──────────────────────────────────────────────────────────────
# BLINK MASKING
# ──────────────────────────────────────────────────────────────

def detect_and_mask_blinks(gaze_df):
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


# ──────────────────────────────────────────────────────────────
# ENGBERT & KLIEGL (2003) VELOCITY + SACCADE DETECTION
# ──────────────────────────────────────────────────────────────

def _ek_velocity(pos, dt):
    """
    E&K's 5-sample moving-average differentiator:
        v(n) = [x(n+2) + x(n+1) - x(n-1) - x(n-2)] / (6*dt)
    dt is the (run-local) median inter-sample interval. The two samples
    nearest each end fall back to a simple centred 2-sample difference;
    the outermost sample on each end is left as NaN (undefined velocity).
    """
    n = len(pos)
    vel = np.full(n, np.nan)
    for i in range(n):
        if 2 <= i <= n - 3:
            vel[i] = (pos[i + 2] + pos[i + 1] - pos[i - 1] - pos[i - 2]) / (6 * dt)
        elif 1 <= i <= n - 2:
            vel[i] = (pos[i + 1] - pos[i - 1]) / (2 * dt)
        # i == 0 or i == n-1 stay NaN
    return vel


def _ek_median_std(vel):
    """x_std = sqrt(median(x_vel**2) - median(x_vel)**2), NaNs excluded."""
    v = vel[~np.isnan(vel)]
    if len(v) == 0:
        return np.nan
    val = np.nanmedian(v ** 2) - np.nanmedian(v) ** 2
    return math.sqrt(val) if val > 0 else np.nan


def detect_saccades_ek(seg, distance_cm, screen_w_cm, screen_h_cm,
                        lam=EK_LAMBDA, min_samples=EK_MIN_SAMPLES):
    """
    Detects saccades in one trial's raw gaze segment using the Engbert &
    Kliegl (2003) algorithm (steps 1-7, see module docstring). Runs
    across invalid (BPOGV==0) or blink-adjacent samples are never
    bridged -- each contiguous valid run is processed, and its own
    median-based x_std/y_std are computed fresh from that run's data
    (step 2), so the effective velocity threshold adapts per run rather
    than being a single fixed deg/s value.
    """
    if "IS_BLINK_ADJACENT" in seg.columns:
        not_blink = ~seg["IS_BLINK_ADJACENT"]
    else:
        not_blink = pd.Series(True, index=seg.index)
    valid = seg[(seg["BPOGV"] == 1) & not_blink][["TIME", "BPOGX", "BPOGY"]].reset_index(drop=True)
    if len(valid) < min_samples + 4:  # need enough samples for the 5-pt differentiator + a run
        return []

    t_all = valid["TIME"].values
    gap_breaks = np.where(np.diff(t_all) > 0.05)[0] + 1  # >50ms gap = new run
    run_bounds = [0] + list(gap_breaks) + [len(t_all)]

    saccades = []
    for ri in range(len(run_bounds) - 1):
        r0, r1 = run_bounds[ri], run_bounds[ri + 1]
        if r1 - r0 < min_samples + 4:
            continue
        run = valid.iloc[r0:r1].reset_index(drop=True)
        t = run["TIME"].values
        dt = float(np.median(np.diff(t)))
        if not (0 < dt < 0.1):
            continue

        # step: raw normalized coords -> signed degrees, per axis (position)
        deg_xy = [
            _norm_xy_to_deg(x, y, distance_cm, screen_w_cm, screen_h_cm)
            for x, y in zip(run["BPOGX"].values, run["BPOGY"].values)
        ]
        deg_x = np.array([p[0] for p in deg_xy])
        deg_y = np.array([p[1] for p in deg_xy])

        # 1. velocities (deg/s), per axis
        vx = _ek_velocity(deg_x, dt)
        vy = _ek_velocity(deg_y, dt)

        # 2. median-based SD estimator, per axis, computed fresh for this run
        x_std = _ek_median_std(vx)
        y_std = _ek_median_std(vy)
        if not (x_std and x_std > 0) or not (y_std and y_std > 0):
            continue  # degenerate run (e.g. constant signal) -- can't normalise

        # 3. normalise
        vx_norm = vx / x_std
        vy_norm = vy / y_std

        # 4-5. elliptic/circular threshold test in normalised (SD) units
        with np.errstate(invalid="ignore"):
            radius2 = vx_norm ** 2 + vy_norm ** 2
        above = radius2 > (lam ** 2)
        above = np.where(np.isnan(radius2), False, above)

        # 6. require >= min_samples consecutive samples above threshold
        i = 0
        n = len(above)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                onset_idx, offset_idx = i, j - 1
                if (offset_idx - onset_idx + 1) >= min_samples:
                    peak_vel = float(np.nanmax(np.sqrt(vx[onset_idx:offset_idx + 1] ** 2 +
                                                         vy[onset_idx:offset_idx + 1] ** 2)))
                    amp_deg = _norm_dist_to_deg(
                        run["BPOGX"].values[offset_idx] - run["BPOGX"].values[onset_idx],
                        run["BPOGY"].values[offset_idx] - run["BPOGY"].values[onset_idx],
                        distance_cm, screen_w_cm, screen_h_cm)
                    saccades.append({
                        "onset_time": t[onset_idx], "offset_time": t[offset_idx],
                        "start_x": run["BPOGX"].values[onset_idx], "start_y": run["BPOGY"].values[onset_idx],
                        "end_x": run["BPOGX"].values[offset_idx], "end_y": run["BPOGY"].values[offset_idx],
                        "peak_velocity_deg_s": round(peak_vel, 1),
                        "amplitude_deg": round(amp_deg, 2),
                        "duration_s": t[offset_idx] - t[onset_idx],
                        "x_std_deg_s": round(x_std, 3), "y_std_deg_s": round(y_std, 3),
                    })
                i = j
            else:
                i += 1
    saccades.sort(key=lambda s: s["onset_time"])
    return saccades


# ──────────────────────────────────────────────────────────────
# ANNOTATION / TRIAL HELPERS
# ──────────────────────────────────────────────────────────────

def build_ann_lookup_full(ann_df):
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


def compute_saccade_probabilities(saccades, audio_onset, verb_onset_s, target_onset_s):
    """
    Saccade-landing probability distribution over the 4 objects, counted
    within the CRITICAL WINDOW = verb ONSET -> noun onset.
    """
    win_start, win_end = audio_onset + verb_onset_s, audio_onset + target_onset_s
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

    for tsv_path in sorted(input_dir.rglob("*.tsv")):
        stem = tsv_path.stem

        if stem.endswith("_log"):
            continue

        csv_path = tsv_path.parent / f"{stem}.csv"
        log_path = tsv_path.parent / f"{stem}_log.txt"

        if csv_path.exists():
            # NOTE: group is no longer pre-assigned here by formula -- it's
            # detected per-subject inside process_subject() once trial_df is
            # loaded, since detection needs each trial's own condition and
            # target_position (see detect_subject_group()).
            subjects.append({
                "subject_id": stem,
                "tsv": tsv_path,
                "csv": csv_path,
                "log": log_path if log_path.exists() else None,
            })

    return subjects


def process_subject(subj, ann_lookup, groups):
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
    print(f"  E&K params: lambda={EK_LAMBDA}, min_samples={EK_MIN_SAMPLES}")

    gaze_df, blink_events = detect_and_mask_blinks(gaze_df)
    print(f"  blinks detected: {len(blink_events)} "
          f"(mean duration {1000*np.mean([b['duration_s'] for b in blink_events]):.0f}ms)"
          if blink_events else "  blinks detected: 0")

    group_df = None
    if groups:
        group_num, group_source, _scores = detect_subject_group(subject_id, trial_df, groups)
        if group_num in groups:
            group_df = groups[group_num]
            print(f"  using participant_group{group_num}.csv (source: {group_source}) "
                  f"for object-name resolution")
        else:
            print(f"  [warn] detected/fallback group {group_num} has no loaded CSV -- "
                  f"object names will be unavailable for {subject_id}")
    else:
        print(f"  [warn] no group CSVs loaded at all -- object names will be "
              f"unavailable, only pos1-4 labels")

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
        noun_onset_abs = audio_onset + info["target_onset_s"]

        saccades = detect_saccades_ek(seg, distance_cm, screen_w_cm, screen_h_cm)
        # FIX (concern 1): candidate window for the first-saccade measure now
        # ends at NOUN ONSET, not audio_offset (end of whole trial audio).
        # Previously this window was wider than the verb-onset -> noun-onset
        # window used for the saccade-landing probabilities below, so
        # first_saccade_landing_aoi / first_saccade_chosen_object could
        # reflect a saccade launched AFTER the noun was already heard -- not
        # anticipatory. Both measures are now bounded by the same window.
        candidates = [s for s in saccades if verb_onset_abs <= s["onset_time"] < noun_onset_abs]
        first_sacc = min(candidates, key=lambda s: s["onset_time"]) if candidates else None

        # Saccade-landing probabilities, critical window = verb onset -> noun onset
        sacc_props, n_sacc_landed = compute_saccade_probabilities(
            saccades, audio_onset, info["verb_onset_s"], info["target_onset_s"])

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
            row[f"saccade_prop_{k}"] = round(sacc_props[k], 4) if sacc_props else np.nan
        row["saccade_prop_target"] = round(sacc_props[tgt_key], 4) if sacc_props else np.nan
        row["n_saccades_landed_critical_window"] = n_sacc_landed

        if group_df is not None and audio_id in group_df.index:
            g_row = group_df.loc[audio_id]
            row["target_object"] = str(g_row[f"position{target_position}"])
            landing_aoi = row.get("first_saccade_landing_aoi")
            if landing_aoi in ["pos1", "pos2", "pos3", "pos4"]:
                pos_num = int(landing_aoi[-1])
                row["first_saccade_chosen_object"] = str(g_row[f"position{pos_num}"])
            elif landing_aoi is not None:
                row["first_saccade_chosen_object"] = landing_aoi
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
    ax.set_title("First-Saccade Latency by Condition (E&K detector)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_landing_accuracy_by_condition(df, out_path):
    d = df[df["first_saccade_is_object"] == True].copy()

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
    ax.set_title("First-Saccade Landing Accuracy by Condition (E&K detector)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_first_saccade_accuracy_by_participant(df, out_path):
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
    ax.set_title("First-Saccade Landing Accuracy per Participant (E&K detector)")
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_saccade_probability_distribution(df, out_path):
    """Mean saccade-landing probability per position, per condition."""
    pos_cols_sacc = ["saccade_prop_pos1", "saccade_prop_pos2", "saccade_prop_pos3", "saccade_prop_pos4"]
    conditions = sorted(df["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.5 * len(conditions), 4.5), squeeze=False)
    axes = axes[0]
    x = np.arange(4)
    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        sacc_means = sub[pos_cols_sacc].mean().values
        ax.bar(x, sacc_means, width=0.5, color="#ff7f0e", label="Saccade-landing probability")
        ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1, label="chance (0.25)")
        ax.set_xticks(x); ax.set_xticklabels(["pos1", "pos2", "pos3", "pos4"])
        ax.set_ylabel("Mean probability"); ax.set_title(cond)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Saccade-Landing Probability Distribution (verb onset \u2192 noun onset, E&K detector)",
                 y=1.02, fontsize=13, fontweight="bold")
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

    groups = load_all_groups(GROUP_CSV_DIR)
    if groups:
        print(f"Loaded {len(groups)} participant-group CSVs for auto-detection "
              f"(concern 2) -- group is now DETECTED per subject from their own "
              f"trial data, not assumed from the subject-number formula.")
    else:
        print(f"[warn] no participant_groupN.csv files found in {GROUP_CSV_DIR} -- "
              f"object-name resolution will be unavailable for all subjects")

    all_dfs = []
    all_blinks = []
    for subj in subjects:
        try:
            subj_df, blink_events = process_subject(subj, ann_lookup, groups)
            all_dfs.append(subj_df)
            for b in blink_events:
                b["subject_id"] = subj["subject_id"]
            all_blinks.extend(blink_events)
        except Exception as e:
            print(f"  [subject FAILED] {subj['subject_id']}: {e}")

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if df.empty:
        raise SystemExit("No trials processed successfully.")

    df.to_csv(CSV_DIR / "first_saccade_per_trial_ek.csv", index=False)
    print(f"\nSaved: {CSV_DIR / 'first_saccade_per_trial_ek.csv'}  ({len(df)} rows)")

    if all_blinks:
        blink_df = pd.DataFrame(all_blinks)
        blink_df.to_csv(CSV_DIR / "blinks_detected.csv", index=False)
        print(f"Saved: {CSV_DIR / 'blinks_detected.csv'}  ({len(blink_df)} blinks across "
              f"{blink_df['subject_id'].nunique()} subject(s))")

    print("\n" + "=" * 90)
    print("SACCADE-LANDING PROBABILITY DISTRIBUTIONS (critical window: verb onset -> noun onset)")
    print("=" * 90)
    prob_summary = (df.groupby(["subject_id", "condition"])[
        ["saccade_prop_pos1", "saccade_prop_pos2", "saccade_prop_pos3", "saccade_prop_pos4"]]
        .mean().round(3))
    print(prob_summary.to_string())
    prob_summary.to_csv(CSV_DIR / "probability_distributions_summary.csv")

    print("\n" + "=" * 90)
    print("FIRST-SACCADE SUMMARY BY SUBJECT x CONDITION (E&K detector)")
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
        print("WHAT OBJECT DID EACH PARTICIPANT CHOOSE? (first saccade after verb onset, E&K detector)")
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

    print("\nGenerating plots...")
    plot_latency_by_condition(df, PLOTS_DIR / "first_saccade_latency_by_condition.png")
    plot_landing_accuracy_by_condition(df, PLOTS_DIR / "first_saccade_accuracy_by_condition.png")
    plot_first_saccade_accuracy_by_participant(df, PLOTS_DIR / "first_saccade_accuracy_by_participant.png")
    plot_saccade_probability_distribution(df, PLOTS_DIR / "saccade_probability_distribution.png")

    return df


if __name__ == "__main__":
    main()