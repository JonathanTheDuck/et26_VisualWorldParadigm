
"""Claude finished the response
1784126744034_image.png
1784126765283_image.png
can we plot all these in the code and save all the visualization graphs in different folder and csv files in different folder


Architecting modular plotting functions and organizing output directories systematically
Architecting modular plotting functions and organizing output directories systematically
This is a great, clear spec — let me implement all of these plus the folder reorganization. Let's look at the current file structure first.


View current file structure to find insertion points
View current file structure to find insertion points
Let's set up the folder split first:


Reconciled outdated documentation with current technical implementation
Reconciled outdated documentation with current technical implementation
Now let's find all the save calls and route them to the right subfolder:


Architecting modular visualization pipeline with organized output directories
Architecting modular visualization pipeline with organized output directories
This is a lot of new plot types — let me build them methodically. First, let's add the new computation helpers (per-trial invalid rate is already computed; need gaze-density-by-time-window and example-trial-trace data extraction), then the plot functions, then wire everything into main() with the new folder structure.


Ran a command, viewed a file
Ran a command, viewed a file
Now let's add the new data-extraction functions for the two milestone-4-style plots (subject-onset density windows, and example-trial gaze trace):


Add gaze-density-by-time-window and example-trial-trace data extraction functions
Add gaze-density-by-time-window and example-trial-trace data extraction functions
Now let's update process_subject to compute and return this new data:


Viewed 3 files, edited a file
Viewed 3 files, edited a file
Now let's add the new plot functions. Let's find where the plotting section is:


Ran a command, viewed a file
Ran a command, viewed a file
Let's replace this with the milestone-4-matching 3-panel version (per-subject histograms + combined mean±SD), and add the four new plot functions:


Replace bias plot with milestone-4-style multi-panel version, and add invalid-rate, gaze-on-AOI, density-window, and example-trace plots
Replace bias plot with milestone-4-style multi-panel version, and add invalid-rate, gaze-on-AOI, density-window, and example-trace plots
Now let's update main() to route to the new folders and wire in all the new plots:


Viewed 2 files
Viewed 2 files
Now let's rewrite the save/plot section of main() completely:


Verifying conditional guards prevent variable reference errors
Verifying conditional guards prevent variable reference errors
Let's do a full syntax check and end-to-end test:


Full syntax check and end-to-end test of the expanded script
Full syntax check and end-to-end test of the expanded script
Everything runs end-to-end with the new folder structure. Let's inspect the key new plots to verify they look right:


Confirm folder structure is correctly split
Confirm folder structure is correctly split
Clean separation. Let's view the two brand-new plot types to confirm correctness:


Viewed 2 files, ran 2 commands
Viewed 2 files, ran 2 commands
All plots have real content (non-zero pixel variance), confirming they rendered actual data, not blank canvases. Let's check the example trial trace's actual data made sense:


Sanity check the actual data behind the example trial trace plot
Sanity check the actual data behind the example trial trace plot
This confirms it's working correctly with real data — sentence "The girl will groom the rabbit", critical window at 1.68–1.95s (consistent with previous ~250ms window durations we've seen), target screen position (1006, 980) correctly falls in the lower-left region matching pos3. Let's ship the final script.


Ran a command, read a file
Ran a command, read a file
Done — all the plots from both reference images are now implemented, and outputs are split into separate folders.

Folder structure:

qc_output/
  ├── csv/     — all 8 CSV outputs
  └── plots/   — all 11 PNG outputs
From your first image (milestone-4 style checks), now matching more precisely:

1_calibration_across_subjects.png — unchanged, already matched.
2_invalid_rate_per_trial.png — new: per-trial invalid-rate bar chart, one panel per subject, with a 20% threshold line (matching your reference exactly, was previously only an overall summary number).
3_gaze_bias_across_subjects.png — rebuilt: now shows per-subject BPOGX histograms plus a combined "Mean BPOGX ± SD" bar panel, matching the 3-panel layout in your image.
4_gaze_on_aoi.png — new (this plot type existed before the deterministic-geometry refactor but got dropped — it's back): gaze scatter over the AOI boxes, now using the deterministic geometry, so it no longer needs an image file to render.
From your second image:

5_subject_onset_density_{subject}.png — new: gaze density in 4 time windows relative to SUBJECT_ONSET_LOG (Preview, At onset, After onset, Late), pooled across all trials — shows whether gaze converges to the subject figure right after it appears.
6_example_trial_trace_{subject}.png — new: Gaze X/Y in screen pixels over time for one representative trial, with the target's screen position and critical window marked — the "gaze locks onto target after the verb" plot. I verified this against real data: it auto-selected trial 3 ("The girl will groom the rabbit," restrictive, target at position 3), with a critical window at 1.68–1.95s and a target screen position that correctly falls in the lower-left region matching pos3.
Tested the full pipeline end-to-end — all 11 plots and 8 CSVs generate correctly, and I verified pixel-level content on the new plots (non-blank, real data) plus the underlying data logic (sentence text, timing, target position all check out).


Quality control
PY 
can you draft me experimental design slide


Architecting missing experimental design components and slide structure
Architecting missing experimental design components and slide structure
Let's get a closer look at the deck's actual colors and layout patterns to match them precisely.


Architected experimental design slide with color palette, layout structure, and interactive visualization components
Architected experimental design slide with color palette, layout structure, and interactive visualization components
Excellent — I found something important: slide 4 explicitly says "see next slide" for the counterbalancing fix, but slide 5 jumps straight to "Research Question" instead. This slide is genuinely missing from the deck. Let me build it to match the existing palette and layout conventions exactly, and slot it in as the real "next slide."


Verify the counterbalancing example numbers before putting them on the slide
Verify the counterbalancing example numbers before putting them on the slide
Confirmed clean pattern. Now let's build the slide.


Ran 2 commands, created a file
Ran 2 commands, created a file
Validation passed. Now let's render it for visual QA:


Validated layout integrity and element positioning constraints
Validated layout integrity and element positioning constraints
Let's zoom into the trickiest regions (timeline labels, ellipse diagram) to check for overlap issues at full resolution:


Debugged layout spacing and resolved text box collision issues
Debugged layout spacing and resolved text box collision issues
Let me check the math on the timeline spacing more carefully — the step labels might be tighter than they should be:


Confirm the timeline label overlap issue mathematically
Confirm the timeline label overlap issue mathematically
Confirmed — the label boxes are wider than the spacing between steps. Let's fix that:


Validated label positioning and spacing constraints
Validated label positioning and spacing constraints
Now let's regenerate, re-validate, and re-render for final QA:


Ran 2 commands, viewed a file
Ran 2 commands, viewed a file
Let's also run the content QA check per the skill's checklist:


Validated formulas, styling, and content for finalization
Validated formulas, styling, and content for finalization
This slide slots in exactly where the deck needs it — it directly answers slide 4's "see next slide" reference for the counterbalancing fix, uses the deck's exact color palette (navy/blue/green/orange extracted from the actual slides), matches its panel/badge/banner conventions, and passed both schema validation and content QA.


Ran a command, read a file
Ran a command, read a file
What's on it, and why:

① Trial Timeline — the 5-step trial sequence (Start Trial → Subject Onset → Audio Onset → Critical Window → Audio Offset), with real timing annotations pulled from your actual data (~3.4s to subject onset, ~20ms to audio onset, 200–400ms critical windows) rather than made-up placeholder numbers.

② Display Layout — a schematic of the subject + 4-object ellipse arrangement, based on the exact compose_trial() geometry from your teammate's split_and_compose.py (angles 15°/65°/115°/165°), with the fixed position1–position4 slots labeled.

③ Counterbalancing — the actual 2×4 Graeco-Latin square design: the real condition/position formulas from the script, plus a worked example (item id=0) showing exactly how each of the 8 groups lands in a different (condition × position) cell. I verified this table's numbers by running the actual formula in Python before putting it on the slide.

The top banner explicitly ties it back to Issue 3 from your "Issues Found" slide (the item–condition confound), since that's the fix this design represents.

Process notes: I pulled the deck's actual colors by sampling pixels from your real slides (navy 
#1A1A2E, blue 
#276EF7, green 
#1EA66D, orange 
#E88A00) rather than guessing, matched the 10×7.5" slide size exactly, and caught a real overlap bug during QA — the timeline step labels were wider than the space between steps — before shipping. Passed schema validation and content QA (no leftover placeholder text).

It's a standalone slide rather than merged into the full deck — if you'd like it spliced in as slide 5 of milestone4_final_43_updated.pptx (pushing Research Question to slide 6), I can do that next.


Experimental design slide
Presentation · PPTX 
where did u get the counterbalancing formulations from ?


Traced formulas to shared script and validated against provided data
Traced formulas to shared script and validated against provided data
Directly from the split_and_compose.py script you uploaded earlier — specifically the participant_row_8() function (the 8-group counterbalanced design generator) and its docstring. The exact lines are:

python
def participant_row_8(
    sid: int,
    group: int,
    sentence_texts: dict[int, tuple[str, str]],
    item_objects: dict[int, dict[str, str]],
) -> dict:
    
    8-group design row for item `sid` as seen by `group` (0-7): condition AND
    target position are fully counterbalanced (a 2x4 Graeco-Latin square).

        condition = restrictive if (sid + group) % 2 == 0 else non-restrictive
        rotation  = (sid + group // 2) % 4
    ...
    
    restrictive = (sid + group) % 2 == 0
    rotation    = (sid + group // 2) % 4
    return build_participant_row(sid, rotation, restrictive, sentence_texts, item_objects)
And target_position comes from build_participant_row():

python
"target_position": rotation + 1,
— which is where the slide's position = (id + group // 2) % 4 + 1 comes from (the +1 converts 0-indexed rotation to the 1-indexed position1–position4 naming).

I didn't just take the docstring's word for it, either — I ran the actual formula in Python against id=0 across all 8 groups before putting the worked-example table on the slide, and separately, back when I first added SUBJECT_GROUP_CSV support to the QC scripts, I cross-checked this formula's predictions against the real participant_group5.csv file you sent (e.g. confirmed id=44 → condition=restrictive, target_position=3, matching row 44 of that CSV exactly). So the numbers on the slide are grounded in both the source code and the actual data file, not just one or the other.

Instead of generic histograms, we can make publication-quality figures such as:

Participant validity
% valid gaze for each participant
Threshold line at 80% or 90%
Condition balance
Restrictive vs Non-restrictive trials
Target position balance (1–4)
Sampling frequency
Logged Hz vs Measured Hz
Difference (%)
Event completeness
START_TRIAL
AUDIO_ONSET
AUDIO_OFFSET
STOP_TRIAL
Resolution check
Percentage of x-values outside [0,1]
Percentage of y-values outside [0,1]
Gaze accuracy
Distribution of Target Advantage
Target vs Distractor gaze
No-object gaze
Per-subject boxplot
one last tiem can you help me plot thes e

Show more

Architecting publication-quality plotting functions for six analyses
Architecting publication-quality plotting functions for six analyses
Good call — let's build these as proper publication-quality figures: consistent styling, clean typography, value labels, appropriate thresholds, 300dpi. Let me add all six to the script.


Ran a command, edited a file
Ran a command, edited a file
Now let's add the six new publication-quality figure functions:


Reconciling tick label handling across categorical plots
Reconciling tick label handling across categorical plots
Let's wire these into main() and test:


Resolved test warnings and refined plotting best practices
Resolved test warnings and refined plotting best practices
All plots generated — let's clean up those two harmless-but-sloppy warnings properly:


Edited 2 files
Edited 2 files
Let's rerun clean and verify all figures visually:


Clean rerun to confirm warnings are gone
Clean rerun to confirm warnings are gone
Clean. Now let's inspect each of the six figures:


Resolved to verify data programmatically rather than visually
Resolved to verify data programmatically rather than visually
Correct — matches the ~146Hz vs 60Hz mismatch we found much earlier (144% difference). Let's ship the final script.


Ran a command, read a file
Ran a command, read a file
Added all six as new fig_* functions, saved to plots/fig1_... through fig6_.... Shared publication styling applied globally: clean sans-serif type, no top/right spines, subtle gridlines, consistent 6-color palette, value labels on bars, 300dpi output.

fig1_participant_validity.png — sorted bar chart, % valid per participant, both 80% and 90% threshold lines.
fig2_condition_balance.png — two panels: (A) restrictive vs. non-restrictive counts, (B) target-position 1–4 balance, both per participant.
fig3_sampling_frequency.png — (A) logged vs. measured Hz grouped bars, (B) % difference, color-flagged red above a 15% discrepancy threshold. Verified against real numbers: correctly shows the 60Hz→146.4Hz, +144% mismatch we found earlier.
fig4_event_completeness.png — grouped bars for exactly the four markers you listed (START_TRIAL, AUDIO_ONSET, AUDIO_OFFSET, STOP_TRIAL), % complete per participant, with a 100% reference line.
fig5_resolution_check.png — % of X and % of Y values outside [0,1], per participant.
fig6_gaze_accuracy.png — 4-panel: (A) target-advantage distribution histogram, (B) target vs. distractor mean gaze probability per subject with a chance line at 0.25, (C) no-object-gaze rate per subject, (D) per-subject boxplot of target-advantage with jittered individual trials.
Tested the full pipeline end-to-end — all six render cleanly with no warnings, at proper resolution (fig6 is 3260×2739px, reflecting the 300dpi setting), and I spot-checked the underlying numbers against the raw data rather than just trusting the plots looked fine.

These are added alongside the existing milestone-4-style plots (calibration, invalid-rate-per-trial, gaze-bias, gaze-on-AOI, subject-onset density, example-trial trace) rather than replacing them — let me know if you'd rather I remove the older, more basic versions where they now overlap with these (e.g. the old validity_across_subjects.png vs. the new fig1).


Quality control
PY 





Claude is AI and can make mistakes. Please double-check responses.
Quality control · PY

Eye-Tracking Visual World Paradigm — Quality Control (multi-subject)
======================================================================
Aggregates QC across EVERY subject found in the Input folder, covering:
 
  1. Screen resolution / AOI coordinate-space sanity check
  2. AOI geometry — computed deterministically from the stimulus-generation
     script's own placement constants (ellipse slot positions around the
     subject), combined with the confirmed OpenSesame display parameters:
     scale=0.8, 2560x1440 canvas, 2400x1400 source images, centered
     placement. Identical for every trial; no per-image measurement needed.
  3. Valid-sample rate (BPOGV==1), overall and per trial, per subject
  4. Time-unit cross-check: TSV clock vs. annotation clock, and the
     eye tracker's actual sampling frequency vs. what each subject's log
     claims
  4.2 Event-marker QC per subject: are START_TRIAL / SUBJECT_ONSET_LOG /
      AUDIO_FILE_ONSET_LOG / AUDIO_FILE_OFFSET / STOP_TRIAL all present,
      correctly ordered, and at sensible latencies?
  5.  Condition / target-position balance per subject (verifies the
      restrictive vs. non-restrictive counterbalancing and target-position
      rotation implemented by each participant's group/arrangement design)
  6.  Valid gaze samples restricted to AOI + critical window, and a
      duration-weighted "gaze-on-target" probability per subject/condition
      — the closest thing to an accuracy/performance measure this design
      supports (there's no explicit per-trial fixation target the way the
      calibration task has, so this is the anticipatory-looking analogue)
  7.  QC visualizations (milestone-4 style: calibration, validity, bias,
      gaze-on-stimulus) generated PER SUBJECT, plus cross-subject
      comparison plots (calibration/validity/gaze-on-target across all
      subjects at once)
 
Multi-subject discovery
------------------------
Scans INPUT_DIR for every {subject_id}.tsv + {subject_id}.csv pair
(+ optional {subject_id}_log.txt). Each subject can optionally be matched
to their own group/arrangement CSV (e.g. participant_group5.csv) via
SUBJECT_GROUP_CSV below — this is used to look up the correct per-trial
stimulus image for AOI measurement. Without a group-CSV match, a subject
still gets every other QC check; only the AOI-based gaze-accuracy check
falls back to a single default image (flagged clearly when this happens).
 
Run: python quality_control.py
"""
 
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
# ── Publication-quality plot style (applies globally) ───────────
PUB_COLORS = {
    "primary": "#2C5F8A", "secondary": "#E67E22", "tertiary": "#27AE60",
    "quaternary": "#C0392B", "quinary": "#8E44AD", "senary": "#16A085",
    "grid_line": "#333333",
}
PUB_PALETTE = [PUB_COLORS["primary"], PUB_COLORS["secondary"], PUB_COLORS["tertiary"],
               PUB_COLORS["quaternary"], PUB_COLORS["quinary"], PUB_COLORS["senary"]]
 
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "medium",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
 
 
def _bar_value_labels(ax, bars, fmt="{:.1f}", offset=0.5, fontsize=8.5):
    """Print the value on top of each bar — standard for publication bar charts."""
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color="#222222")
 
HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE
ANNOTATION_CSV = HERE / "annotation_audiov2.csv"
OUT_DIR = HERE / "qc_output"
PLOTS_DIR = OUT_DIR / "plots"
CSV_DIR = OUT_DIR / "csv"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)
 
# Group/arrangement CSVs (participant_group1.csv ... participant_group8.csv)
# live in a separate directory from the raw gaze/trial data. Update this
# path to match your machine:
GROUP_CSV_DIR = Path(r"C:\Users\ramas\Documents\GitHub\et26_VisualWorldParadigm"
                      r"\et26_VisualWorldParadigm\01_experiment\stimuli"
                      r"\creatingDataStructure\participant_groups_8")
 
# The stimulus images (image_sub/image_nosub referenced inside each group
# CSV, e.g. 45_pos3_sub.png / 45_pos3_nosub.png) live here:
IMAGES_DIR = Path(r"C:\Users\ramas\Documents\GitHub\et26_VisualWorldParadigm"
                   r"\et26_VisualWorldParadigm\01_experiment\stimuli\img_composition")
 
REQUIRED_EVENTS = ["START_TRIAL", "SUBJECT_ONSET_LOG", "AUDIO_FILE_ONSET_LOG",
                   "AUDIO_FILE_OFFSET", "STOP_TRIAL"]
 
 
def subject_id_to_group_number(subject_id):
    """
    Maps a subject id to its participant_group number using the confirmed
    formula: group = (subject_number % 8) + 1  (e.g. subject-4 -> group 5).
    Pulls the LAST integer found in the subject_id string (handles
    "subject-4", "subject4", "j6" style ids alike). Returns None if no
    number can be extracted.
    """
    matches = re.findall(r"\d+", subject_id)
    if not matches:
        return None
    subject_num = int(matches[-1])
    return (subject_num % 8) + 1
 
 
# Manual overrides, if the formula above doesn't hold for a particular
# subject (e.g. {"subject-4": "participant_group5.csv"}) — takes priority
# over the automatic formula-based mapping when present.
SUBJECT_GROUP_CSV_OVERRIDE = {
    # "subject-4": "participant_group5.csv",
}
 
 
def find_group_csv_for_subject(subject_id):
    if subject_id in SUBJECT_GROUP_CSV_OVERRIDE:
        p = GROUP_CSV_DIR / SUBJECT_GROUP_CSV_OVERRIDE[subject_id]
        return p if p.exists() else None
    group_num = subject_id_to_group_number(subject_id)
    if group_num is None:
        return None
    p = GROUP_CSV_DIR / f"participant_group{group_num}.csv"
    return p if p.exists() else None
 
# ──────────────────────────────────────────────────────────────
# AOI GEOMETRY — DETERMINISTIC, derived directly from split_and_compose.py's
# compose_trial() constants, not measured per-image.
#
# compose_trial() places the subject at a fixed box and the 4 objects at
# fixed positions on an ellipse around it (SEMICIRCLE_ANGLES = [15,65,115,
# 165] degrees) — these placement constants NEVER vary by item or rotation;
# only WHICH OBJECT PHOTO occupies each slot changes. Verified against the
# real composed image (1_pos1_sub.png): every one of the 4 objects landed
# exactly where this geometry predicts (cake/TL -> slot0 -> right side;
# toy-car/TR -> slot1 -> bottom-right; toy-train/BR -> slot2 -> bottom-left;
# ball/BL -> slot3 -> left side — all confirmed pixel-for-pixel).
#
# CSV columns position1..position4 are ALSO rotation-invariant physical
# slots (position1 is always slot0/15 degrees, etc — only the object
# identity listed for that position changes with rotation), which matches
# this script's pos1..pos4 naming directly: no relabeling needed.
#
# This replaces the earlier flood-fill image measurement entirely: it's
# simpler, avoids the pale-object content-detection issue altogether (the
# AOI is the ALLOCATED slot, not detected visible pixels), is identical
# across every trial, and doesn't require any image file to be present.
#
# Then: OpenSesame scale=0.8 explicit flat scale, 2560x1440 canvas,
# 2400x1400 source images, centered placement (confirmed from the actual
# `draw image ... scale=0.8` OpenSesame item).
# ──────────────────────────────────────────────────────────────
import math
 
# compose_trial() constants (split_and_compose.py)
_COMPOSE_SCALE = 2
CANVAS_W, CANVAS_H = 1200 * _COMPOSE_SCALE, 700 * _COMPOSE_SCALE   # = 2400, 1400
_SUBJECT_SIZE = (420 * _COMPOSE_SCALE, 300 * _COMPOSE_SCALE)
_SUBJECT_Y = 50 * _COMPOSE_SCALE
_OPTION_SIZE = (200 * _COMPOSE_SCALE, 200 * _COMPOSE_SCALE)
_SEMICIRCLE_GAP = 95 * _COMPOSE_SCALE
_SEMICIRCLE_ANGLES = [15, 65, 115, 165]   # degrees; slot0..slot3 = position1..position4
 
IMG_W, IMG_H = CANVAS_W, CANVAS_H   # 2400, 1400 — the composed image IS this canvas
 
# OpenSesame display transform (confirmed from the actual `draw image` item)
SCALE = 0.8
SCREEN_W, SCREEN_H = 2560, 1440
_DISP_W, _DISP_H = IMG_W * SCALE, IMG_H * SCALE
_X_OFFSET = (SCREEN_W - _DISP_W) / 2
_Y_OFFSET = (SCREEN_H - _DISP_H) / 2
 
 
def _img_to_screen_px(x, y):
    return x * SCALE + _X_OFFSET, y * SCALE + _Y_OFFSET
 
 
def _img_box_to_norm(x1, y1, x2, y2):
    sx1, sy1 = _img_to_screen_px(x1, y1)
    sx2, sy2 = _img_to_screen_px(x2, y2)
    return (sx1 / SCREEN_W, sy1 / SCREEN_H, sx2 / SCREEN_W, sy2 / SCREEN_H)
 
 
def compute_aoi_boxes_img_px():
    """
    The 5 AOI boxes (pos1-4 + subject) in IMAGE-PIXEL space (2400x1400
    canvas), computed directly from compose_trial()'s own placement
    constants. Identical for every item/rotation/subject-variant.
    """
    center_x = CANVAS_W // 2
    center_y = _SUBJECT_Y + _SUBJECT_SIZE[1] // 2
    a = _SUBJECT_SIZE[0] / 2 + _SEMICIRCLE_GAP + _OPTION_SIZE[0] / 2
    b = _SUBJECT_SIZE[1] / 2 + _SEMICIRCLE_GAP + _OPTION_SIZE[1] / 2
 
    boxes = {}
    for k in range(1, 5):   # position1..position4 == slot0..slot3, rotation-invariant
        angle = math.radians(_SEMICIRCLE_ANGLES[k - 1])
        ox = center_x + a * math.cos(angle)
        oy = center_y + b * math.sin(angle)
        boxes[f"pos{k}"] = (ox - _OPTION_SIZE[0] / 2, oy - _OPTION_SIZE[1] / 2,
                             ox + _OPTION_SIZE[0] / 2, oy + _OPTION_SIZE[1] / 2)
 
    sx1 = (CANVAS_W - _SUBJECT_SIZE[0]) // 2
    boxes["subject"] = (sx1, _SUBJECT_Y, sx1 + _SUBJECT_SIZE[0], _SUBJECT_Y + _SUBJECT_SIZE[1])
    return boxes
 
 
DEFAULT_AOI_NORM = {name: _img_box_to_norm(*box) for name, box in compute_aoi_boxes_img_px().items()}
 
 
def classify_aoi(x, y, aoi_norm=DEFAULT_AOI_NORM):
    for name, (x1, y1, x2, y2) in aoi_norm.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "elsewhere"
 
 
 
# ──────────────────────────────────────────────────────────────
# SUBJECT DISCOVERY
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
            subjects.append({
                "subject_id": stem, "tsv": tsv_path, "csv": csv_path,
                "log": log_path if log_path.exists() else None,
                "group_csv": find_group_csv_for_subject(stem),
            })
        else:
            print(f"  [discover] skipping {tsv_path.name}: no matching {stem}.csv found")
    return subjects
 
 
# ──────────────────────────────────────────────────────────────
# PER-SUBJECT QC CHECKS
# ──────────────────────────────────────────────────────────────
 
def check_resolution(subject_id, gaze_df, log_text):
    m = re.search(r"display resolution:\s*(\d+)x(\d+)", log_text) if log_text else None
    logged_res = m.groups() if m else None
    valid = gaze_df[gaze_df["BPOGV"] == 1]
    pct_x_outside = 100 * ((valid["BPOGX"] < 0) | (valid["BPOGX"] > 1)).mean() if len(valid) else np.nan
    pct_y_outside = 100 * ((valid["BPOGY"] < 0) | (valid["BPOGY"] > 1)).mean() if len(valid) else np.nan
    return {"subject_id": subject_id, "logged_resolution": logged_res,
            "pct_x_outside_01": round(pct_x_outside, 2), "pct_y_outside_01": round(pct_y_outside, 2)}
 
 
def check_validity(subject_id, gaze_df, segments):
    overall = 100 * gaze_df["BPOGV"].mean()
    per_trial = pd.Series([100 * seg["BPOGV"].mean() if len(seg) else np.nan for seg in segments])
    return {"subject_id": subject_id, "pct_valid_overall": round(overall, 2),
            "pct_valid_min_trial": round(per_trial.min(), 2), "pct_valid_median_trial": round(per_trial.median(), 2)}, per_trial
 
 
def check_time_units(subject_id, gaze_df, ann_df, events, log_text):
    m = re.search(r"samplerate:\s*([\d.]+)\s*Hz", log_text) if log_text else None
    logged_hz = float(m.group(1)) if m else None
 
    diffs = gaze_df["TIME"].diff().dropna()
    small = diffs[diffs < 0.05]
    measured_hz = 1000 / (small.median() * 1000) if len(small) else np.nan
    mismatch = bool(logged_hz and abs(measured_hz - logged_hz) > 0.15 * logged_hz)
 
    audio_on = events.loc[events.USER == "AUDIO_FILE_ONSET_LOG", "TIME"].values
    audio_off = events.loc[events.USER == "AUDIO_FILE_OFFSET", "TIME"].values
    n = min(len(audio_on), len(audio_off))
    measured_dur = (audio_off[:n] - audio_on[:n]).mean() if n else np.nan
    ann_dur = ann_df.groupby(["id", "SentenceRole"])["end"].max().mean()
    dur_diff_pct = 100 * abs(measured_dur - ann_dur) / ann_dur if ann_dur else np.nan
 
    return {"subject_id": subject_id, "logged_hz": logged_hz, "measured_hz": round(measured_hz, 1),
            "samplerate_mismatch": mismatch, "measured_audio_dur_mean": round(measured_dur, 3),
            "annotation_dur_mean": round(ann_dur, 3), "duration_diff_pct": round(dur_diff_pct, 1)}
 
 
def check_event_markers(subject_id, events, n_trials_expected):
    counts = {ev: int((events["USER"] == ev).sum()) for ev in REQUIRED_EVENTS}
    starts = events.loc[events.USER == "START_TRIAL", "TIME"].values
    subj = events.loc[events.USER == "SUBJECT_ONSET_LOG", "TIME"].values
    audio_on = events.loc[events.USER == "AUDIO_FILE_ONSET_LOG", "TIME"].values
    n = min(len(starts), len(subj), len(audio_on))
    lat_start_subj = subj[:n] - starts[:n]
    lat_subj_audio = audio_on[:n] - subj[:n]
    row = {"subject_id": subject_id, "n_trials_expected": n_trials_expected}
    for ev in REQUIRED_EVENTS:
        row[f"n_{ev}"] = counts[ev]
        row[f"pct_{ev}_complete"] = round(100 * counts[ev] / n_trials_expected, 1) if n_trials_expected else np.nan
    row["n_ordering_violations"] = int((lat_start_subj < 0).sum() + (lat_subj_audio < 0).sum())
    return row
 
 
def check_condition_balance(subject_id, trial_df):
    cond_counts = trial_df["condition"].value_counts().to_dict()
    pos_counts = trial_df["target_position"].value_counts().sort_index().to_dict()
    return {"subject_id": subject_id, **{f"n_{k}": v for k, v in cond_counts.items()},
            **{f"n_target_pos{k}": v for k, v in pos_counts.items()}}
 
 
def build_ann_lookup(ann_df):
    verbs = ann_df[ann_df.WordRole == "ROOT"][["id", "SentenceRole", "end"]].rename(
        columns={"end": "verb_offset_s", "SentenceRole": "condition"})
    targets = ann_df[ann_df.WordRole == "dobj"][["id", "SentenceRole", "start", "word"]].rename(
        columns={"start": "target_onset_s", "word": "target_word", "SentenceRole": "condition"})
    merged = verbs.merge(targets, on=["id", "condition"])
    lookup = {}
    for _, r in merged.iterrows():
        lookup[(int(r["id"]), str(r["condition"]))] = (float(r["verb_offset_s"]), float(r["target_onset_s"]))
    return lookup
 
 
def check_group_consistency(subject_id, trial_df, group_df):
    """
    Cross-checks this subject's own trial CSV against their matched group/
    arrangement CSV: for every trial, does the actual condition and
    target_position match what the group design specifies for that
    sentence id? A real data-integrity check — independent of AOI geometry
    (which is now deterministic and doesn't need this at all).
    """
    n_checked = n_cond_mismatch = n_pos_mismatch = 0
    for _, row in trial_df.iterrows():
        try:
            audio_id = int(str(row["audio_file"]).split("_")[0])
        except ValueError:
            continue
        if audio_id not in group_df.index:
            continue
        n_checked += 1
        g_row = group_df.loc[audio_id]
        if str(g_row["condition"]) != str(row["condition"]):
            n_cond_mismatch += 1
        if int(g_row["target_position"]) != int(row["target_position"]):
            n_pos_mismatch += 1
    return {"subject_id": subject_id, "n_trials_checked": n_checked,
            "n_condition_mismatch": n_cond_mismatch, "n_target_position_mismatch": n_pos_mismatch}
 
 
def compute_gaze_accuracy(subject_id, gaze_df, trial_df, ann_lookup):
    """
    For every trial: get valid samples in the critical window, classify
    into AOIs using the deterministic AOI geometry (identical for every
    trial — see compute_aoi_boxes_img_px), and compute duration-weighted
    prop_target / prop_distractor_mean / target_advantage.
    """
    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    starts = events[events.USER == "START_TRIAL"]["TIME"].values
    audio_onset = events[events.USER == "AUDIO_FILE_ONSET_LOG"]["TIME"].values
 
    rows = []
    for i in range(min(len(starts), len(trial_df))):
        t_start = starts[i]
        t_end = starts[i + 1] if i + 1 < len(starts) else gaze_df["TIME"].max()
        seg = gaze_df[(gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)]
 
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
        verb_off, target_on = ann_lookup[key]
        win_start, win_end = audio_onset[i] + verb_off, audio_onset[i] + target_on
 
        mask = (seg["TIME"] >= win_start) & (seg["TIME"] < win_end) & (seg["BPOGV"] == 1)
        w = seg.loc[mask, ["TIME", "BPOGX", "BPOGY"]].copy()
        if w.empty:
            continue
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
        tgt_key = f"pos{target_position}"
        prop_target = durs[tgt_key] / total if total > 0 else 0.0
        prop_distractor_mean = (total - durs[tgt_key]) / 3 / total if total > 0 else 0.0
 
        rows.append({
            "subject_id": subject_id, "trial": i, "sentence_id": audio_id, "condition": condition,
            "target_position": target_position, "prop_target": round(prop_target, 4),
            "prop_distractor_mean": round(prop_distractor_mean, 4),
            "target_advantage": round(prop_target - prop_distractor_mean, 4),
            "no_object_gaze": total == 0,
        })
    return pd.DataFrame(rows)
 
 
def compute_aoi_classified_samples(gaze_df, trial_df, ann_lookup):
    """
    Every valid gaze sample inside a critical window, across all trials,
    classified into an AOI. Used for the "does gaze land where the objects
    actually are" sanity-check scatter plot.
    """
    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    starts = events[events.USER == "START_TRIAL"]["TIME"].values
    audio_onset = events[events.USER == "AUDIO_FILE_ONSET_LOG"]["TIME"].values
 
    rows = []
    for i in range(min(len(starts), len(trial_df))):
        t_start = starts[i]
        t_end = starts[i + 1] if i + 1 < len(starts) else gaze_df["TIME"].max()
        seg = gaze_df[(gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)]
        t_row = trial_df.iloc[i]
        try:
            audio_id = int(str(t_row["audio_file"]).split("_")[0])
        except ValueError:
            continue
        condition = str(t_row["condition"])
        key = (audio_id, condition)
        if key not in ann_lookup:
            continue
        verb_off, target_on = ann_lookup[key]
        win_start, win_end = audio_onset[i] + verb_off, audio_onset[i] + target_on
        mask = (seg["TIME"] >= win_start) & (seg["TIME"] < win_end) & (seg["BPOGV"] == 1)
        w = seg.loc[mask, ["BPOGX", "BPOGY"]].copy()
        if w.empty:
            continue
        w["aoi"] = [classify_aoi(x, y) for x, y in zip(w["BPOGX"], w["BPOGY"])]
        rows.append(w)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["BPOGX", "BPOGY", "aoi"])
 
 
# Time windows relative to SUBJECT_ONSET_LOG, matching the milestone-4 spec
SUBJECT_ONSET_WINDOWS = [
    ("Preview\n(-1.5s to 0)", -1.5, 0.0),
    ("At onset\n(0-150ms)", 0.0, 0.15),
    ("After onset\n(150ms-500ms)", 0.15, 0.5),
    ("Late\n(500ms-1.5s)", 0.5, 1.5),
]
 
 
def compute_subject_onset_density(gaze_df, trial_df):
    """
    For every trial, take valid gaze samples in each SUBJECT_ONSET_WINDOWS
    bucket (time relative to that trial's SUBJECT_ONSET_LOG), pooled across
    all trials. Used for the "gaze converges to subject on screen-onset"
    density-by-time-window plot.
    Returns {window_label: DataFrame(BPOGX, BPOGY)}.
    """
    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    starts = events[events.USER == "START_TRIAL"]["TIME"].values
 
    buckets = {label: [] for label, _, _ in SUBJECT_ONSET_WINDOWS}
    for i in range(min(len(starts), len(trial_df))):
        t_start = starts[i]
        t_end = starts[i + 1] if i + 1 < len(starts) else gaze_df["TIME"].max()
        seg = gaze_df[(gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)]
        seg_events = events[(events["TIME"] >= t_start) & (events["TIME"] < t_end)]
        onset_vals = seg_events.loc[seg_events.USER == "SUBJECT_ONSET_LOG", "TIME"].values
        if len(onset_vals) == 0:
            continue
        onset = onset_vals[0]
        rel_t = seg["TIME"] - onset
        valid = seg["BPOGV"] == 1
        for label, lo, hi in SUBJECT_ONSET_WINDOWS:
            m = valid & (rel_t >= lo) & (rel_t < hi)
            if m.any():
                buckets[label].append(seg.loc[m, ["BPOGX", "BPOGY"]])
 
    return {label: (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["BPOGX", "BPOGY"]))
            for label, dfs in buckets.items()}
 
 
def select_example_trial(subject_id, trial_df, gaze_acc_df, condition="restrictive"):
    """Pick a representative trial for the gaze-trace-vs-target-position plot:
    prefer a trial in `condition` that actually had object gaze in the
    critical window; fall back to the first trial of that condition."""
    if gaze_acc_df is not None and not gaze_acc_df.empty:
        good = gaze_acc_df[(gaze_acc_df["subject_id"] == subject_id) &
                            (gaze_acc_df["condition"] == condition) & (~gaze_acc_df["no_object_gaze"])]
        if not good.empty:
            return int(good.iloc[0]["trial"])
    cond_trials = trial_df[trial_df["condition"] == condition]
    return int(cond_trials.index[0]) if not cond_trials.empty else None
 
 
def extract_trial_trace(gaze_df, trial_df, ann_lookup, trial_idx):
    """
    Gaze X/Y in SCREEN PIXELS (not normalized) over time relative to audio
    onset for one trial, plus the target's screen-pixel position and the
    critical-window boundaries — everything needed for the "gaze locks onto
    target after the verb" line plot.
    """
    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    starts = events[events.USER == "START_TRIAL"]["TIME"].values
    t_start = starts[trial_idx]
    t_end = starts[trial_idx + 1] if trial_idx + 1 < len(starts) else gaze_df["TIME"].max()
    seg = gaze_df[(gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)].copy()
    seg_events = events[(events["TIME"] >= t_start) & (events["TIME"] < t_end)]
 
    t_row = trial_df.iloc[trial_idx]
    audio_id = int(str(t_row["audio_file"]).split("_")[0])
    condition = str(t_row["condition"])
    target_position = int(t_row["target_position"])
    sentence = str(t_row.get("sentence", ""))
    key = (audio_id, condition)
    if key not in ann_lookup:
        return None
    verb_off, target_on = ann_lookup[key]
 
    audio_onset_vals = seg_events.loc[seg_events.USER == "AUDIO_FILE_ONSET_LOG", "TIME"].values
    if len(audio_onset_vals) == 0:
        return None
    audio_onset = audio_onset_vals[0]
    crit_start, crit_end = audio_onset + verb_off, audio_onset + target_on
 
    valid = seg[seg["BPOGV"] == 1].copy()
    valid["rel_t"] = valid["TIME"] - audio_onset
    valid["screen_x"] = valid["BPOGX"] * SCREEN_W
    valid["screen_y"] = valid["BPOGY"] * SCREEN_H
 
    target_box = compute_aoi_boxes_img_px()[f"pos{target_position}"]
    tx1, ty1, tx2, ty2 = _img_box_to_norm(*target_box)
    target_screen_x = (tx1 + tx2) / 2 * SCREEN_W
    target_screen_y = (ty1 + ty2) / 2 * SCREEN_H
 
    return {
        "trial": trial_idx, "sentence": sentence, "condition": condition,
        "target_position": target_position, "target_screen_x": target_screen_x,
        "target_screen_y": target_screen_y, "crit_start_rel": crit_start - audio_onset,
        "crit_end_rel": crit_end - audio_onset, "trace": valid[["rel_t", "screen_x", "screen_y"]],
    }
 
 
# ──────────────────────────────────────────────────────────────
# PER-SUBJECT PROCESSING
# ──────────────────────────────────────────────────────────────
 
def segment_trials_simple(gaze_df):
    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]]
    starts = events[events.USER == "START_TRIAL"]["TIME"].values
    segs = []
    for i in range(len(starts)):
        t_end = starts[i + 1] if i + 1 < len(starts) else gaze_df["TIME"].max()
        segs.append(gaze_df[(gaze_df["TIME"] >= starts[i]) & (gaze_df["TIME"] < t_end)])
    return segs
 
 
def process_subject(subj, ann_df, ann_lookup):
    subject_id = subj["subject_id"]
    print(f"\n{'='*70}\nSubject: {subject_id}\n{'='*70}")
 
    gaze_df = pd.read_csv(subj["tsv"], sep="\t", low_memory=False)
    gaze_df["USER"] = gaze_df["USER"].fillna("0").astype(str).str.strip()
    trial_df = pd.read_csv(subj["csv"], low_memory=False)
    trial_df = trial_df[trial_df["audio_file"].notna() & (trial_df["audio_file"].astype(str) != "undefined")]
    trial_df = trial_df.sort_values("count_trial_loop").reset_index(drop=True)
    log_text = subj["log"].read_text() if subj["log"] else ""
 
    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    segments = segment_trials_simple(gaze_df)
    n_trials = len(trial_df)
    print(f"  {len(gaze_df)} gaze samples, {n_trials} trials")
 
    res_row = check_resolution(subject_id, gaze_df, log_text)
    valid_row, per_trial_valid = check_validity(subject_id, gaze_df, segments)
    time_row = check_time_units(subject_id, gaze_df, ann_df, events, log_text)
    marker_row = check_event_markers(subject_id, events, n_trials)
    cond_row = check_condition_balance(subject_id, trial_df)
 
    group_df = None
    group_consistency_row = None
    if subj["group_csv"] is not None and subj["group_csv"].exists():
        group_df = pd.read_csv(subj["group_csv"]).set_index("id")
        print(f"  matched group CSV: {subj['group_csv'].name}")
        group_consistency_row = check_group_consistency(subject_id, trial_df, group_df)
        if group_consistency_row["n_condition_mismatch"] or group_consistency_row["n_target_position_mismatch"]:
            print(f"  [warn] group consistency: {group_consistency_row['n_condition_mismatch']} condition "
                  f"mismatch(es), {group_consistency_row['n_target_position_mismatch']} target_position "
                  f"mismatch(es) out of {group_consistency_row['n_trials_checked']} trials checked")
    else:
        print(f"  [warn] no group CSV found for {subject_id} (looked for "
              f"participant_group{subject_id_to_group_number(subject_id)}.csv in {GROUP_CSV_DIR}) — "
              f"skipping group-consistency cross-check for this subject")
 
    gaze_acc_df = compute_gaze_accuracy(subject_id, gaze_df, trial_df, ann_lookup)
    aoi_classified_df = compute_aoi_classified_samples(gaze_df, trial_df, ann_lookup)
    density_windows = compute_subject_onset_density(gaze_df, trial_df)
 
    example_trial_idx = select_example_trial(subject_id, trial_df, gaze_acc_df, condition="restrictive")
    example_trace = extract_trial_trace(gaze_df, trial_df, ann_lookup, example_trial_idx) \
        if example_trial_idx is not None else None
 
    return {
        "resolution": res_row, "validity": valid_row, "per_trial_valid": per_trial_valid,
        "time": time_row, "markers": marker_row, "condition_balance": cond_row,
        "gaze_accuracy": gaze_acc_df, "log_text": log_text, "gaze_df": gaze_df,
        "group_consistency": group_consistency_row,
        "aoi_classified": aoi_classified_df, "density_windows": density_windows,
        "example_trace": example_trace,
    }
 
 
# ──────────────────────────────────────────────────────────────
# CROSS-SUBJECT VISUALIZATIONS
# ──────────────────────────────────────────────────────────────
 
def plot_calibration_across_subjects(log_texts, out_path):
    rows = []
    for subject_id, text in log_texts.items():
        m = re.search(r"accuracy \(degrees\):\s*LX=([\d.]+),\s*LY=([\d.]+),\s*RX=([\d.]+),\s*RY=([\d.]+)", text or "")
        if m:
            rows.append({"subject_id": subject_id, "LX": float(m.group(1)), "LY": float(m.group(2)),
                         "RX": float(m.group(3)), "RY": float(m.group(4))})
    if not rows:
        print("  [plot] skipped calibration comparison — no parsable calibration reports")
        return
    df = pd.DataFrame(rows)
    axes_cols = ["LX", "LY", "RX", "RY"]
    x = np.arange(len(df)); width = 0.8 / 4
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(df) + 2), 4.5))
    colors = ["#1f77b4", "#4fa3d1", "#d62728", "#e8807f"]
    for i, col in enumerate(axes_cols):
        ax.bar(x + i * width - (0.8 - width) / 2, df[col], width=width, label=col, color=colors[i])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="1° threshold")
    ax.set_xticks(x); ax.set_xticklabels(df["subject_id"], fontsize=9)
    ax.set_ylabel("degrees"); ax.set_title("Calibration accuracy across subjects")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_validity_across_subjects(valid_rows, out_path):
    df = pd.DataFrame(valid_rows).sort_values("pct_valid_overall")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df) + 2), 4.5))
    ax.bar(df["subject_id"], df["pct_valid_overall"], color="#1f77b4")
    ax.axhline(90, color="black", linestyle="--", linewidth=1, alpha=0.6, label="90% threshold")
    ax.set_ylim(0, 105); ax.set_ylabel("% valid samples")
    ax.set_title("Valid data captured across subjects")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_gaze_accuracy_across_subjects(all_gaze_acc_df, out_path):
    if all_gaze_acc_df.empty:
        print("  [plot] skipped gaze-accuracy comparison — no data")
        return
    df = all_gaze_acc_df[~all_gaze_acc_df["no_object_gaze"]]
    if df.empty:
        print("  [plot] skipped gaze-accuracy comparison — no trials with object gaze")
        return
    subjects = sorted(df["subject_id"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(subjects) + 2), 4.5))
    data = [df.loc[df["subject_id"] == s, "prop_target"].values for s in subjects]
    ax.boxplot(data, positions=range(len(subjects)), widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#1f77b4", alpha=0.4), showfliers=False)
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=15, alpha=0.6, color="#1f77b4")
    ax.axhline(0.25, color="black", linestyle=":", alpha=0.5, label="chance (0.25)")
    ax.set_xticks(range(len(subjects))); ax.set_xticklabels(subjects, fontsize=9)
    ax.set_ylabel("prop_target (duration-weighted)")
    ax.set_title("Gaze-on-target probability across subjects\n(critical window, object-gaze trials only)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_condition_balance(cond_rows, out_path):
    df = pd.DataFrame(cond_rows).set_index("subject_id")
    cond_cols = [c for c in df.columns if c.startswith("n_restrictive") or c.startswith("n_non-restrictive")]
    if not cond_cols:
        print("  [plot] skipped condition balance — no condition columns found")
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df) + 2), 4.5))
    bottom = np.zeros(len(df))
    for col, color in zip(cond_cols, ["#1f77b4", "#d62728"]):
        vals = df[col].fillna(0).values
        ax.bar(df.index, vals, bottom=bottom, label=col.replace("n_", ""), color=color)
        bottom += vals
    ax.set_ylabel("Trial count"); ax.set_title("Restrictive vs. non-restrictive trials per subject")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_gaze_bias_per_subject(gaze_dfs, out_path):
    """
    Milestone-4 check ③: per-subject BPOGX histograms (small multiples) +
    a combined mean +/- SD bar chart against the ideal center (0.5).
    """
    subjects = list(gaze_dfs.keys())
    fig, axes = plt.subplots(1, len(subjects) + 1, figsize=(4.2 * (len(subjects) + 1), 4.2))
    if len(subjects) == 1:
        axes = [axes[0], axes[1]] if hasattr(axes, "__len__") else [axes]
 
    means, stds = [], []
    for i, subject_id in enumerate(subjects):
        valid = gaze_dfs[subject_id][gaze_dfs[subject_id]["BPOGV"] == 1]
        x = valid["BPOGX"]
        means.append(x.mean()); stds.append(x.std())
        ax = axes[i]
        ax.hist(x, bins=40, alpha=0.7, color=f"C{i}")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="ideal centre (0.5)")
        ax.axvline(x.mean(), color="black", linestyle="-", linewidth=1, label=f"mean={x.mean():.3f}")
        ax.set_title(f"{subject_id} — BPOGX distribution"); ax.set_xlabel("Normalized X (BvOGX-left/right)")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
 
    ax = axes[-1]
    ax.bar(subjects, means, yerr=stds, capsize=5, color=[f"C{i}" for i in range(len(subjects))])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="ideal centre (0.5)")
    ax.set_title("Mean BPOGX \u00b1 SD"); ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
 
    fig.suptitle("Horizontal Gaze Bias Check (BPOGX, valid samples only)", y=1.02)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_invalid_rate_per_trial(per_trial_valid_dict, out_path, threshold_pct=20.0):
    """
    Milestone-4 check ②: per-trial INVALID rate (100 - valid%) as small
    multiples, one panel per subject, with a threshold line.
    """
    subjects = list(per_trial_valid_dict.keys())
    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 4), squeeze=False)
    axes = axes[0]
    for i, subject_id in enumerate(subjects):
        invalid_pct = 100 - per_trial_valid_dict[subject_id]
        ax = axes[i]
        ax.bar(range(len(invalid_pct)), invalid_pct, color=f"C{i}")
        ax.axhline(threshold_pct, color="red", linestyle="--", linewidth=1, label=f"{threshold_pct:.0f}% threshold")
        mean_invalid = invalid_pct.mean()
        ax.set_title(f"{subject_id}: per-trial invalid sample rate\nmean invalid = {mean_invalid:.1f}%")
        ax.set_xlabel("Trial"); ax.set_ylabel("Invalid rate (%)")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_gaze_on_aoi(aoi_classified_dfs, out_path):
    """
    Milestone-4 check ④: does gaze land where the objects actually are?
    Scatter of critical-window valid samples over the (deterministic) AOI
    boxes, one panel per subject.
    """
    subjects = list(aoi_classified_dfs.keys())
    fig, axes = plt.subplots(1, len(subjects), figsize=(6.5 * len(subjects), 4.2), squeeze=False)
    axes = axes[0]
    aoi_colors = {"pos1": "#1f77b4", "pos2": "#ff7f0e", "pos3": "#2ca02c",
                  "pos4": "#d62728", "subject": "#7f7f7f", "elsewhere": "#cccccc"}
    for i, subject_id in enumerate(subjects):
        df = aoi_classified_dfs[subject_id]
        ax = axes[i]
        for name, (x1, y1, x2, y2) in DEFAULT_AOI_NORM.items():
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                             edgecolor=aoi_colors.get(name, "black"), facecolor="none"))
            ax.text(x1, y1 - 0.02, name, fontsize=8, color=aoi_colors.get(name, "black"))
        if not df.empty:
            for aoi, sub in df.groupby("aoi"):
                ax.scatter(sub["BPOGX"], sub["BPOGY"], s=8, alpha=0.5,
                          color=aoi_colors.get(aoi, "black"), label=aoi)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        ax.set_title(f"{subject_id}: gaze vs. AOI positions\n(critical window, valid samples, n={len(df)})")
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_subject_onset_density(subject_id, density_windows, out_path):
    """
    Milestone-4 extra: gaze density in 4 time windows relative to
    SUBJECT_ONSET_LOG, pooled across all trials for one subject — shows
    whether gaze converges to the subject figure right after it appears.
    """
    fig, axes = plt.subplots(1, len(SUBJECT_ONSET_WINDOWS), figsize=(4 * len(SUBJECT_ONSET_WINDOWS), 4))
    for ax, (label, lo, hi) in zip(axes, SUBJECT_ONSET_WINDOWS):
        df = density_windows[label]
        n = len(df)
        if n > 10:
            ax.hist2d(df["BPOGX"], df["BPOGY"], bins=30, range=[[0, 1], [0, 1]], cmap="YlOrRd")
        else:
            ax.scatter(df["BPOGX"], df["BPOGY"], s=5, alpha=0.5, color="red")
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        ax.set_title(f"{label}\n(n={n} samples)", fontsize=9)
        ax.set_xlabel("BPOGX (left/right)")
    axes[0].set_ylabel("BPOGY (top/bottom)")
    fig.suptitle(f"Gaze density around subject onset — {subject_id}, all trials", y=1.03)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def plot_example_trial_trace(subject_id, example_trace, out_path):
    """
    Milestone-4 extra: for one example trial, Gaze X/Y in screen pixels
    over time relative to audio onset, with the target's screen position
    and the critical window marked — shows the "gaze locks onto the
    target after the verb" pattern.
    """
    if example_trace is None:
        print(f"  [plot] skipped example trial trace for {subject_id} — no suitable trial found")
        return
    trace = example_trace["trace"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trace["rel_t"], trace["screen_x"], color="#1f77b4", label="Gaze X", linewidth=1)
    ax.plot(trace["rel_t"], trace["screen_y"], color="#d62728", label="Gaze Y", linewidth=1)
    ax.axhline(example_trace["target_screen_x"], color="#1f77b4", linestyle="--", alpha=0.6,
               label=f"Target X ({example_trace['target_screen_x']:.0f}px)")
    ax.axhline(example_trace["target_screen_y"], color="#d62728", linestyle="--", alpha=0.6,
               label=f"Target Y ({example_trace['target_screen_y']:.0f}px)")
    ax.axvspan(example_trace["crit_start_rel"], example_trace["crit_end_rel"],
               color="#2ca02c", alpha=0.2, label="critical window")
    ax.set_xlabel("Time relative to audio onset (s)"); ax.set_ylabel("Screen position (pixels)")
    ax.set_title(f'{subject_id}: "{example_trace["sentence"]}" — {example_trace["condition"]}, '
                 f'target at position {example_trace["target_position"]}')
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
# ──────────────────────────────────────────────────────────────
# PUBLICATION-QUALITY FIGURES
# ──────────────────────────────────────────────────────────────
 
def fig_participant_validity(validity_df, out_path, thresholds=(80, 90)):
    """1. % valid gaze per participant, sorted, with threshold line(s)."""
    df = validity_df.sort_values("pct_valid_overall").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.75 * len(df) + 2), 4.5))
    bars = ax.bar(df["subject_id"], df["pct_valid_overall"], color=PUB_COLORS["primary"], width=0.6, zorder=3)
    _bar_value_labels(ax, bars)
    for i, t in enumerate(thresholds):
        ax.axhline(t, color=PUB_COLORS["quaternary"] if i == 0 else "#888888",
                    linestyle="--", linewidth=1.1, label=f"{t}% threshold", zorder=2)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Valid gaze samples (%)")
    ax.set_xlabel("Participant")
    ax.set_title("Participant Data Validity")
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def fig_condition_balance(condition_df, out_path):
    """2. Restrictive vs. non-restrictive trial counts, and target-position balance, per participant."""
    df = condition_df.set_index("subject_id")
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.1 * len(df) + 4), 4.5))
 
    cond_cols = [c for c in df.columns if c.startswith("n_restrictive") or c.startswith("n_non-restrictive")]
    x = np.arange(len(df)); width = 0.35
    for i, col in enumerate(cond_cols):
        bars = axes[0].bar(x + (i - 0.5) * width, df[col].fillna(0), width,
                           label=col.replace("n_", "").replace("-", "-").capitalize(),
                           color=PUB_PALETTE[i], zorder=3)
        _bar_value_labels(axes[0], bars, fmt="{:.0f}", offset=0.3, fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(df.index, rotation=30, ha="right")
    axes[0].set_ylabel("Trial count"); axes[0].set_title("A. Condition Balance")
    axes[0].legend()
 
    pos_cols = [c for c in df.columns if c.startswith("n_target_pos")]
    width2 = 0.8 / len(pos_cols)
    for i, col in enumerate(pos_cols):
        label = col.replace("n_target_pos", "Position ")
        axes[1].bar(x + i * width2 - (0.8 - width2) / 2, df[col].fillna(0), width2,
                    label=label, color=PUB_PALETTE[i % len(PUB_PALETTE)], zorder=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df.index, rotation=30, ha="right")
    axes[1].set_ylabel("Trial count"); axes[1].set_title("B. Target-Position Balance")
    axes[1].legend(ncol=2, fontsize=8)
 
    fig.suptitle("Condition and Target-Position Balance per Participant", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def fig_sampling_frequency(time_df, out_path):
    """3. Logged vs. measured sampling rate, and the % difference, per participant."""
    df = time_df.copy()
    df["diff_pct"] = 100 * (df["measured_hz"] - df["logged_hz"]) / df["logged_hz"]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.1 * len(df) + 4), 4.5))
 
    x = np.arange(len(df)); width = 0.35
    b1 = axes[0].bar(x - width / 2, df["logged_hz"], width, label="Logged (Hz)", color=PUB_COLORS["primary"], zorder=3)
    b2 = axes[0].bar(x + width / 2, df["measured_hz"], width, label="Measured (Hz)", color=PUB_COLORS["secondary"], zorder=3)
    _bar_value_labels(axes[0], b1, fmt="{:.0f}"); _bar_value_labels(axes[0], b2, fmt="{:.0f}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["subject_id"], rotation=30, ha="right")
    axes[0].set_ylabel("Sampling rate (Hz)"); axes[0].set_title("A. Logged vs. Measured Samplerate")
    axes[0].legend()
 
    colors = [PUB_COLORS["quaternary"] if abs(v) > 15 else PUB_COLORS["tertiary"] for v in df["diff_pct"]]
    bars = axes[1].bar(df["subject_id"], df["diff_pct"], color=colors, zorder=3)
    _bar_value_labels(axes[1], bars, fmt="{:+.1f}%")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["subject_id"], rotation=30, ha="right")
    axes[1].set_ylabel("Difference from logged rate (%)"); axes[1].set_title("B. Samplerate Discrepancy")
 
    fig.suptitle("Eye-Tracker Sampling Frequency: Logged vs. Actual", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def fig_event_completeness(marker_df, out_path):
    """4. % of trials with each required event marker present, per participant."""
    events = [("pct_START_TRIAL_complete", "START_TRIAL"),
              ("pct_AUDIO_FILE_ONSET_LOG_complete", "AUDIO_ONSET"),
              ("pct_AUDIO_FILE_OFFSET_complete", "AUDIO_OFFSET"),
              ("pct_STOP_TRIAL_complete", "STOP_TRIAL")]
    df = marker_df.set_index("subject_id")
    x = np.arange(len(df)); width = 0.8 / len(events)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(df) + 3), 4.8))
    for i, (col, label) in enumerate(events):
        if col not in df.columns:
            continue
        ax.bar(x + i * width - (0.8 - width) / 2, df[col], width, label=label,
               color=PUB_PALETTE[i % len(PUB_PALETTE)], zorder=3)
    ax.axhline(100, color="#888888", linestyle=":", linewidth=1, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Trials with marker present (%)")
    ax.set_title("Event-Marker Completeness per Participant")
    ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def fig_resolution_check(resolution_df, out_path):
    """5. % of BPOGX/BPOGY values outside the valid [0,1] screen range, per participant."""
    df = resolution_df.set_index("subject_id")
    x = np.arange(len(df)); width = 0.35
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(df) + 2), 4.5))
    b1 = ax.bar(x - width / 2, df["pct_x_outside_01"], width, label="X outside [0,1]", color=PUB_COLORS["primary"], zorder=3)
    b2 = ax.bar(x + width / 2, df["pct_y_outside_01"], width, label="Y outside [0,1]", color=PUB_COLORS["secondary"], zorder=3)
    _bar_value_labels(ax, b1, fmt="{:.1f}%"); _bar_value_labels(ax, b2, fmt="{:.1f}%")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylabel("Valid samples outside [0,1] (%)")
    ax.set_title("Coordinate-Space Resolution Check")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
def fig_gaze_accuracy(gaze_acc_df, out_path):
    """6. Target-advantage distribution, target-vs-distractor gaze, no-object-gaze rate,
    and a per-subject boxplot — the fullest picture of anticipatory-gaze accuracy."""
    if gaze_acc_df.empty:
        print(f"  [plot] skipped {out_path} — no gaze accuracy data")
        return
    df = gaze_acc_df.copy()
    subjects = sorted(df["subject_id"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
 
    # A. Target-advantage distribution (pooled, object-gaze trials only)
    ax = axes[0, 0]
    valid = df[~df["no_object_gaze"]]
    ax.hist(valid["target_advantage"], bins=25, color=PUB_COLORS["primary"], alpha=0.85, zorder=3)
    ax.axvline(0, color=PUB_COLORS["quaternary"], linestyle="--", linewidth=1.2, label="no advantage (0)")
    ax.set_xlabel("Target advantage  (P(target) \u2212 mean P(distractor))")
    ax.set_ylabel("Trial count"); ax.set_title("A. Target-Advantage Distribution")
    ax.legend()
 
    # B. Target vs. distractor gaze, mean per subject
    ax = axes[0, 1]
    means = valid.groupby("subject_id")[["prop_target", "prop_distractor_mean"]].mean().reindex(subjects)
    x = np.arange(len(subjects)); width = 0.35
    b1 = ax.bar(x - width / 2, means["prop_target"], width, label="Target", color=PUB_COLORS["tertiary"], zorder=3)
    b2 = ax.bar(x + width / 2, means["prop_distractor_mean"], width, label="Distractor (mean)", color=PUB_COLORS["quaternary"], zorder=3)
    _bar_value_labels(ax, b1, fmt="{:.2f}", offset=0.01); _bar_value_labels(ax, b2, fmt="{:.2f}", offset=0.01)
    ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1, label="chance (0.25)")
    ax.set_xticks(x); ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("Mean gaze probability"); ax.set_title("B. Target vs. Distractor Gaze")
    ax.legend(fontsize=8)
 
    # C. No-object-gaze rate per subject
    ax = axes[1, 0]
    no_obj_pct = df.groupby("subject_id")["no_object_gaze"].mean().reindex(subjects) * 100
    bars = ax.bar(subjects, no_obj_pct, color=PUB_COLORS["quinary"], zorder=3)
    _bar_value_labels(ax, bars, fmt="{:.1f}%")
    ax.set_xticks(range(len(subjects))); ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("Trials with no object gaze (%)"); ax.set_title("C. No-Object-Gaze Rate")
 
    # D. Per-subject boxplot of target_advantage
    ax = axes[1, 1]
    data = [valid.loc[valid["subject_id"] == s, "target_advantage"].values for s in subjects]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(bp["boxes"], PUB_PALETTE * (len(subjects) // len(PUB_PALETTE) + 1)):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.6, color=PUB_COLORS["primary"], zorder=4)
    ax.axhline(0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(range(1, len(subjects) + 1)); ax.set_xticklabels(subjects, rotation=30, ha="right")
    ax.set_ylabel("Target advantage"); ax.set_title("D. Per-Subject Target-Advantage Distribution")
 
    fig.suptitle("Gaze Accuracy: Anticipatory Looking to the Target Object", y=1.01, fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    print(f"  [plot] saved: {out_path}")
 
 
# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
 
def main():
    if not ANNOTATION_CSV.exists():
        raise SystemExit(f"ERROR: annotation CSV not found at {ANNOTATION_CSV}")
    ann_df = pd.read_csv(ANNOTATION_CSV)
    ann_lookup = build_ann_lookup(ann_df)
 
    subjects = discover_subjects(INPUT_DIR)
    if not subjects:
        raise SystemExit(f"ERROR: no {{subject}}.tsv + {{subject}}.csv pairs found in {INPUT_DIR}")
    print(f"Found {len(subjects)} subject(s): {', '.join(s['subject_id'] for s in subjects)}")
 
    results = {}
    for subj in subjects:
        try:
            results[subj["subject_id"]] = process_subject(subj, ann_df, ann_lookup)
        except Exception as e:
            print(f"  [subject FAILED] {subj['subject_id']}: {e}")
 
    if not results:
        raise SystemExit("ERROR: no subjects processed successfully.")
 
    # ── aggregate ─────────────────────────────────────────────
    resolution_df = pd.DataFrame([r["resolution"] for r in results.values()])
    validity_df = pd.DataFrame([r["validity"] for r in results.values()])
    time_df = pd.DataFrame([r["time"] for r in results.values()])
    marker_df = pd.DataFrame([r["markers"] for r in results.values()])
    condition_df = pd.DataFrame([r["condition_balance"] for r in results.values()])
    gaze_acc_df = pd.concat([r["gaze_accuracy"] for r in results.values()], ignore_index=True) \
        if any(len(r["gaze_accuracy"]) for r in results.values()) else pd.DataFrame()
    log_texts = {sid: r["log_text"] for sid, r in results.items()}
    gaze_dfs = {sid: r["gaze_df"] for sid, r in results.items()}
 
    # ── print summary tables ─────────────────────────────────
    pd.set_option("display.width", 160)
    print("\n" + "=" * 90 + "\nRESOLUTION / COORDINATE-SPACE CHECK (per subject)\n" + "=" * 90)
    print(resolution_df.to_string(index=False))
 
    print("\n" + "=" * 90 + "\nVALID SAMPLE RATE (per subject)\n" + "=" * 90)
    print(validity_df.to_string(index=False))
 
    print("\n" + "=" * 90 + "\nTIME UNIT / SAMPLING FREQUENCY (per subject)\n" + "=" * 90)
    print(time_df.to_string(index=False))
    n_mismatch = time_df["samplerate_mismatch"].sum()
    if n_mismatch:
        print(f"  >>> {n_mismatch}/{len(time_df)} subject(s) show a samplerate mismatch (log vs. measured)")
 
    print("\n" + "=" * 90 + "\nEVENT MARKER COMPLETENESS (per subject)\n" + "=" * 90)
    print(marker_df.to_string(index=False))
 
    print("\n" + "=" * 90 + "\nCONDITION / TARGET-POSITION BALANCE (per subject)\n" + "=" * 90)
    print(condition_df.to_string(index=False))
 
    group_consistency_rows = [r["group_consistency"] for r in results.values() if r["group_consistency"] is not None]
    if group_consistency_rows:
        gc_df = pd.DataFrame(group_consistency_rows)
        print("\n" + "=" * 90 + "\nGROUP-CSV CONSISTENCY CHECK (per subject, where a group CSV was matched)\n" + "=" * 90)
        print(gc_df.to_string(index=False))
        total_mismatch = gc_df["n_condition_mismatch"].sum() + gc_df["n_target_position_mismatch"].sum()
        if total_mismatch:
            print(f"  >>> {total_mismatch} total mismatch(es) between subjects' actual trial data "
                  f"and their matched group design — worth investigating.")
 
    if not gaze_acc_df.empty:
        print("\n" + "=" * 90 + "\nGAZE-ON-TARGET PROBABILITY (mean by subject x condition)\n" + "=" * 90)
        summary = (gaze_acc_df[~gaze_acc_df["no_object_gaze"]]
                   .groupby(["subject_id", "condition"])[["prop_target", "target_advantage"]]
                   .mean().round(3))
        print(summary.to_string())
 
    # ── save CSVs ─────────────────────────────────────────────
    resolution_df.to_csv(CSV_DIR / "qc_resolution.csv", index=False)
    validity_df.to_csv(CSV_DIR / "qc_validity.csv", index=False)
    time_df.to_csv(CSV_DIR / "qc_time_units.csv", index=False)
    marker_df.to_csv(CSV_DIR / "qc_event_markers.csv", index=False)
    condition_df.to_csv(CSV_DIR / "qc_condition_balance.csv", index=False)
    if not gaze_acc_df.empty:
        gaze_acc_df.to_csv(CSV_DIR / "qc_gaze_accuracy.csv", index=False)
    if group_consistency_rows:
        gc_df.to_csv(CSV_DIR / "qc_group_consistency.csv", index=False)
    per_trial_valid_df = pd.DataFrame({sid: r["per_trial_valid"] for sid, r in results.items()})
    per_trial_valid_df.to_csv(CSV_DIR / "qc_per_trial_valid_pct.csv", index_label="trial")
    print(f"\nSaved QC CSVs to {CSV_DIR}")
 
    # ── plots ─────────────────────────────────────────────────
    print("\n" + "=" * 90 + "\nVISUALIZATIONS\n" + "=" * 90)
    per_trial_valid_dict = {sid: r["per_trial_valid"] for sid, r in results.items()}
    aoi_classified_dfs = {sid: r["aoi_classified"] for sid, r in results.items()}
 
    plot_calibration_across_subjects(log_texts, PLOTS_DIR / "1_calibration_across_subjects.png")
    plot_invalid_rate_per_trial(per_trial_valid_dict, PLOTS_DIR / "2_invalid_rate_per_trial.png")
    plot_gaze_bias_per_subject(gaze_dfs, PLOTS_DIR / "3_gaze_bias_across_subjects.png")
    plot_gaze_on_aoi(aoi_classified_dfs, PLOTS_DIR / "4_gaze_on_aoi.png")
    plot_validity_across_subjects([r["validity"] for r in results.values()], PLOTS_DIR / "validity_across_subjects.png")
    plot_condition_balance([r["condition_balance"] for r in results.values()], PLOTS_DIR / "condition_balance.png")
    plot_gaze_accuracy_across_subjects(gaze_acc_df, PLOTS_DIR / "gaze_accuracy_across_subjects.png")
 
    print("\nPublication-quality figures:")
    fig_participant_validity(validity_df, PLOTS_DIR / "fig1_participant_validity.png")
    fig_condition_balance(condition_df, PLOTS_DIR / "fig2_condition_balance.png")
    fig_sampling_frequency(time_df, PLOTS_DIR / "fig3_sampling_frequency.png")
    fig_event_completeness(marker_df, PLOTS_DIR / "fig4_event_completeness.png")
    fig_resolution_check(resolution_df, PLOTS_DIR / "fig5_resolution_check.png")
    fig_gaze_accuracy(gaze_acc_df, PLOTS_DIR / "fig6_gaze_accuracy.png")
 
    for sid, r in results.items():
        plot_subject_onset_density(sid, r["density_windows"], PLOTS_DIR / f"5_subject_onset_density_{sid}.png")
        plot_example_trial_trace(sid, r["example_trace"], PLOTS_DIR / f"6_example_trial_trace_{sid}.png")
 
    print(f"\nSaved plots to {PLOTS_DIR}")
    return results
 
 
if __name__ == "__main__":
    main()
 


















