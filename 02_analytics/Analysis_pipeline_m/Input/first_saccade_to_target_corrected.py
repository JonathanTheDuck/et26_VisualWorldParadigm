import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# saccade_analysis_ek.py and corrected_aoi.py live in the Input/ folder,
# same convention used throughout this project (scripts/support modules
# alongside the data, resolved relative to this script's own location so
# it works straight after cloning -- no machine-specific path needed).
sys.path.insert(0, str(HERE / "Input"))

import saccade_analysis_ek as sa  # Engbert & Kliegl (2003) detector module
from corrected_aoi import classify_aoi_tight, classify_aoi_pieslice
import pandas as pd
import numpy as np

# Full multi-subject dataset lives in its own sibling folder (distinct from
# Input/, which holds the support scripts + the original demo subject).
INPUT_DIR = HERE
ANNOTATION_CSV = INPUT_DIR / "annotation_audiov2.csv"

# Output goes to a dedicated csv/ folder, matching the OUT_DIR/CSV_DIR
# convention used in quality_control.py and saccade_analysis.py.
OUT_DIR = HERE / "saccade_output" / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "first_saccade_to_target_corrected_ek.csv"

ann_df = pd.read_csv(ANNOTATION_CSV)
ann_lookup = sa.build_ann_lookup_full(ann_df)
subjects = sa.discover_subjects(INPUT_DIR)
print(f"Found {len(subjects)} subjects")
print(f"Using Engbert & Kliegl (2003) detector: lambda={sa.EK_LAMBDA}, min_samples={sa.EK_MIN_SAMPLES}")

rows = []
for subj in subjects:
    subject_id = subj["subject_id"]
    try:
        gaze_df = pd.read_csv(subj["tsv"], sep="\t", low_memory=False)
        gaze_df["USER"] = gaze_df["USER"].fillna("0").astype(str).str.strip()
        trial_df = pd.read_csv(subj["csv"], low_memory=False)
        trial_df = trial_df[trial_df["audio_file"].notna() & (trial_df["audio_file"].astype(str) != "undefined")]
        trial_df = trial_df.sort_values("count_trial_loop").reset_index(drop=True)

        log_text = subj["log"].read_text() if subj["log"] else ""
        distance_cm, screen_w_cm, screen_h_cm = sa._load_session_geometry(log_text)
        gaze_df, blink_events = sa.detect_and_mask_blinks(gaze_df)

        events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
        starts = events[events.USER == "START_TRIAL"]["TIME"].values

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
            target_key = f"pos{target_position}"

            saccades = sa.detect_saccades_ek(seg, distance_cm, screen_w_cm, screen_h_cm)  # <-- E&K detector
            candidates = sorted(
                [s for s in saccades if verb_onset_abs <= s["onset_time"] < audio_offset],
                key=lambda s: s["onset_time"]
            )

            row = {"subject_id": subject_id, "trial": i, "sentence_id": audio_id,
                   "condition": condition, "target_position": target_position}

            for scheme_name, classify_fn in [("tight", classify_aoi_tight), ("pieslice", classify_aoi_pieslice)]:
                first_to_target = next(
                    (s for s in candidates if classify_fn(s["end_x"], s["end_y"]) == target_key), None)
                if first_to_target is not None:
                    row[f"found_{scheme_name}"] = True
                    row[f"before_noun_{scheme_name}"] = first_to_target["onset_time"] < noun_onset_abs
                else:
                    row[f"found_{scheme_name}"] = False
                    row[f"before_noun_{scheme_name}"] = False
            rows.append(row)
    except Exception as e:
        print(f"  [FAILED] {subject_id}: {e}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} trial rows to {OUT_CSV}")

for scheme in ["tight", "pieslice"]:
    print(f"\n=== scheme: {scheme} ===")
    print(f"found target-landing saccade (any time post verb-onset): {df[f'found_{scheme}'].mean():.1%}")
    print(f"  by condition:")
    print(df.groupby("condition")[f"found_{scheme}"].mean().round(3))
    print(f"before noun onset (% of ALL trials): {df[f'before_noun_{scheme}'].mean():.1%}")
    print(f"  by condition:")
    print(df.groupby("condition")[f"before_noun_{scheme}"].mean().round(3))