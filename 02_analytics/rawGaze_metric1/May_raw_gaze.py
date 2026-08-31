import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import pandas as pd

# pls run this script from within the dir human-predictions ^^

# ── constants (hoisted out of the participant loop - they never change) ────
N_PARTICIPANTS = 24
N_TRIALS = 50

SCREEN_W, SCREEN_H = 2560, 1440  # OpenSesame virtual canvas
IMG_W, IMG_H = 2560, 1440
SCALE = 1  # explicitly set in OpenSesame sketchpad
IMG_PATH = "../../01_experiment/stimuli/img_composition/1_pos1_nosub.png"

POSITIONS = {
    "pos1": (1891, 429),
    "pos2": (1423, 915),
    "pos3": (697, 915),
    "pos4": (229, 429),
}
# not squares - resolution/scaling should be handled more precisely next time
BOX_WIDTH = 440
BOX_HEIGHT = 440

# sanity-check plot only needs to run once, not once per participant
RUN_SANITY_CHECK = True
SANITY_CHECK_PARTICIPANT = 1


def assign_aoi(df, positions, box_width, box_height, x_col="screen_x", y_col="screen_y"):
    df = df.copy()
    df["AOI"] = None

    for name, (x0, y0) in positions.items():
        mask = (
            (df[x_col] >= x0) &
            (df[x_col] <= x0 + box_width) &
            (df[y_col] >= y0) &
            (df[y_col] <= y0 + box_height)
        )
        df.loc[mask, "AOI"] = name

    return df.dropna(subset=["AOI"]).reset_index(drop=True)


def plot_sanity_check(valid_samples):
    """Overlay valid gaze samples and AOI boxes on the stimulus canvas, saved to disk for inspection."""
    disp_w = IMG_W * SCALE
    disp_h = IMG_H * SCALE
    offset_x = (SCREEN_W - disp_w) / 2
    offset_y = (SCREEN_H - disp_h) / 2

    fig, ax = plt.subplots(figsize=(12, 6.75))  # matches 16:9 aspect
    ax.add_patch(patches.Rectangle((0, 0), SCREEN_W, SCREEN_H, color="black", zorder=0))

    img = np.array(Image.open(IMG_PATH))
    # extent y is [top, bottom] reversed because imshow origin is upper-left by default
    ax.imshow(img, extent=[offset_x, offset_x + disp_w, offset_y + disp_h, offset_y], zorder=1)
    ax.scatter(valid_samples["screen_x"], valid_samples["screen_y"], s=10, c="red", alpha=0.5, zorder=2, label="gaze samples")

    for _, (x, y) in POSITIONS.items():
        ax.add_patch(Rectangle((x, y), BOX_WIDTH, BOX_HEIGHT, linewidth=3, edgecolor="blue", facecolor="none"))

    ax.set_xlim(0, SCREEN_W)
    ax.set_ylim(SCREEN_H, 0)  # flipped so (0,0) is top-left, matching screen coords
    ax.set_aspect("equal")
    ax.set_title(f"{len(valid_samples)} valid gaze samples on screen ({SCREEN_W}x{SCREEN_H})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig("sanity_check_gaze_plot.png")
    plt.close(fig)  # avoid leaking one open figure per participant


partID = []
out_stimuliId = []
out_obj = []
is_target = []
out_count_per_obj_per_part = []
condition = []

for participantNumber in range(1, N_PARTICIPANTS + 1):
    path = "../../01_experiment/recordings/SUBJECTS (FINAL)/"
    part_df_tmp = pd.read_csv(path + "subject-" + str(participantNumber) + ".csv")
    gazeSamples_df = pd.read_table(path + "subject-" + str(participantNumber) + ".tsv")
    gazeSamples_df_reduced = gazeSamples_df[["TIME", "BPOGX", "BPOGY", "BPOGV", "USER"]]

    annotationPath = "../../01_experiment/stimuli/annotation_audiov2.csv"
    annotation_df = pd.read_csv(annotationPath)

    part_df = part_df_tmp[["id", "position1", "position2", "position3", "position4",
                            "count_audio_file_offset", "audio_file", "condition", "target_position"]]
    part_df = part_df.rename(columns={"count_audio_file_offset": "trialID", "id": "stimuliID"})

    # reconstruct trial ids from event markers: AUDIO_FILE_OFFSET is used as the end
    # marker because STOP_TRIAL wasn't always logged reliably
    is_start = gazeSamples_df_reduced["USER"].eq("START_TRIAL")
    is_stop = gazeSamples_df_reduced["USER"].eq("AUDIO_FILE_OFFSET")

    inside = False
    current_id = -1
    ids = []
    for start, stop in zip(is_start, is_stop):
        if start:
            inside = True
            current_id += 1
        ids.append(str(current_id) if inside else None)  # str for easier merging later
        if stop:
            inside = False

    gazeSamples_df_reduced["trialId"] = ids
    samples_onlyTrials = gazeSamples_df_reduced[gazeSamples_df_reduced["trialId"].notna()]

    if RUN_SANITY_CHECK and participantNumber == SANITY_CHECK_PARTICIPANT:
        valid = samples_onlyTrials[samples_onlyTrials["BPOGV"] == 1].copy()
        valid["screen_x"] = valid["BPOGX"] * SCREEN_W
        valid["screen_y"] = valid["BPOGY"] * SCREEN_H
        plot_sanity_check(valid)

    onsetsetsAudioFiles = samples_onlyTrials[samples_onlyTrials["USER"] == "AUDIO_FILE_ONSET_LOG"]
    onsetsetsAudioFiles = onsetsetsAudioFiles.rename(columns={"TIME": "audio_onset_time"})
    onsetsetsAudioFiles = onsetsetsAudioFiles[["audio_onset_time", "trialId"]]

    # renamed distinctly from part_df's own "condition" column to avoid confusing the two
    annotation_df = annotation_df.rename(columns={"id": "stimuliID", "SentenceRole": "annotation_condition"})

    for trialNr in range(0, N_TRIALS):
        trial_info = part_df[part_df["trialID"] == trialNr]
        stimuliID = trial_info["stimuliID"].values[0]
        condition_trial = trial_info["condition"].values[0]

        audioStart_trial = onsetsetsAudioFiles[
            onsetsetsAudioFiles["trialId"] == str(trialNr)
        ]["audio_onset_time"].values[0]

        trial_annotation_df = annotation_df[annotation_df["stimuliID"] == stimuliID]
        trial_annotation_df = trial_annotation_df[trial_annotation_df["annotation_condition"] == condition_trial]

        # AOI window = verb onset -> target-object-word onset (the anticipatory prediction window)
        try:
            verbOnset = trial_annotation_df[trial_annotation_df["WordRole"] == "ROOT"]["start"].values[0]
            # dobj is the usual role for the target object, but pobj occurs once too
            objectOnset = trial_annotation_df[
                (trial_annotation_df["WordRole"] == "dobj") | (trial_annotation_df["WordRole"] == "pobj")
            ]["start"].values[0]
        except IndexError:
            print(f"skipping trial {trialNr} for participant {participantNumber}: "
                  f"missing ROOT/dobj/pobj annotation for stimuliID {stimuliID}")
            continue

        start = audioStart_trial + verbOnset
        stop = audioStart_trial + objectOnset

        # restrict to this trial's time window AND its trialId, as a safety net against timing overlaps
        samples_restricted = samples_onlyTrials[
            (samples_onlyTrials["trialId"] == str(trialNr)) &
            (samples_onlyTrials["TIME"] > start) &
            (samples_onlyTrials["TIME"] < stop)
        ]

        valid_trial = samples_restricted[samples_restricted["BPOGV"] == 1].copy()
        valid_trial["screen_x"] = valid_trial["BPOGX"] * SCREEN_W
        valid_trial["screen_y"] = valid_trial["BPOGY"] * SCREEN_H

        mappedToPositions = assign_aoi(valid_trial, POSITIONS, box_width=BOX_WIDTH, box_height=BOX_HEIGHT,
                                        x_col="screen_x", y_col="screen_y")
        count_perAOI_df = mappedToPositions.groupby("AOI").size().reset_index(name="count")

        t_pos = trial_info["target_position"].values[0]
        for i, pos in enumerate(["pos1", "pos2", "pos3", "pos4"]):
            # downstream pipeline uses stimuli ids starting at 1
            out_stimuliId.append(stimuliID + 1)
            out_obj.append(trial_info["position" + str(i + 1)].values[0])
            is_target.append(1 if i + 1 == t_pos else 0)

            matching_count = count_perAOI_df[count_perAOI_df["AOI"] == pos]["count"]
            out_count_per_obj_per_part.append(matching_count.values[0] if not matching_count.empty else 0)

            partID.append(participantNumber)
            condition.append(condition_trial)


out_df_source = pd.DataFrame({
    "stimuliId": out_stimuliId,
    "partID": partID,
    "condition": condition,
    "obj": out_obj,
    "is_target": is_target,
    "count_per_obj_per_part": out_count_per_obj_per_part,
})

# count of participants that contributed any non-zero gaze data to a given stimuli/condition
count_intel_df = out_df_source.groupby(["stimuliId", "condition", "partID"]).agg(
    {"count_per_obj_per_part": "sum"}
).reset_index()
count_nonZeroParts_df = (
    count_intel_df[count_intel_df["count_per_obj_per_part"] > 0]
    .groupby(["stimuliId", "condition"])
    .size()
    .reset_index(name="countOfNonZeroParticipants")
)
print(count_nonZeroParts_df.to_string())

out_df = out_df_source.groupby(["stimuliId", "condition", "obj"]).agg(
    {"count_per_obj_per_part": "sum"}
).reset_index()
out_df["percentages"] = out_df["count_per_obj_per_part"] / out_df.groupby(
    ["stimuliId", "condition"]
)["count_per_obj_per_part"].transform("sum")

out_df = out_df.merge(count_nonZeroParts_df, on=["stimuliId", "condition"], how="left")

# drop_duplicates prevents a many-to-many merge (out_df_source still has one row per
# participant) from duplicating each aggregated row once per participant
is_target_lookup = out_df_source[["stimuliId", "condition", "obj", "is_target"]].drop_duplicates()
out_df = out_df.merge(is_target_lookup, on=["stimuliId", "condition", "obj"], how="left")

out_df.to_csv("output_all_withPercentages.csv", index=False)