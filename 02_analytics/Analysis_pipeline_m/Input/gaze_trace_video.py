"""
Gaze Trace Video — QC visualization
====================================
Renders an MP4 for one trial showing:
  - the actual stimulus image, positioned exactly as OpenSesame displayed it
  - AOI boxes (pos1-4 + subject), MEASURED directly from the real image
    files (connected-component blob detection for the 4 objects, pixel-diff
    for the subject) rather than hand-tweaked numbers
  - a moving dot tracking the real gaze position sample-by-sample, with a
    short fading trail; GREEN while inside the critical window, cyan outside
  - a timeline bar marking START_TRIAL, SUBJECT_ONSET_LOG,
    AUDIO_FILE_ONSET_LOG, the critical window (shaded), AUDIO_FILE_OFFSET
  - a live "gaze leaderboard": running valid-sample counts per object AOI,
    accumulated ONLY while inside the critical window, with the current
    leader highlighted — freezes at the final winner once the window closes
  - real audio muxed in if you pass --audio-file (via the imageio-ffmpeg
    bundled binary — no system ffmpeg install/PATH needed)

Coordinate geometry (confirmed against the actual OpenSesame item)
--------------------------------------------------------------------
    draw image center=1 file="[image_sub]" scale=0.8 x=0 y=0 z_index=0
  - stimulus image file is 2560x1440 px
  - OpenSesame canvas / experiment display is 2560x1440 px (NOT the
    1920x1080 that pygaze's init log declares — that number is very likely
    a stale/default config value, the same way the log's "60 Hz" samplerate
    turned out to be stale vs. the ~146 Hz actually recorded. 2560x1440 is
    what you confirmed as the actual monitor/experiment resolution.)
  - scale=0.8 is an EXPLICIT flat scale (not auto-fit-to-screen): the image
    is drawn at exactly 2560*0.8 x 1440*0.8 = 1920x1120 px
  - center=1, x=0, y=0 means it's centered on the canvas center (1280,720)
  => offset_x = (2560-1920)/2 = 320,  offset_y = (1440-1120)/2 = 160
  => screen_px = image_px * 0.8 + offset,  normalized = screen_px / (2560,1440)

If any of these OpenSesame parameters differ for your actual experiment
file, update SCALE / SCREEN_W / SCREEN_H below accordingly — everything
else derives from those three numbers.

Usage:
    python gaze_trace_video.py --trial 45 --group-csv participant_group5.csv --audio-file 44_r.wav
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy import ndimage

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = None

HERE = Path(__file__).resolve().parent
DEFAULT_GAZE_TSV = HERE / "j.tsv"
DEFAULT_TRIAL_CSV = HERE / "j.csv"
DEFAULT_ANNOTATION_CSV = HERE / "annotation_audiov1.csv"
DEFAULT_STIMULUS_IMG = HERE / "1_pos1_sub.png"
DEFAULT_GROUP_CSV = None
DEFAULT_IMAGES_DIR = HERE
OUT_DIR = HERE / "qc_output"
FRAMES_DIR = HERE / "video_frames"

# ──────────────────────────────────────────────────────────────
# GEOMETRY — from the actual OpenSesame `draw image` item
# ──────────────────────────────────────────────────────────────
IMG_W, IMG_H = 2560, 1440
SCALE = 0.8                        # explicit OpenSesame scale=0.8 (NOT auto-fit)
SCREEN_W, SCREEN_H = 2560, 1440    # actual OpenSesame/monitor canvas
_DISP_W, _DISP_H = IMG_W * SCALE, IMG_H * SCALE
_X_OFFSET = (SCREEN_W - _DISP_W) / 2
_Y_OFFSET = (SCREEN_H - _DISP_H) / 2


def _img_to_screen_px(x, y):
    return x * SCALE + _X_OFFSET, y * SCALE + _Y_OFFSET


def _img_box_to_norm(x1, y1, x2, y2):
    sx1, sy1 = _img_to_screen_px(x1, y1)
    sx2, sy2 = _img_to_screen_px(x2, y2)
    return (sx1 / SCREEN_W, sy1 / SCREEN_H, sx2 / SCREEN_W, sy2 / SCREEN_H)


def measure_aoi_boxes(sub_img_path, nosub_img_path):
    """
    Measure AOI boxes directly from the real stimulus images instead of
    using hand-tweaked constants:
      - the 4 object boxes: connected-component blob detection on the
        no-subject image (each object is an isolated non-white blob)
      - the subject box: pixel-difference between the sub/nosub image pair

    Returns {"pos1": (x1,y1,x2,y2), ..., "subject": (...)} in NORMALIZED
    (0-1) screen coordinates, ready to compare against BPOGX/BPOGY.

    Raises FileNotFoundError / RuntimeError with a clear, specific message
    if the images can't be read or don't contain exactly 4 object blobs —
    the caller (load_trial_data) catches this and falls back to the
    default AOI set rather than crashing the whole render.
    """
    nosub = np.array(Image.open(nosub_img_path).convert("RGB"))

    # Foreground detection via border-connected flood-fill, NOT a flat
    # brightness threshold. A flat cutoff (e.g. "sum(RGB) < 3*245 = object")
    # silently clips pale/cream-colored objects (bread dough, light wood,
    # etc.) because their lighter pixels get misclassified as background —
    # this was confirmed empirically (a "dough" object's box was cut short
    # by ~40px on its pale top edge). Flood-fill instead follows the actual
    # connected white canvas starting from the image border; anything NOT
    # reachable from the border is real object content, regardless of how
    # light-colored it is internally, as long as there's a genuine edge.
    is_bgish = nosub.sum(axis=2) > 3 * 250   # near-PURE white only — tight on purpose, so pale/cream
                                              # objects (e.g. bread dough) aren't swallowed into "background"
    labeled_bg, _ = ndimage.label(is_bgish)
    border_labels = set(labeled_bg[0, :]) | set(labeled_bg[-1, :]) | \
                     set(labeled_bg[:, 0]) | set(labeled_bg[:, -1])
    border_labels.discard(0)
    background_mask = np.isin(labeled_bg, list(border_labels))
    non_white = ~background_mask

    labeled, n = ndimage.label(non_white)
    boxes = []
    for sl in ndimage.find_objects(labeled):
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop
        area = (x2 - x1) * (y2 - y1)
        if area > 2000:
            boxes.append((x1, y1, x2, y2))

    if len(boxes) == 5:
        # Most likely cause: a _sub.png (with subject) got passed in as the
        # nosub image (e.g. the real _nosub file is missing on disk and a
        # mismatched file was found instead). Try to recover by dropping
        # the subject-like blob: portrait-oriented (taller than wide) and
        # horizontally near the image center — objects in this design are
        # off-center and not portrait-shaped the way the subject figure is.
        def _is_subject_like(b):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2
            near_center = abs(cx - IMG_W / 2) < IMG_W * 0.12
            portrait = h > w * 1.3
            return near_center and portrait

        candidates = [b for b in boxes if _is_subject_like(b)]
        if len(candidates) == 1:
            print(f"  [warn] found 5 blobs in {nosub_img_path} (expected 4) — this image likely "
                  f"includes the subject, meaning the true *_nosub.png file wasn't found and a "
                  f"mismatched file was used instead. Auto-dropped the subject-shaped blob and "
                  f"continued with the remaining 4 — but you should check that the correct "
                  f"*_nosub.png file actually exists in --images-dir for this trial.")
            boxes = [b for b in boxes if b not in candidates]

    if len(boxes) != 4:
        raise RuntimeError(f"Expected 4 object blobs in {nosub_img_path}, found {len(boxes)}. "
                            f"This usually means the wrong file (not the true *_nosub.png) was "
                            f"used, or --images-dir doesn't contain the matching file for this trial.")
    # sort into pos1 (top-right/rightmost) -> pos2 (bottom-right) -> pos3 (bottom-left) -> pos4 (top-left)
    # matches this design's convention: pos1/pos4 are the upper pair, pos2/pos3 the lower pair,
    # ordered right-to-left within each pair
    upper = sorted([b for b in boxes if (b[1] + b[3]) / 2 < IMG_H / 2], key=lambda b: -b[0])
    lower = sorted([b for b in boxes if (b[1] + b[3]) / 2 >= IMG_H / 2], key=lambda b: -b[0])
    if len(upper) != 2 or len(lower) != 2:
        raise RuntimeError(f"Found 4 blobs in {nosub_img_path} but they aren't split 2-upper/2-lower "
                            f"as expected for this design — got {len(upper)} upper, {len(lower)} lower.")
    ordered = {"pos1": upper[0], "pos4": upper[1], "pos2": lower[0], "pos3": lower[1]}

    sub = np.array(Image.open(sub_img_path).convert("RGB"))
    diff = np.abs(sub.astype(int) - nosub.astype(int)).sum(axis=2)
    ys, xs = np.where(diff > 15)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(f"sub/nosub image diff is empty for {sub_img_path} vs {nosub_img_path} "
                            f"(they look identical) — can't locate the subject box from these two files.")
    subject_img_px = (xs.min(), ys.min(), xs.max(), ys.max())

    aoi_norm = {name: _img_box_to_norm(*box) for name, box in ordered.items()}
    aoi_norm["subject"] = _img_box_to_norm(*subject_img_px)
    return aoi_norm


def classify_aoi(x, y, aoi_norm):
    for name, (x1, y1, x2, y2) in aoi_norm.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "elsewhere"


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_trial_data(gaze_tsv, trial_csv, annotation_csv, group_csv, images_dir,
                     trial_idx=None, sentence_id=None, condition=None):
    gaze_df = pd.read_csv(gaze_tsv, sep="\t", low_memory=False)
    gaze_df["USER"] = gaze_df["USER"].fillna("0").astype(str).str.strip()
    trial_df = pd.read_csv(trial_csv, low_memory=False)
    trial_df = trial_df[trial_df["audio_file"].notna() & (trial_df["audio_file"].astype(str) != "undefined")]
    trial_df = trial_df.sort_values("count_trial_loop").reset_index(drop=True)
    ann_df = pd.read_csv(annotation_csv)

    group_df = None
    if group_csv is not None:
        group_df = pd.read_csv(group_csv).set_index("id")

    events = gaze_df[gaze_df["USER"] != "0"][["TIME", "USER"]].reset_index(drop=True)
    starts = events[events.USER == "START_TRIAL"]["TIME"].values

    if trial_idx is None:
        matches = trial_df[(trial_df["audio_file"].astype(str).str.startswith(f"{sentence_id}_")) &
                            (trial_df["condition"] == condition)]
        if matches.empty:
            raise ValueError(f"No trial found for sentence_id={sentence_id}, condition={condition}")
        trial_idx = matches.index[0]

    t_row = trial_df.iloc[trial_idx]
    audio_file = str(t_row["audio_file"])
    audio_id = int(audio_file.split("_")[0])
    condition = str(t_row["condition"])
    target_position = int(t_row["target_position"])

    stimulus_img_sub, stimulus_img_nosub = DEFAULT_STIMULUS_IMG, None
    aoi_labels = {"pos1": "pos1", "pos2": "pos2", "pos3": "pos3", "pos4": "pos4"}
    if group_df is not None:
        if audio_id not in group_df.index:
            print(f"  [warn] no row for id={audio_id} in group CSV — falling back to default image")
        else:
            g_row = group_df.loc[audio_id]
            if str(g_row["condition"]) != condition:
                print(f"  [warn] condition mismatch for id={audio_id}: "
                      f"trial CSV says '{condition}', group CSV says '{g_row['condition']}'")
            if int(g_row["target_position"]) != target_position:
                print(f"  [warn] target_position mismatch for id={audio_id}: "
                      f"trial CSV says {target_position}, group CSV says {int(g_row['target_position'])}")
            sub_candidate = Path(images_dir) / str(g_row["image_sub"])
            nosub_candidate = Path(images_dir) / str(g_row["image_nosub"])
            if sub_candidate.exists():
                stimulus_img_sub = sub_candidate
            else:
                print(f"  [warn] image file not found: {sub_candidate} — falling back to default image")
            if nosub_candidate.exists():
                stimulus_img_nosub = nosub_candidate
            aoi_labels = {f"pos{i}": f"pos{i}: {g_row[f'position{i}']}" for i in range(1, 5)}
    else:
        print("  [warn] no --group-csv given — using a single fixed stimulus image for every trial.")

    # measure AOI boxes from THIS trial's own image pair when available,
    # falling back to the default 1_pos1_* pair otherwise
    if stimulus_img_nosub is not None:
        try:
            aoi_norm = measure_aoi_boxes(stimulus_img_sub, stimulus_img_nosub)
        except Exception as e:
            print(f"  [warn] AOI measurement failed for this trial's own images ({e}); "
                  f"falling back to the default 1_pos1 image pair's AOI boxes. Gaze-to-AOI "
                  f"classification for THIS trial may be inaccurate — check that the correct "
                  f"*_nosub.png file exists in --images-dir.")
            aoi_norm = measure_aoi_boxes(DEFAULT_STIMULUS_IMG, HERE / "1_pos1_nosub.png")
    else:
        aoi_norm = measure_aoi_boxes(DEFAULT_STIMULUS_IMG, HERE / "1_pos1_nosub.png")

    verbs = ann_df[ann_df.WordRole == "ROOT"][["id", "SentenceRole", "end"]].rename(
        columns={"end": "verb_offset_s", "SentenceRole": "condition"})
    targets = ann_df[ann_df.WordRole == "dobj"][["id", "SentenceRole", "start", "word"]].rename(
        columns={"start": "target_onset_s", "word": "target_word", "SentenceRole": "condition"})
    lookup_df = verbs.merge(targets, on=["id", "condition"])
    row = lookup_df[(lookup_df["id"] == audio_id) & (lookup_df["condition"] == condition)].iloc[0]
    verb_offset_s, target_onset_s, target_word = row["verb_offset_s"], row["target_onset_s"], row["target_word"]

    t_start = starts[trial_idx]
    t_end = starts[trial_idx + 1] if trial_idx + 1 < len(starts) else gaze_df["TIME"].max()
    seg = gaze_df[(gaze_df["TIME"] >= t_start) & (gaze_df["TIME"] < t_end)].copy()
    seg_events = events[(events["TIME"] >= t_start) & (events["TIME"] < t_end)]

    def _ev_time(name):
        v = seg_events.loc[seg_events.USER == name, "TIME"].values
        return v[0] if len(v) else None

    audio_onset = _ev_time("AUDIO_FILE_ONSET_LOG")
    info = {
        "trial_idx": trial_idx, "sentence_id": audio_id, "condition": condition,
        "sentence": str(t_row["sentence"]), "target_word": target_word,
        "target_position": target_position,
        "stimulus_img": stimulus_img_sub, "aoi_labels": aoi_labels, "aoi_norm": aoi_norm,
        "t_start": t_start, "t_end": t_end,
        "subject_onset": _ev_time("SUBJECT_ONSET_LOG"),
        "audio_onset": audio_onset,
        "audio_offset": _ev_time("AUDIO_FILE_OFFSET"),
        "stop_trial": _ev_time("STOP_TRIAL"),
        "crit_start": (audio_onset + verb_offset_s) if audio_onset is not None else None,
        "crit_end": (audio_onset + target_onset_s) if audio_onset is not None else None,
    }
    return seg, info


# ──────────────────────────────────────────────────────────────
# FRAME RENDERING
# ──────────────────────────────────────────────────────────────

def render_video(gaze_tsv=DEFAULT_GAZE_TSV, trial_csv=DEFAULT_TRIAL_CSV,
                  annotation_csv=DEFAULT_ANNOTATION_CSV, group_csv=DEFAULT_GROUP_CSV,
                  images_dir=DEFAULT_IMAGES_DIR,
                  trial_idx=None, sentence_id=None, condition=None, fps=30,
                  trail_seconds=0.4, audio_file=None, out_path=None):
    seg, info = load_trial_data(gaze_tsv, trial_csv, annotation_csv, group_csv, images_dir,
                                 trial_idx, sentence_id, condition)
    aoi_norm = info["aoi_norm"]
    aoi_labels = info["aoi_labels"]
    obj_aois = ["pos1", "pos2", "pos3", "pos4"]

    print(f"Rendering trial {info['trial_idx']}: sentence_id={info['sentence_id']}, "
          f"condition={info['condition']}, target='{info['target_word']}' (pos{info['target_position']})")
    print(f"  stimulus image: {info['stimulus_img']}")
    print(f"  AOI boxes (normalized): " + ", ".join(f"{k}={tuple(round(v,3) for v in b)}" for k, b in aoi_norm.items()))
    print(f"  subject_onset={info['subject_onset']:.3f}  audio_onset={info['audio_onset']:.3f}  "
          f"critical_window=[{info['crit_start']:.3f}, {info['crit_end']:.3f}]  "
          f"audio_offset={info['audio_offset']:.3f}")

    t0 = info["t_start"]
    t_last = seg["TIME"].max()
    frame_times = np.arange(t0, t_last, 1 / fps)
    samp_times = seg["TIME"].values
    samp_x, samp_y, samp_v = seg["BPOGX"].values, seg["BPOGY"].values, seg["BPOGV"].values

    # pre-classify every valid sample once (avoids recomputation per frame)
    samp_aoi = np.array([classify_aoi(x, y, aoi_norm) if v == 1 else "invalid"
                         for x, y, v in zip(samp_x, samp_y, samp_v)])

    # Duration each sample represents (time until the next sample), used to
    # compute TIME-weighted probabilities rather than naive sample counts.
    # Raw sample count is a biased proxy for "time spent": if tracking drops
    # out more often while looking at one object than another, that object
    # would look artificially under-represented by count alone even if the
    # real gaze duration was the same. Capped at 3x the median interval so a
    # single large gap (e.g. after a tracking-loss dropout) doesn't get
    # attributed entirely to whichever AOI preceded it.
    if len(samp_times) > 1:
        raw_gaps = np.diff(samp_times, append=samp_times[-1] + np.median(np.diff(samp_times)))
        median_gap = np.median(np.diff(samp_times))
        samp_dur = np.clip(raw_gaps, 0, median_gap * 3)
    else:
        samp_dur = np.array([0.0] * len(samp_times))

    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    img = Image.open(info["stimulus_img"])
    target_aoi = f"pos{info['target_position']}"

    for fi, tf in enumerate(frame_times):
        fig, (ax, ax_tl) = plt.subplots(2, 1, figsize=(9.6, 6.6),
                                         gridspec_kw={"height_ratios": [8, 1]})

        ax.add_patch(mpatches.Rectangle((0, 0), SCREEN_W, SCREEN_H, color="black", zorder=0))
        ax.imshow(img, extent=[_X_OFFSET, _X_OFFSET + _DISP_W, _Y_OFFSET + _DISP_H, _Y_OFFSET], zorder=1)

        for name, (x1, y1, x2, y2) in aoi_norm.items():
            is_target = (name == target_aoi)
            bx1, by1, bx2, by2 = x1 * SCREEN_W, y1 * SCREEN_H, x2 * SCREEN_W, y2 * SCREEN_H
            ax.add_patch(mpatches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                             linewidth=2.5 if is_target else 1,
                                             edgecolor="yellow" if is_target else "cyan",
                                             facecolor="none", alpha=0.9 if is_target else 0.4, zorder=2))
            label = aoi_labels.get(name, name)
            if is_target:
                label += "  [TARGET]"
            ax.text(bx1, by1 - 10, label, color="yellow" if is_target else "cyan", fontsize=8, zorder=2)

        in_crit_now = info["crit_start"] is not None and info["crit_start"] <= tf <= info["crit_end"]

        mask = (samp_times <= tf) & (samp_times >= tf - trail_seconds)
        if mask.any():
            tx, ty, tv, tt = samp_x[mask], samp_y[mask], samp_v[mask], samp_times[mask]
            age = tf - tt
            alpha = np.clip(1 - age / trail_seconds, 0.05, 1.0)
            for xi, yi, vi, ai, ti in zip(tx, ty, tv, alpha, tt):
                if vi != 1:
                    continue
                is_crit = info["crit_start"] is not None and info["crit_start"] <= ti <= info["crit_end"]
                color = "#00ff44" if is_crit else "#66d9ff"
                ax.scatter(xi * SCREEN_W, yi * SCREEN_H, s=60, color=color, alpha=float(ai), zorder=3)
            cx, cy, cv = tx[-1], ty[-1], tv[-1]
            if cv == 1:
                color = "#00ff44" if in_crit_now else "#66d9ff"
                ax.scatter(cx * SCREEN_W, cy * SCREEN_H, s=220, color=color,
                           edgecolor="black", linewidth=1.5, zorder=4)

        ax.set_xlim(0, SCREEN_W); ax.set_ylim(SCREEN_H, 0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        rel_t = tf - t0
        title = f"Trial {info['trial_idx']}  |  t = {rel_t:5.2f}s"
        if in_crit_now:
            title += "   ●  CRITICAL WINDOW"
        audio_playing = (info["audio_onset"] is not None and info["audio_offset"] is not None
                          and info["audio_onset"] <= tf <= info["audio_offset"])
        if audio_playing:
            title += "   [AUDIO PLAYING]"
        ax.set_title(title, fontsize=12, loc="left")

        # ── live gaze leaderboard: TIME-weighted probability per AOI, ──
        # ── accumulated ONLY while inside the critical window          ──
        if info["crit_start"] is not None:
            counted_mask = (samp_times <= min(tf, info["crit_end"])) & (samp_times >= info["crit_start"])
            durations = {a: float(samp_dur[(samp_aoi == a) & counted_mask].sum()) for a in obj_aois}
        else:
            durations = {a: 0.0 for a in obj_aois}
        total_dur = sum(durations.values())
        probs = {a: (durations[a] / total_dur if total_dur > 0 else 0.0) for a in obj_aois}
        leader = max(probs, key=probs.get) if total_dur > 0 else None
        window_closed = info["crit_end"] is not None and tf >= info["crit_end"]

        board_lines = []
        for a in obj_aois:
            obj_name = aoi_labels.get(a, a).split(": ", 1)[-1]
            marker = " <=" if (a == leader and probs[a] > 0) else ""
            board_lines.append(f"{a} ({obj_name}): {probs[a]:.2f}{marker}")
        board_text = "P(gaze | critical window):\n" + "\n".join(board_lines)
        if leader and probs[leader] > 0:
            status = "WINNER" if window_closed else "leading"
            board_text += f"\n\n{status}: {aoi_labels.get(leader, leader)}"

        ax.text(0.015, 0.985, board_text, transform=ax.transAxes, fontsize=8,
                va="top", ha="left", color="white", zorder=6,
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.55, edgecolor="none"))

        # ── timeline bar ──
        t_span = (t_last - t0)
        ax_tl.set_xlim(0, t_span); ax_tl.set_ylim(0, 1.3)
        ax_tl.set_yticks([])
        ax_tl.set_xlabel("Time since START_TRIAL (s)")
        if info["crit_start"] is not None:
            ax_tl.axvspan(info["crit_start"] - t0, info["crit_end"] - t0, color="#00ff44", alpha=0.3)
        for name, tval, color in [
            ("subject onset", info["subject_onset"], "#1f77b4"),
            ("audio onset", info["audio_onset"], "#2ca02c"),
            ("audio offset", info["audio_offset"], "#d62728"),
            ("stop trial", info["stop_trial"], "#7f7f7f"),
        ]:
            if tval is not None:
                ax_tl.axvline(tval - t0, color=color, linewidth=1.5)
                ax_tl.text(tval - t0, 1.1, name, rotation=45, fontsize=7, color=color, ha="left")
        ax_tl.axvline(rel_t, color="black", linewidth=2)

        fig.tight_layout()
        fig.savefig(FRAMES_DIR / f"frame_{fi:05d}.png", dpi=110)
        plt.close(fig)

        if fi % 30 == 0:
            print(f"  rendered frame {fi}/{len(frame_times)}")

    # final winner summary printed to console too
    if info["crit_start"] is not None:
        final_mask = (samp_times <= info["crit_end"]) & (samp_times >= info["crit_start"])
        final_dur = {a: float(samp_dur[(samp_aoi == a) & final_mask].sum()) for a in obj_aois}
        final_total = sum(final_dur.values())
        final_probs = {a: round(final_dur[a] / final_total, 4) if final_total > 0 else 0.0 for a in obj_aois}
        winner = max(final_probs, key=final_probs.get) if final_total > 0 else None
        print(f"  Final critical-window gaze probabilities (duration-weighted): {final_probs}")
        if winner and final_probs[winner] > 0:
            is_correct = (winner == target_aoi)
            print(f"  Most-fixated object: {aoi_labels.get(winner, winner)} "
                  f"({'MATCHES target' if is_correct else 'does NOT match target'})")

    print(f"Rendered {len(frame_times)} frames. Encoding video...")
    out_path = Path(out_path) if out_path else OUT_DIR / f"gaze_trace_trial{info['trial_idx']}.mp4"
    OUT_DIR.mkdir(exist_ok=True)

    ffmpeg_bin = FFMPEG_EXE or "ffmpeg"
    cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", str(FRAMES_DIR / "frame_%05d.png")]
    if audio_file and Path(audio_file).exists():
        audio_delay_s = max(0.0, info["audio_onset"] - t0)
        cmd += ["-itsoffset", str(audio_delay_s), "-i", str(audio_file),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += [str(out_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Could not find an ffmpeg binary. Run: pip install imageio-ffmpeg"
        )
    if result.returncode != 0:
        print("ffmpeg STDERR:\n", result.stderr[-2000:])
        raise RuntimeError("ffmpeg encoding failed")

    shutil.rmtree(FRAMES_DIR)
    print(f"Saved: {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Render a gaze-trace QC video for one trial.")
    p.add_argument("--gaze-tsv", type=str, default=str(DEFAULT_GAZE_TSV))
    p.add_argument("--trial-csv", type=str, default=str(DEFAULT_TRIAL_CSV))
    p.add_argument("--annotation-csv", type=str, default=str(DEFAULT_ANNOTATION_CSV))
    p.add_argument("--group-csv", type=str, default=None,
                    help="Participant's arrangement/design CSV (e.g. participant_group5.csv)")
    p.add_argument("--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR))
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--sentence-id", type=int, default=None)
    p.add_argument("--condition", choices=["restrictive", "non-restrictive"], default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--trail-seconds", type=float, default=0.4)
    p.add_argument("--audio-file", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.trial is None and (args.sentence_id is None or args.condition is None):
        p.error("Provide --trial, or both --sentence-id and --condition")
    return args


if __name__ == "__main__":
    args = parse_args()
    render_video(gaze_tsv=args.gaze_tsv, trial_csv=args.trial_csv, annotation_csv=args.annotation_csv,
                 group_csv=args.group_csv, images_dir=args.images_dir,
                 trial_idx=args.trial, sentence_id=args.sentence_id, condition=args.condition,
                 fps=args.fps, trail_seconds=args.trail_seconds, audio_file=args.audio_file,
                 out_path=args.out)