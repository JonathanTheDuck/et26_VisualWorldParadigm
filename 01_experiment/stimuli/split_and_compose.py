#!/usr/bin/env python3
"""
split_and_compose.py
────────────────────
Batch pipeline for the VWP eye-tracking experiment:
  img_matrices/{N}_{TL}_{TR}_{BR}_{BL}.png
      → discrete crops  (img_discrete/)
      → trial canvases  (img_composition/{N}_pos{1-4}_{nosub|sub}.png)

Filename convention
───────────────────
  {N}_{obj1}_{obj2}_{obj3}_{obj4}.png
  obj1 = TL (target, top-left)
  obj2 = TR (distractor 1, top-right)
  obj3 = BL (distractor 2, bottom-left)
  obj4 = BR (distractor 3, bottom-right)

  Multi-word objects use hyphens, NOT underscores (e.g. magnifying-glass).
  This keeps the underscore separator unambiguous.

Trial canvas layout  (1200 × 700 px, white background)
───────────────────
  ┌─────────────────────────────────────────────┐
  │             [subject image]                  │  ← per-item, top-centre
  │     [slot0]              [slot3]             │  ← 4 options on a
  │           [slot1]  [slot2]                   │    semicircle/ellipse
  └─────────────────────────────────────────────┘    around the subject

  Options sit on an ellipse around the subject (see SEMICIRCLE_GAP /
  SEMICIRCLE_ANGLES) rather than a fixed row, so every option is an equal
  visual gap from the subject regardless of which slot it lands in.

Subject image lookup
─────────────────────
  Each item's subject noun is extracted from sentence1 in sentences.csv
  (e.g. 'The boy will eat the cake' -> 'boy') and looked up as
  img_matrices/subjects/subject-the {noun}.png (the team's actual naming
  convention). Falls back to --subject FILE (one image for every trial)
  if no per-item match exists, else no subject image is drawn.

Trial canvas variants
─────────────────────
  Every item is composed in all 4 target positions (filename pos1-pos4;
  rotation 0-3 internally) and both with/without the subject image — 8
  files per item — so any participant-group/condition combination can
  pick the right one later without re-running PIL. See compose_trial.

Participant-group CSVs (--participants)
────────────────────────────────────────
  4 CSVs (creatingDataStructure/participant_group{1-4}.csv), one per
  participant group, 50 rows each (one row per item; `id` = item id
  0-49, matching sentences.csv, not a subject identifier). 20 real
  participants / 4 groups = 5 each; all 5 participants in a group see
  the identical file, so which subjects use which file is tracked
  externally, not as a column.
    - Verb version is fixed by item order for every group: id 0-24 use
      the restrictive sentence, id 25-49 use the non-restrictive one.
    - Target position rotates for every item (restrictive and
      non-restrictive alike): rotation = (id + group) % 4, so the same
      item lands in a different on-screen slot for each group, and a
      single group cycles through all 4 slots across its items. All 4
      rotated images already exist for every item (see compose_trial),
      so this is purely a CSV lookup, no image regeneration needed.
    - Both subject variants are listed per row (image_nosub, image_sub,
      same target position) rather than picking one per item.
    - position{1-4} name the object shown in each on-screen slot for that
      trial (object set is fixed per item, cyclically rotated by rotation;
      the target always lands on position == target_position).

Usage
─────
  # Process all images in img_matrices/
  python split_and_compose.py

  # Add a subject/agent image at the top of every trial
  python split_and_compose.py --subject path/to/subject.png

  # Dry-run: show what would be processed without writing files
  python split_and_compose.py --dry-run

  # Only process specific pic IDs
  python split_and_compose.py --ids 14 15 16

  # Keep existing img_composition/*.png files (only write discrete crops)
  python split_and_compose.py --no-overwrite

  # Generate the 4 participant-group CSVs only
  python split_and_compose.py --participants
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed.  Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# ── Directory layout ───────────────────────────────────────────────────────────

STIMULI_DIR  = Path(__file__).parent
MATRICES_DIR = STIMULI_DIR / "img_matrices"
DISCRETE_DIR = STIMULI_DIR / "img_discrete"
SUBJECTS_DIR = MATRICES_DIR / "subjects"
IMG_DIR      = STIMULI_DIR / "img_composition"

SENTENCES_CSV       = STIMULI_DIR / "creatingDataStructure" / "sentences.csv"
PARTICIPANT_CSV_DIR = STIMULI_DIR / "creatingDataStructure"

# ── Trial canvas settings (match Code 2 from project notes) ───────────────────
#
# Layout proportions are defined at 1x below, then scaled up by SCALE so every
# element's target box is closer to the source assets' native resolution
# (subjects ~1100-1400px, discrete crops ~550-700px) and needs less downscaling.

SCALE = 2

CANVAS_W, CANVAS_H = 1200 * SCALE, 700 * SCALE
SUBJECT_SIZE        = (420 * SCALE, 300 * SCALE)
SUBJECT_Y           = 50 * SCALE
OPTION_SIZE         = (200 * SCALE, 200 * SCALE)
ROLE_ORDER          = ["TL", "TR", "BR", "BL"]   # TL = target, others = distractors

# Options sit on an ellipse around the subject (semicircle layout) instead of
# a fixed row, so every option is an equal visual gap from the subject's edge
# regardless of which slot the target rotates into.
SEMICIRCLE_GAP    = 95 * SCALE
SEMICIRCLE_ANGLES = [15, 65, 115, 165]   # degrees, slot0..slot3 (screen y-down)

# ── Participant-group design ───────────────────────────────────────────────────

N_GROUPS       = 4
N_RESTRICTIVE  = 25   # item ids < this use the restrictive sentence/audio

# ── Filename parser ────────────────────────────────────────────────────────────

_RE = re.compile(r"^(\d+)_(.+)\.png$", re.IGNORECASE)


def parse_matrix_filename(fname: str):
    """
    Parse '{N}_{obj1}_{obj2}_{obj3}_{obj4}.png'.
    Returns (N: int, [TL, TR, BL, BR]) or None if unrecognised.
    Multi-word objects must use hyphens (magnifying-glass), not underscores.
    """
    m = _RE.match(fname)
    if not m:
        return None
    n    = int(m.group(1))
    rest = m.group(2)          # everything after the id and first underscore
    objs = rest.split("_")    # split only on underscores → hyphens inside names survive
    if len(objs) != 4:
        return None
    return n, objs             # [TL, TR, BL, BR]


# ── Image helpers ──────────────────────────────────────────────────────────────

def crop_quadrants(img: Image.Image) -> dict[str, Image.Image]:
    """Split a 2×2 collage into {TL, TR, BR, BL} quadrant images."""
    w, h   = img.size
    mx, my = w // 2, h // 2
    return {
        "TL": img.crop((0,  0,  mx, my)),
        "TR": img.crop((mx, 0,  w,  my)),
        "BR": img.crop((mx, my, w,  h )),
        "BL": img.crop((0,  my, mx, h )),
    }


def fit_image(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Resize-to-fit (preserving aspect ratio) and centre on a transparent canvas."""
    img = img.convert("RGBA")
    img.thumbnail(target, Image.LANCZOS)
    canvas = Image.new("RGBA", target, (255, 255, 255, 0))
    x = (target[0] - img.width)  // 2
    y = (target[1] - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def role_to_slot(rotation: int) -> dict[str, int]:
    """Map each role (TL/TR/BR/BL) to a screen slot index (0-3) for a given rotation."""
    return {role: (i + rotation) % 4 for i, role in enumerate(ROLE_ORDER)}


def compose_trial(
    quadrants:   dict[str, Image.Image],
    subject_img: Image.Image | None,
    rotation:    int = 0,
) -> Image.Image:
    """
    Compose a trial canvas: optional subject at top-centre, 4 objects on a
    semicircle ellipse around it. `rotation` (0-3) cyclically shifts which
    angle slot each role lands in; rotation=0 puts TL (the target) at
    SEMICIRCLE_ANGLES[0].
    """
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")

    center_x = CANVAS_W // 2
    center_y = SUBJECT_Y + SUBJECT_SIZE[1] // 2

    if subject_img is not None:
        subj = fit_image(subject_img.copy(), SUBJECT_SIZE)
        sx   = (CANVAS_W - SUBJECT_SIZE[0]) // 2
        canvas.paste(subj, (sx, SUBJECT_Y), subj)

    a = SUBJECT_SIZE[0] / 2 + SEMICIRCLE_GAP + OPTION_SIZE[0] / 2
    b = SUBJECT_SIZE[1] / 2 + SEMICIRCLE_GAP + OPTION_SIZE[1] / 2

    slots = role_to_slot(rotation)
    for pos_key in ROLE_ORDER:
        angle = math.radians(SEMICIRCLE_ANGLES[slots[pos_key]])
        ox = center_x + a * math.cos(angle)
        oy = center_y + b * math.sin(angle)
        opt = fit_image(quadrants[pos_key].copy(), OPTION_SIZE)
        canvas.paste(opt, (int(ox - OPTION_SIZE[0] / 2), int(oy - OPTION_SIZE[1] / 2)), opt)

    return canvas


# ── Per-item subject image lookup ───────────────────────────────────────────────
#
# Looks up img_matrices/subjects/{subject-slug}.png per sentence, keyed off the
# sentence's subject noun (e.g. 'boy', 'businessman', 'grandmother'). Naming
# convention is provisional — adjust SUBJECTS_DIR/find_subject_file once the
# real subject image set + naming scheme is provided.

def extract_subject(sentence: str) -> str:
    """
    Extract the sentence's subject noun (lowercase, spaces kept as-is).
    'The boy will eat the cake'            -> 'boy'
    'The businessman will wear the hat'    -> 'businessman'
    Strategy: text between a leading 'The ' and the first ' will '.
    """
    m = re.match(r"^\s*[Tt]he\s+(.+?)\s+will\b", sentence.strip())
    if not m:
        return ""
    return m.group(1).strip().lower()


def load_subjects() -> dict[int, str]:
    """Map sentence id -> subject noun slug, read from sentences.csv (sentence1)."""
    subjects = {}
    with open(SENTENCES_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)   # header
        for row in reader:
            if row and row[0].strip().isdigit():
                sid = int(row[0].strip())
                sentence1 = row[1].strip() if len(row) > 1 else ""
                subjects[sid] = extract_subject(sentence1)
    return subjects


def find_subject_file(subject: str) -> Path | None:
    """
    Return the subject image for a given subject noun, if one exists.
    Primary convention (matches the team's actual files):
    'subject-the {noun}.png', e.g. 'subject-the boy.png'. A couple of
    looser variants are tried too in case the convention drifts.
    """
    if not subject or not SUBJECTS_DIR.exists():
        return None
    stems = [f"subject-the {subject}", f"subject-{subject}", subject]
    for stem in stems:
        for ext in (".png", ".jpg", ".jpeg"):
            p = SUBJECTS_DIR / f"{stem}{ext}"
            if p.exists():
                return p
    return None


# ── Participant-group CSVs ──────────────────────────────────────────────────────

def load_sentence_ids() -> list[int]:
    """Read just the item ids (column 0) from sentences.csv, in file order."""
    ids = []
    with open(SENTENCES_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)   # header
        for row in reader:
            if row and row[0].strip().isdigit():
                ids.append(int(row[0].strip()))
    return ids


def load_item_objects() -> dict[int, dict[str, str]]:
    """
    Map item id (= pic N - 1) -> {role: object} read from img_matrices
    filenames ({N}_{TL}_{TR}_{BL}_{BR}.png). Used to record which object sits
    in each on-screen slot per trial (see participant_row / position columns).
    """
    objects = {}
    if not MATRICES_DIR.exists():
        return objects
    for p in sorted(MATRICES_DIR.glob("*.png")):
        parsed = parse_matrix_filename(p.name)
        if parsed is None:
            continue
        n, objs = parsed                       # objs = [TL, TR, BL, BR]
        tl, tr, bl, br = objs
        objects[n - 1] = {"TL": tl, "TR": tr, "BR": br, "BL": bl}
    return objects


def load_sentence_texts() -> dict[int, tuple[str, str]]:
    """Map sentence id -> (sentence1 restrictive text, sentence2 non-restrictive text)."""
    texts = {}
    with open(SENTENCES_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)   # header
        for row in reader:
            if row and row[0].strip().isdigit():
                sid = int(row[0].strip())
                texts[sid] = (
                    row[1].strip() if len(row) > 1 else "",
                    row[2].strip() if len(row) > 2 else "",
                )
    return texts


def participant_row(
    sid: int,
    group: int,
    sentence_texts: dict[int, tuple[str, str]],
    item_objects: dict[int, dict[str, str]],
) -> dict:
    """
    One OpenSesame trial-table row for item `sid` as seen by `group` (0-3).
    See the module docstring ("Participant-group CSVs") for the rules.

    position{1-4} record which object sits in each on-screen slot for this
    trial. rotation cyclically shifts the roles across slots (role_to_slot);
    inverting that, the object at position k is the role that maps to slot
    (k-1), i.e. ROLE_ORDER[(k-1-rotation) % 4]. The target (TL) therefore
    always lands on position == target_position.
    """
    restrictive  = sid < N_RESTRICTIVE
    rotation     = (sid + group) % 4
    sent1, sent2 = sentence_texts.get(sid, ("", ""))
    roles        = item_objects.get(sid, {})
    positions    = {
        f"position{k}": roles.get(ROLE_ORDER[(k - 1 - rotation) % 4], "")
        for k in range(1, 5)
    }
    return {
        "id":              sid,
        "image_nosub":     f"{sid + 1}_pos{rotation + 1}_nosub.png",
        "image_sub":       f"{sid + 1}_pos{rotation + 1}_sub.png",
        **positions,
        "audio_file":      f"{sid}_{'r' if restrictive else 'n'}.wav",
        "sentence":        sent1 if restrictive else sent2,
        "condition":       "restrictive" if restrictive else "non-restrictive",
        "target_position": rotation + 1,
        "timeout":         3000,
    }


def generate_participant_csvs() -> list[Path]:
    """Write creatingDataStructure/participant_group{1-4}.csv, 50 rows each."""
    ids            = load_sentence_ids()
    sentence_texts = load_sentence_texts()
    item_objects   = load_item_objects()
    out_paths = []
    for group in range(N_GROUPS):
        rows = [participant_row(sid, group, sentence_texts, item_objects) for sid in ids]
        out  = PARTICIPANT_CSV_DIR / f"participant_group{group + 1}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "image_nosub", "image_sub",
                "position1", "position2", "position3", "position4",
                "audio_file", "sentence",
                "condition", "target_position", "timeout",
            ])
            writer.writeheader()
            writer.writerows(rows)
        out_paths.append(out)
    return out_paths


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subject", metavar="FILE",
                    help="Fallback subject image used when no per-item match is "
                         "found in img_matrices/subjects/{subject-slug}.png")
    ap.add_argument("--ids", nargs="+", type=int, metavar="N",
                    help="Process only these pic IDs (e.g. --ids 14 15 16)")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="Skip writing to img_composition/pic{N}.png if the file already exists")
    ap.add_argument("--discrete-only", action="store_true",
                    help="Only write img_discrete/ crops; never touch img_composition/")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be processed without writing any files")
    ap.add_argument("--participants", action="store_true",
                    help="Generate the 4 creatingDataStructure/participant_group{1-4}.csv "
                         "files and exit")
    args = ap.parse_args()

    if args.participants:
        paths = generate_participant_csvs()
        for p in paths:
            print(f"Wrote 50 rows -> {p}")
        return

    if not MATRICES_DIR.exists():
        print(f"ERROR: img_matrices/ not found at {MATRICES_DIR}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        DISCRETE_DIR.mkdir(exist_ok=True)
        IMG_DIR.mkdir(exist_ok=True)

    # Fallback subject image (used only when no per-item match is found)
    fallback_subject_img: Image.Image | None = None
    if args.subject:
        fallback_subject_img = Image.open(args.subject)
        print(f"Fallback subject image: {args.subject}")

    # Per-item subject lookup: sentence id (= pic N - 1) -> subject noun slug
    subjects_by_id = load_subjects() if SENTENCES_CSV.exists() else {}
    n_subject_missing = 0

    # Collect and parse all matrix PNGs
    matrix_files = sorted(MATRICES_DIR.glob("*.png"))
    entries = []
    for p in matrix_files:
        parsed = parse_matrix_filename(p.name)
        if parsed is None:
            print(f"SKIP (unrecognised): {p.name}")
            continue
        n, objs = parsed
        if args.ids and n not in args.ids:
            continue
        entries.append((n, objs, p))

    if not entries:
        print("No matching files found.")
        return

    n_ok = n_err = 0
    n_trial_files = 0

    for n, objs, src_path in entries:
        tl, tr, bl, br = objs   # filename order: obj1=TL, obj2=TR, obj3=BL, obj4=BR
        disc_tl  = DISCRETE_DIR / f"{n}_{tl}_TL.png"
        disc_tr  = DISCRETE_DIR / f"{n}_{tr}_TR.png"
        disc_br  = DISCRETE_DIR / f"{n}_{br}_BR.png"
        disc_bl  = DISCRETE_DIR / f"{n}_{bl}_BL.png"

        sid           = n - 1   # sentence id in sentences.csv (pic N = id N-1)
        subject       = subjects_by_id.get(sid, "")
        subject_path  = find_subject_file(subject)
        if subject_path is not None:
            item_subject_img = Image.open(subject_path)
            subject_status    = f"subjects/{subject_path.name}"
        elif fallback_subject_img is not None:
            item_subject_img = fallback_subject_img
            subject_status    = "(fallback --subject image)"
        else:
            item_subject_img = None
            n_subject_missing += 1
            subject_status    = (f"MISSING (expected subjects/subject-the {subject}.png)"
                                  if subject else "MISSING (no subject noun found)")

        print(f"\npic{n:02d}  {tl}(TL)  {tr}(TR)  {br}(BR)  {bl}(BL)")
        print(f"  source:   {src_path.name}")
        print(f"  subject:  {subject_status}")

        variants = [("nosub", None)]
        if item_subject_img is not None:
            variants.append(("sub", item_subject_img))

        disc_paths   = [("TL", disc_tl), ("TR", disc_tr), ("BR", disc_br), ("BL", disc_bl)]
        discrete_exists = all(path.exists() for _, path in disc_paths)

        if args.dry_run:
            if discrete_exists:
                print(f"  [dry-run] discrete already exists, would keep as-is (not overwritten)")
            else:
                print(f"  [dry-run] would write discrete → img_discrete/{n}_*.png")
            if not args.discrete_only:
                names = [f"{n}_pos{r + 1}_{v}.png" for r in range(4) for v, _ in variants]
                print(f"  [dry-run] would write {len(names)} trial file(s): "
                      f"{names[0]} … {names[-1]}")
            n_ok += 1
            continue

        try:
            if discrete_exists:
                # Never re-crop over existing discrete files — they may have
                # been hand-fixed (e.g. cleaner split than the automatic crop).
                quads = {key: Image.open(path).convert("RGBA") for key, path in disc_paths}
                print(f"  discrete: kept existing (not re-cropped)")
            else:
                collage = Image.open(src_path).convert("RGBA")
                quads   = crop_quadrants(collage)
                for key, path in disc_paths:
                    quads[key].convert("RGB").save(path)
                print(f"  discrete: {n}_{tl}_TL.png  {n}_{tr}_TR.png  "
                      f"{n}_{br}_BR.png  {n}_{bl}_BL.png")

            # 2. Compose and save every (rotation x subject-variant) trial canvas
            if args.discrete_only:
                print(f"  trial:    SKIPPED (--discrete-only)")
            else:
                written = 0
                for rotation in range(4):
                    for variant_name, variant_img in variants:
                        out_path = IMG_DIR / f"{n}_pos{rotation + 1}_{variant_name}.png"
                        if args.no_overwrite and out_path.exists():
                            continue
                        trial = compose_trial(quads, variant_img, rotation=rotation)
                        trial.save(out_path)
                        written += 1
                print(f"  trial:    {written} file(s) written  ({CANVAS_W}×{CANVAS_H})")
                n_trial_files += written

            n_ok += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            n_err += 1

    print(f"\n{'─'*52}")
    print(f"Done.  OK: {n_ok}   Errors: {n_err}   Trial files written: {n_trial_files}")
    print(f"Discrete images : {DISCRETE_DIR}")
    print(f"Trial images    : {IMG_DIR}")
    if n_subject_missing:
        print(f"Subject images missing: {n_subject_missing}/{len(entries)} "
              f"(expected in {SUBJECTS_DIR}/)")


if __name__ == "__main__":
    main()
