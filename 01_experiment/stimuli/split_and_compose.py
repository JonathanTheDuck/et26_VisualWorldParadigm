#!/usr/bin/env python3
"""
split_and_compose.py
────────────────────
Batch pipeline for the VWP eye-tracking experiment:
  img_matrices/{N}_{TL}_{TR}_{BR}_{BL}.png
      → discrete crops  (img_discrete/)
      → trial canvas    (img_composition/pic{N}.png)

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
  │                                              │
  │  [slot0]  [slot1]  [slot2]  [slot3]          │  ← 4 options, bottom row
  └─────────────────────────────────────────────┘

Subject image lookup
─────────────────────
  Each item's subject noun is extracted from sentence1 in sentences.csv
  (e.g. 'The boy will eat the cake' -> 'boy') and looked up as
  img_matrices/subjects/subject-the {noun}.png (the team's actual naming
  convention). Falls back to --subject FILE (one image for every trial)
  if no per-item match exists, else no subject image is drawn.

Latin-square counterbalancing
──────────────────────────────
  1. Target screen position
     By default TL (the target) always lands in slot0 (leftmost), which
     confounds "target identity" with "screen position". Each pic N is
     instead given a cyclic rotation r = N % 4, which shifts every role
     (TL/TR/BR/BL) by r slots. Across the 50-item set this makes the
     target land in each of the 4 slots an (approximately) equal number
     of times. The resulting target slot/x-pixel for every pic is written
     to position_assignment.csv for later AOI/analysis cross-referencing.

  2. Restrictive / non-restrictive list assignment
     Mirrors the two-list design in Altmann & Kamide (1999): for each
     item, List A and List B always use opposite verb versions, so each
     list ends up with an even 25/25 restrictive/non-restrictive split.
     Generated on demand into creatingDataStructure/list_assignment.csv.

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

  # Keep existing img_composition/pic{N}.png files (only write discrete crops)
  python split_and_compose.py --no-overwrite

  # Generate the List A / List B verb-version counterbalancing CSV only
  python split_and_compose.py --lists
"""

import argparse
import csv
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
POSITION_LOG        = STIMULI_DIR / "position_assignment.csv"
LIST_ASSIGNMENT_CSV = STIMULI_DIR / "creatingDataStructure" / "list_assignment.csv"

# ── Trial canvas settings (match Code 2 from project notes) ───────────────────

CANVAS_W, CANVAS_H = 1200, 700
SUBJECT_SIZE        = (350, 250)
SUBJECT_Y           = 50
OPTION_SIZE         = (200, 200)
OPTION_Y            = 420
OPTION_X            = [80, 340, 600, 860]   # 4 horizontal slots
ROLE_ORDER          = ["TL", "TR", "BR", "BL"]   # TL = target, others = distractors

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


def latin_square_rotation(n: int) -> int:
    """
    Cyclic Latin-square rotation index (0-3) for pic N.
    Rotation 0 reproduces the original fixed layout (target/TL in slot0).
    Rotation r shifts every role r slots to the right (wrapping around),
    so across the full stimulus set each role lands in each of the 4
    screen slots an (approximately) equal number of times.
    """
    return n % 4


def role_to_slot(rotation: int) -> dict[str, int]:
    """Map each role (TL/TR/BR/BL) to a screen slot index (0-3) for a given rotation."""
    return {role: (i + rotation) % 4 for i, role in enumerate(ROLE_ORDER)}


def compose_trial(
    quadrants:   dict[str, Image.Image],
    subject_img: Image.Image | None,
    rotation:    int = 0,
) -> Image.Image:
    """
    Compose a trial canvas: optional subject at top, 4 objects in a row at bottom.
    `rotation` (see latin_square_rotation) cyclically shifts which screen
    slot each role lands in; rotation=0 puts TL (the target) in slot0,
    same as the original fixed layout.
    """
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")

    if subject_img is not None:
        subj = fit_image(subject_img.copy(), SUBJECT_SIZE)
        sx   = (CANVAS_W - SUBJECT_SIZE[0]) // 2
        canvas.paste(subj, (sx, SUBJECT_Y), subj)

    slots = role_to_slot(rotation)
    for pos_key in ROLE_ORDER:
        opt = fit_image(quadrants[pos_key].copy(), OPTION_SIZE)
        canvas.paste(opt, (OPTION_X[slots[pos_key]], OPTION_Y), opt)

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


# ── List A / List B verb-version assignment ────────────────────────────────────

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


def generate_list_assignment() -> list[dict]:
    """
    Two-list counterbalancing of verb version (restrictive/non-restrictive),
    mirroring Altmann & Kamide (1999): for every item, List A and List B
    always take opposite versions, so each list ends up with an even
    25/25 restrictive/non-restrictive split across the 50-item set.
    """
    rows = []
    for sid in load_sentence_ids():
        list_a = "restrictive" if sid % 2 == 0 else "non-restrictive"
        list_b = "non-restrictive" if list_a == "restrictive" else "restrictive"
        rows.append({
            "id":            sid,
            "image_file":    f"pic{sid + 1}.png",
            "listA_audio":   f"{sid}_{'r' if list_a == 'restrictive' else 'n'}.wav",
            "listA_version": list_a,
            "listB_audio":   f"{sid}_{'r' if list_b == 'restrictive' else 'n'}.wav",
            "listB_version": list_b,
        })
    return rows


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
    ap.add_argument("--lists", action="store_true",
                    help="Generate creatingDataStructure/list_assignment.csv "
                         "(List A/B verb-version counterbalancing) and exit")
    args = ap.parse_args()

    if args.lists:
        rows = generate_list_assignment()
        with open(LIST_ASSIGNMENT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "image_file", "listA_audio", "listA_version",
                "listB_audio", "listB_version",
            ])
            writer.writeheader()
            writer.writerows(rows)
        n_a_restrictive = sum(1 for r in rows if r["listA_version"] == "restrictive")
        print(f"Wrote {len(rows)} rows -> {LIST_ASSIGNMENT_CSV}")
        print(f"  List A: {n_a_restrictive} restrictive / {len(rows) - n_a_restrictive} non-restrictive")
        print(f"  List B: {len(rows) - n_a_restrictive} restrictive / {n_a_restrictive} non-restrictive")
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

    n_ok = n_skip = n_err = 0
    position_log_rows = []

    for n, objs, src_path in entries:
        tl, tr, bl, br = objs   # filename order: obj1=TL, obj2=TR, obj3=BL, obj4=BR
        rotation = latin_square_rotation(n)
        slots    = role_to_slot(rotation)
        pic_out  = IMG_DIR      / f"pic{n}.png"
        disc_tl  = DISCRETE_DIR / f"{n}_{tl}_TL.png"
        disc_tr  = DISCRETE_DIR / f"{n}_{tr}_TR.png"
        disc_br  = DISCRETE_DIR / f"{n}_{br}_BR.png"
        disc_bl  = DISCRETE_DIR / f"{n}_{bl}_BL.png"

        position_log_rows.append({
            "pic":         n,
            "target":      tl,
            "rotation":    rotation,
            "target_slot": slots["TL"],
            "target_x":    OPTION_X[slots["TL"]],
        })

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

        print(f"\npic{n:02d}  {tl}(TL)  {tr}(TR)  {br}(BR)  {bl}(BL)"
              f"   [rotation={rotation}, target→slot{slots['TL']} (x={OPTION_X[slots['TL']]})]")
        print(f"  source:   {src_path.name}")
        print(f"  subject:  {subject_status}")

        if args.no_overwrite and pic_out.exists():
            print(f"  SKIP trial canvas (exists): {pic_out.name}")
            n_skip += 1

        if args.dry_run:
            print(f"  [dry-run] would write discrete → img_discrete/{n}_*.png")
            print(f"  [dry-run] would write trial   → img_composition/pic{n}.png")
            n_ok += 1
            continue

        try:
            collage = Image.open(src_path).convert("RGBA")
            quads   = crop_quadrants(collage)

            # 1. Save discrete quadrant crops
            for key, path in [("TL", disc_tl), ("TR", disc_tr),
                               ("BR", disc_br), ("BL", disc_bl)]:
                quads[key].convert("RGB").save(path)
            print(f"  discrete: {n}_{tl}_TL.png  {n}_{tr}_TR.png  "
                  f"{n}_{br}_BR.png  {n}_{bl}_BL.png")

            # 2. Compose and save trial canvas
            if args.discrete_only:
                print(f"  trial:    SKIPPED (--discrete-only)")
            elif not (args.no_overwrite and pic_out.exists()):
                trial = compose_trial(quads, item_subject_img, rotation=rotation)
                trial.save(pic_out)
                print(f"  trial:    pic{n}.png  ({CANVAS_W}×{CANVAS_H})")

            n_ok += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            n_err += 1

    if not args.dry_run and position_log_rows:
        with open(POSITION_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pic", "target", "rotation", "target_slot", "target_x",
            ])
            writer.writeheader()
            writer.writerows(position_log_rows)

    print(f"\n{'─'*52}")
    print(f"Done.  OK: {n_ok}   Skipped: {n_skip}   Errors: {n_err}")
    print(f"Discrete images : {DISCRETE_DIR}")
    print(f"Trial images    : {IMG_DIR}")
    if not args.dry_run and position_log_rows:
        print(f"Position log    : {POSITION_LOG}")
    if n_subject_missing:
        print(f"Subject images missing: {n_subject_missing}/{len(entries)} "
              f"(expected in {SUBJECTS_DIR}/)")


if __name__ == "__main__":
    main()
