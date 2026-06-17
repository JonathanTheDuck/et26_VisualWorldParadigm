#!/usr/bin/env python3
"""
collage_status.py
─────────────────
Status dashboard for all 50 VWP stimuli.

Reads sentences.csv and cross-checks against img_matrices/ and img/
to show which pics are fully done, which need a ChatGPT collage,
and the suggested filename to use when generating each missing collage.

Usage
─────
  python collage_status.py           # full status table
  python collage_status.py --missing # show only missing collages
  python collage_status.py --names   # print suggested collage filenames only
"""

import argparse
import csv
import re
from pathlib import Path

STIMULI_DIR  = Path(__file__).parent
SENTENCES    = STIMULI_DIR / "creatingDataStructure" / "sentences.csv"
MATRICES_DIR = STIMULI_DIR / "img_matrices"
IMG_DIR      = STIMULI_DIR / "img_composition"


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_slug(text: str) -> str:
    """'washing machine' → 'washing-machine'  (spaces→hyphens, stripped)"""
    return text.strip().lower().replace(" ", "-")


def extract_target(sentence: str) -> str:
    """
    Extract target noun from sentence1.
    'The boy will eat the cake'  → 'cake'
    'The man will repair the washing machine' → 'washing-machine'
    Strategy: take everything after the LAST ' the ' (case-insensitive).
    """
    parts = re.split(r" [Tt]he ", sentence.strip())
    raw = parts[-1].strip().rstrip(".,;")
    return to_slug(raw)


def load_sentences() -> list[dict]:
    """Parse sentences.csv into a list of dicts (one per sentence)."""
    entries = []
    with open(SENTENCES, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            sid     = int(row[0].strip())
            sent1   = row[1].strip() if len(row) > 1 else ""
            sent2   = row[2].strip() if len(row) > 2 else ""
            # columns 3-6: ot1 ot2 oot3 0t4 (optional)
            dists_raw = [row[i].strip() for i in range(3, min(7, len(row)))
                         if row[i].strip()]
            dists = [to_slug(d) for d in dists_raw]
            entries.append({
                "id":     sid,
                "pic":    sid + 1,
                "sent1":  sent1,
                "sent2":  sent2,
                "dists":  dists,         # all listed distractors (up to 4)
            })
    return entries


def find_matrix_file(pic_n: int) -> Path | None:
    """Return the img_matrices PNG for pic{N} if it exists (any object names)."""
    matches = list(MATRICES_DIR.glob(f"{pic_n}_*.png"))
    return matches[0] if matches else None


def suggest_filename(target: str, dists: list[str]) -> str:
    """
    Suggest the img_matrix filename to use when asking ChatGPT.
    Uses first 3 distractors (the 4th, if any, is typically excluded from the 2×2 grid).
    Multi-word objects should use hyphens (already converted by to_slug).
    """
    used_dists = dists[:3]
    parts = [target] + used_dists
    joined = "_".join(parts)
    return f"??_REPLACE_WITH_PIC_N_{joined}.png"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missing", action="store_true",
                    help="Show only pics that still need a ChatGPT collage")
    ap.add_argument("--names", action="store_true",
                    help="Print suggested collage filenames for missing pics only")
    args = ap.parse_args()

    entries = load_sentences()

    n_done = n_partial = n_missing = 0

    if not args.names:
        print(f"{'pic':<6} {'matrix exists?':<18} {'img exists?':<14} "
              f"{'target':<22} distractors")
        print("─" * 90)

    for e in entries:
        n       = e["pic"]
        target  = extract_target(e["sent1"])
        dists   = e["dists"]

        matrix  = find_matrix_file(n)
        img_ok  = (IMG_DIR / f"pic{n}.png").exists()

        if matrix:
            matrix_label = f"✓  {matrix.name[:28]}"
        else:
            matrix_label = "✗  (missing)"

        img_label = "✓  pic{}.png".format(n) if img_ok else "✗  (missing)"

        status = "DONE" if matrix else ("img only" if img_ok else "MISSING")

        if matrix:
            n_done += 1
        elif img_ok:
            n_partial += 1
        else:
            n_missing += 1

        if args.missing and matrix:
            continue   # skip already done

        if args.names:
            if not matrix:
                used = dists[:3]
                parts = "_".join([target] + used)
                print(f"{n}_{parts}.png")
            continue

        print(f"pic{n:<4} {matrix_label:<36} {img_label:<18} "
              f"{target:<22} {', '.join(dists)}")

    if not args.names:
        print("─" * 90)
        print(f"  Done (matrix + img): {n_done}")
        print(f"  Img only, no matrix: {n_partial}")
        print(f"  Fully missing:       {n_missing}")
        print()
        if n_partial or n_missing:
            print("To see suggested ChatGPT filenames for missing collages:")
            print("  python collage_status.py --names")


if __name__ == "__main__":
    main()
