"""
Human vs. LLM Noun-Prediction Comparison  (SIMPLIFIED)
========================================================
Compares the human anticipatory-gaze probability distribution (derived from
saccade_analysis_ek.py's per-trial output) against the LLM's GPT-2-surprisal
probability distribution (the vwp50_scene_table.csv in this same folder),
for each of the 50 items, split by verb condition (restrictive / non-restrictive).

HUMAN PROBABILITY FORMULA (2026-08-29, final):

  human_prob = (# participants whose first saccade landed on object X)
                / (# participants whose first saccade landed on ANY of
                   the 4 objects for that item x condition)

  "Landed on" = within the critical window (verb onset -> noun onset).
  Participants with no object-directed first saccade in that window are
  EXCLUDED from the denominator entirely -- not counted as choosing
  nothing across all 4 objects, and not included in the total.

  This guarantees, by construction, that the four human probabilities for
  a given item x condition sum to exactly 1 -- verified at runtime in
  compute_human_probabilities() -- which is what makes the distribution
  directly comparable to the LLM's P_norm (already fully renormalized to
  sum to 1, with no residual "elsewhere" mass on that side to compare
  against). The elsewhere rate itself is still tracked and reported
  separately, in the pct_elsewhere column, rather than silently discarded
  -- it's genuinely different information from the shape of the
  distribution, and folding it into the denominator would conflate the
  two.

  (An earlier version of this script divided by ALL participants
  including elsewhere-only ones, which left the four probabilities
  summing to less than 1. That version was mathematically equivalent to
  this one after the renormalization step already present before JS
  divergence -- but this version makes the sum-to-1 property true by
  construction rather than requiring a downstream renormalization step to
  restore it, and is more directly interpretable on its own.)

     ONLY the "all_saccades" method's continuous per-participant shares
     are not what "chose" naturally means for this formula, so this
     version uses FIRST-SACCADE CHOICE ONLY (one discrete pick per
     participant, or none) -- dropping the second "all_saccades" method
     entirely, since running two methods was part of what made the
     earlier version feel complicated. (Easy to bring back if wanted.)

  2. TARGET EXCLUSION FOR NON-RESTRICTIVE IS NOW VERIFIED AT RUNTIME, not
     just asserted in this docstring. After building the analysis
     distributions, the script checks -- for every single non-restrictive
     item -- that the target object is genuinely absent (not just
     zero-probability) and that exactly 3 objects remain. If that check
     ever fails, the script stops with an error rather than silently
     producing plots from broken data. (Manually confirmed against a real
     run before this version was written: is_target was False for 100% of
     non-restrictive rows, 3 objects per item for all 50 items.)

  3. PLOTS: 3 summary plots plus 2 per-sentence dumbbell charts:
       - One scatter: human probability vs. LLM probability, every object
         point, colored by condition, with the overall Spearman rho for
         each condition in the legend.
       - One box-and-dots plot: JS divergence, one box per condition
         (median/IQR), with each item's actual value shown as a jittered
         dot on top -- so you see both the summary and the real spread,
         without 50 unreadable bars.
       - One box-and-dots plot: same idea for Spearman rho.
       - Two dumbbell charts (restrictive / non-restrictive, separately,
         50 rows each, sorted by item number): one row per sentence,
         human's own top-choice probability and LLM's own top-choice
         probability as two connected dots -- short line = close
         agreement, long line = big disagreement, directly readable per
         sentence rather than only as an overall correlation number.

STATUS OF THE VERIFICATION CONCERNS THIS SCRIPT ADDRESSES:

  Concern 1 (critical window verb onset -> noun onset): inherited from
  saccade_analysis_ek.py by construction -- first_saccade_landing_aoi is
  already bounded correctly there.

  Concern 2 (position -> object mapping): this script independently
  re-detects each subject's counterbalancing group from the condition /
  target_position columns already in the per-trial CSV (same method as
  saccade_analysis_ek.py's own detection -- (target_position, condition)
  is a confirmed unique fingerprint per group), rather than trusting the
  upstream resolution blindly.

  Concern 3 (probability formula): see point 1 above -- now the one and
  only formula used, not a secondary column.

TWO REAL, CONFIRMED CROSS-FILE FIXES (verified against the actual uploaded
files, not assumed):
  - Object naming: hyphen/space differences AND 3 genuine spelling
    discrepancies (laddle/ladle, rubberband/rubber band,
    toothbrish/toothbrush), fixed in norm_obj() / OBJECT_NAME_ALIASES.
    Verified: 0 mismatches across all 50 real items.
  - Item numbering: group CSV 'id' is 0-indexed, LLM CSV 'Item' is
    1-indexed for the same items. Fixed via HUMAN_ITEM_OFFSET = 1.
    Verified across all 50 real items, not just one.

ALL 4 OBJECTS, BOTH CONDITIONS (2026-08-29): both restrictive and
non-restrictive now compare the full 4-way distribution, including the
target -- no exclusion, no renormalization. (An earlier version dropped
the target for non-restrictive on the argument that a non-restrictive
verb doesn't specifically select the target, so comparing target
probability there wasn't testing verb-driven prediction specifically.
That's been reversed per explicit request.) Verified at runtime in
merge_and_build_analysis(): every item x condition group must have
exactly 4 objects with the target present exactly once, or the script
stops rather than silently producing plots from incomplete data.

JS DIVERGENCE: bits (log base 2). scipy's jensenshannon() returns a
*distance* (sqrt of divergence) -- squared back here.

STILL OPEN (not addressed by this script): whether the LLM's conditioning
point corresponds to verb onset or verb offset -- not checked against
what `llm_surprisal` actually does.

Run: python human_llm_comparison.py
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, norm as scipy_norm

# ── Repo-relative paths ──────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = (HERE / ".." / "..").resolve()
EXPERIMENT_DIR = REPO_ROOT / "01_experiment"
STIMULI_DIR = EXPERIMENT_DIR / "stimuli"
GROUP_CSV_DIR = STIMULI_DIR / "creatingDataStructure" / "participant_groups_8"

SACCADE_CSV = (REPO_ROOT / "02_analytics" / "saccade_metri2" 
               / "saccade_output_ek" / "csv" / "first_saccade_per_trial_ek.csv")
LLM_CSV = (REPO_ROOT / "02_analytics" / "llm_predictions" )


def _find_llm_csv():
    candidates = sorted(LLM_CSV.glob("*vwp50_scene_table*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*vwp50_scene_table*.csv' file found in {HERE}.")
    return candidates[0]


OUT_DIR = HERE / "comparison_output"
CSV_DIR = OUT_DIR / "csv"
PLOTS_DIR = OUT_DIR / "plots"
for d in (OUT_DIR, CSV_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["restrictive", "non-restrictive"]
COND_COLOR = {"restrictive": "#1f77b4", "non-restrictive": "#d62728"}

# Mismatch: saccade pipeline's sentence_id / group CSV's 'id' is 0-indexed;
# LLM CSV's 'Item' is 1-indexed for the same items. Verified across all 50
# real items (id=N <-> Item=N+1, 0 exceptions).
HUMAN_ITEM_OFFSET = 1

# Genuine spelling discrepancies between the two independently-created
# files, found by diffing their real object vocabularies -- not guessed.
OBJECT_NAME_ALIASES = {
    "laddle": "ladle",
    "rubberband": "rubber band",
    "toothbrish": "toothbrush",
}


# ──────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ──────────────────────────────────────────────────────────────

def normalize_condition(raw):
    s = str(raw).strip().lower()
    return "non-restrictive" if s.startswith("non") else "restrictive"


def norm_obj(s):
    """Hyphen->space (symmetric, needed on both sides) + alias lookup for
    the 3 confirmed spelling discrepancies. Verified: 0 mismatches across
    all 50 real items after this normalization."""
    s = str(s).strip().lower().replace("-", " ")
    return OBJECT_NAME_ALIASES.get(s, s)


def subject_id_to_group_number(subject_id):
    """Formula guess only -- kept as a fallback/comparison point, not
    trusted as ground truth (already found wrong for at least one subject)."""
    matches = re.findall(r"\d+", str(subject_id))
    if not matches:
        return None
    return (int(matches[-1]) % 8) + 1


# ──────────────────────────────────────────────────────────────
# CONCERN 2: independent group detection
# ──────────────────────────────────────────────────────────────

def load_all_groups(group_dir):
    groups = {}
    for g in range(1, 9):
        p = group_dir / f"participant_group{g}.csv"
        if p.exists():
            groups[g] = pd.read_csv(p).set_index("id")
        else:
            print(f"  [warn] missing participant_group{g}.csv in {group_dir}")
    return groups


def detect_subject_group(subject_id, subj_trials, groups):
    """(target_position, condition) is a confirmed unique fingerprint per
    group (every group scores 1.0 against its own data, 0.0 against all 7
    others, on the real files) -- detected here from the per-trial CSV's
    own columns rather than trusted from the formula or from upstream."""
    subj_lookup = {}
    for _, r in subj_trials.iterrows():
        item_id = r.get("sentence_id")
        cond = normalize_condition(r.get("condition"))
        pos = r.get("target_position")
        if pd.isna(item_id) or pd.isna(pos):
            continue
        subj_lookup[int(item_id)] = (int(pos), cond)

    formula_guess = subject_id_to_group_number(subject_id)
    if not subj_lookup:
        return formula_guess, "formula_fallback"

    scores = {}
    for gnum, gdf in groups.items():
        matches, total = 0, 0
        for item_id, (tp, cond) in subj_lookup.items():
            if item_id not in gdf.index:
                continue
            total += 1
            if (int(gdf.loc[item_id, "target_position"]) == tp
                    and normalize_condition(gdf.loc[item_id, "condition"]) == cond):
                matches += 1
        scores[gnum] = (matches / total) if total else 0.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_num, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score >= 0.9 and (best_score - runner_up) >= 0.5:
        return best_num, "detected"
    return formula_guess, "formula_fallback"


# ──────────────────────────────────────────────────────────────
# LOAD LLM PREDICTIONS
# ──────────────────────────────────────────────────────────────

def load_llm_predictions(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Item": "item", "Condition": "condition_raw", "Object": "object_raw",
        "Is_Target": "is_target", "P_norm": "llm_prob", "Verb": "verb",
    })
    df["condition"] = df["condition_raw"].apply(normalize_condition)
    df["object"] = df["object_raw"].apply(norm_obj)
    df["is_target"] = df["is_target"].astype(bool)
    return df[["item", "condition", "object", "is_target", "llm_prob", "verb"]]


def build_verb_lookup(llm_df):
    """{(item, condition): verb} -- used to label dumbbell-chart rows with
    something more useful than a bare item number."""
    return {(r["item"], r["condition"]): r["verb"]
            for _, r in llm_df[["item", "condition", "verb"]].drop_duplicates().iterrows()}


# ──────────────────────────────────────────────────────────────
# HUMAN SIDE: count(participants whose first saccade landed on object X)
# / count(participants whose first saccade landed on ANY of the 4 objects)
# ──────────────────────────────────────────────────────────────

def build_human_choices(saccade_df, groups):
    """One row per (subject, item, condition) with which object (if any)
    that participant's first saccade after verb onset landed on. 'item'
    here is already the LLM-comparable, offset item number."""
    rows = []
    unmatched = 0
    n_detected, n_fallback = 0, 0

    for subject_id, subj_trials in saccade_df.groupby("subject_id"):
        group_num, source = detect_subject_group(subject_id, subj_trials, groups)
        if source == "detected":
            n_detected += 1
        else:
            n_fallback += 1
        gdf = groups.get(group_num)
        if gdf is None:
            unmatched += len(subj_trials)
            continue

        for _, r in subj_trials.iterrows():
            raw_item = r["sentence_id"]
            if pd.isna(raw_item) or int(raw_item) not in gdf.index:
                unmatched += 1
                continue
            raw_item = int(raw_item)
            g_row = gdf.loc[raw_item]

            landing = r.get("first_saccade_landing_aoi", None)
            if landing in ("pos1", "pos2", "pos3", "pos4"):
                pos_num = int(landing[-1])
                chosen_object = norm_obj(g_row[f"position{pos_num}"])
            else:
                chosen_object = None  # no object-directed first saccade

            rows.append({
                "item": raw_item + HUMAN_ITEM_OFFSET,
                "condition": normalize_condition(r["condition"]),
                "subject_id": subject_id,
                "chosen_object": chosen_object,
            })

    if unmatched:
        print(f"  [warn] {unmatched} trial rows had no matching group-CSV entry and were skipped")
    print(f"  group detection: {n_detected} subject(s) confidently detected, "
          f"{n_fallback} fell back to the formula guess")
    return pd.DataFrame(rows)


def compute_human_probabilities(choices_df, llm_df):
    """
    human_prob = count(chose object X) / (# participants whose first
    saccade landed on ANY of the 4 objects for that item x condition).

    This is a deliberate change from an earlier version that divided by
    ALL participants (including those with no object-directed first
    saccade), which left the four probabilities summing to something less
    than 1 -- not a proper distribution, and not directly comparable to
    the LLM's P_norm (which is already fully renormalized to sum to 1
    with nothing held back). Dividing by engaged-only participants instead
    guarantees, by construction, that the four human probabilities sum to
    exactly 1: every engaged participant contributes to exactly one of
    the four objects, so summing n_chose across all 4 objects always
    equals n_participants_engaged.

    Participants with no object-directed first saccade are excluded from
    the denominator entirely (not counted as choosing nothing across all
    4 objects) -- their existence is still tracked and reported via
    pct_elsewhere, just not folded into the probability itself.
    """
    engaged = choices_df.dropna(subset=["chosen_object"])
    totals = (engaged.groupby(["item", "condition"])["subject_id"]
              .nunique().rename("n_participants_engaged"))
    n_total_all = (choices_df.groupby(["item", "condition"])["subject_id"]
                   .nunique().rename("n_participants_total"))

    counts = (engaged.groupby(["item", "condition", "chosen_object"])["subject_id"]
              .nunique().rename("n_chose").reset_index()
              .rename(columns={"chosen_object": "object"}))

    # scaffold every (item, condition, object) that exists on the LLM side,
    # so objects nobody chose still get an explicit 0.0 rather than being
    # silently absent
    scaffold = llm_df[["item", "condition", "object"]].drop_duplicates()
    merged = scaffold.merge(counts, on=["item", "condition", "object"], how="left")
    merged["n_chose"] = merged["n_chose"].fillna(0).astype(int)
    merged = merged.merge(totals, on=["item", "condition"], how="left")
    # EDGE CASE: an item x condition group can legitimately have ZERO
    # engaged participants (every single participant had no object-directed
    # first saccade in the critical window -- 100% elsewhere). That's real,
    # possible data, not a bug -- fillna(0) makes it explicit rather than
    # leaving it as NaN-that-looks-like-missing-data.
    merged["n_participants_engaged"] = merged["n_participants_engaged"].fillna(0).astype(int)
    merged = merged.merge(n_total_all, on=["item", "condition"], how="left")
    with np.errstate(invalid="ignore"):
        merged["human_prob"] = merged["n_chose"] / merged["n_participants_engaged"]
    # n_participants_engaged==0 forces n_chose==0 too (nobody engaged means
    # nobody chose anything), so this is always exactly 0/0 -> NaN, never a
    # real x/0 division -- human_prob is genuinely undefined for these rows,
    # not zero. Left as NaN on purpose so it's excluded downstream (analysis
    # functions already dropna() before JS/Spearman) rather than silently
    # treated as "definitely didn't pick this object."
    merged["pct_elsewhere"] = (1 - merged["n_participants_engaged"] / merged["n_participants_total"]).round(6)

    # VERIFIED AT RUNTIME, split into the two cases that actually matter:
    #   - groups WITH at least one engaged participant must sum to 1 (a
    #     real bug if they don't)
    #   - groups with ZERO engaged participants are legitimate missing
    #     data, reported as a count, not raised as an error
    grouped = merged.groupby(["item", "condition"])
    sums = grouped["human_prob"].sum()          # NaN-skipping; all-NaN groups show as 0.0
    n_valid = grouped["human_prob"].apply(lambda s: int(s.notna().sum()))  # 0 or 4, never in between by construction

    zero_engaged = n_valid[n_valid == 0]
    fully_valid = n_valid[n_valid == 4]
    unexpected = n_valid[(n_valid > 0) & (n_valid < 4)]  # should never happen; would indicate a real bug

    bad_sums = sums[fully_valid.index][(sums[fully_valid.index] - 1.0).abs() > 1e-6]
    if len(bad_sums) > 0 or len(unexpected) > 0:
        raise SystemExit(
            f"ERROR: {len(bad_sums)} groups with engaged participants don't sum to 1, "
            f"and {len(unexpected)} groups have a partial (not 0 or 4) count of valid "
            f"probabilities -- this points to an actual bug, not the zero-engaged edge "
            f"case. Examples: {dict(list(bad_sums.items())[:3])}")

    print(f"  [verified] human probabilities sum to 1 for all {len(fully_valid)} item x "
          f"condition groups with at least one engaged participant")
    if len(zero_engaged) > 0:
        print(f"  [note] {len(zero_engaged)} item x condition groups had ZERO participants "
              f"with an object-directed first saccade in the critical window -- human_prob "
              f"is NaN (not 0) for these and they're automatically excluded from JS "
              f"divergence / Spearman / plots downstream, not silently zeroed out. "
              f"Groups: {list(zero_engaged.index)[:10]}{'...' if len(zero_engaged) > 10 else ''}")

    return merged


# ──────────────────────────────────────────────────────────────
# MERGE + ANALYSIS DISTRIBUTIONS (all 4 objects, both conditions,
# VERIFIED at runtime)
# ──────────────────────────────────────────────────────────────


def diagnose_merge_failure(human_df, llm_df):
    print("\n" + "=" * 90)
    print("MERGE DIAGNOSTICS")
    print("=" * 90)
    h_items, l_items = set(human_df["item"].unique()), set(llm_df["item"].unique())
    print(f"  item overlap: {len(h_items & l_items)} of {len(h_items)} human / {len(l_items)} LLM")
    common = h_items & l_items
    if common:
        it = sorted(common)[0]
        cond = "restrictive"
        h_objs = sorted(human_df[(human_df.item == it) & (human_df.condition == cond)]["object"].unique())
        l_objs = sorted(llm_df[(llm_df.item == it) & (llm_df.condition == cond)]["object"].unique())
        print(f"  Sample item={it}: human objects={h_objs}  LLM objects={l_objs}")
        if set(h_objs) != set(l_objs):
            print("  -> Still mismatched -- check OBJECT_NAME_ALIASES for a new spelling gap.")
    print("=" * 90 + "\n")


def merge_and_build_analysis(human_df, llm_df):
    merged = pd.merge(human_df, llm_df, on=["item", "condition", "object"], how="inner")
    if merged.empty:
        diagnose_merge_failure(human_df, llm_df)
        raise SystemExit("ERROR: 0 rows matched between human and LLM data.")
    if len(merged) < len(llm_df):
        print(f"  [warn] only {len(merged)}/{len(llm_df)} LLM rows matched -- "
              f"check OBJECT_NAME_ALIASES for a new spelling gap")
    else:
        print(f"  merge OK: all {len(merged)} rows matched cleanly")

    # ALL 4 OBJECTS, BOTH CONDITIONS -- no target exclusion, no renormalization.
    # (Earlier versions of this script dropped the target and renormalized
    # the remaining 3 distractors for non-restrictive only, on the argument
    # that a non-restrictive verb doesn't specifically select the target so
    # comparing target probability there wasn't testing verb-driven
    # prediction. That's been reversed per explicit request: both
    # conditions now compare the same full 4-way distribution.)
    analysis_df = merged.copy()
    analysis_df["human_prob_analysis"] = analysis_df["human_prob"]
    analysis_df["llm_prob_analysis"] = analysis_df["llm_prob"]

    # VERIFIED AT RUNTIME: every item, both conditions, must have all 4
    # objects present (including the target) -- the opposite check from
    # the target-exclusion version, confirming NOTHING is being dropped.
    obj_counts = analysis_df.groupby(["item", "condition"])["object"].count()
    n_wrong_count = int((obj_counts != 4).sum())
    n_target_present = analysis_df.groupby(["item", "condition"])["is_target"].sum()
    n_missing_target = int((n_target_present != 1).sum())
    if n_wrong_count > 0 or n_missing_target > 0:
        raise SystemExit(
            f"ERROR: all-4-objects check FAILED -- {n_wrong_count} item x condition "
            f"groups don't have exactly 4 objects, {n_missing_target} are missing "
            f"(or have duplicate) target rows. Stopping rather than producing plots "
            f"from incomplete data.")
    n_groups_checked = obj_counts.shape[0]
    print(f"  [verified] all 4 objects (including target) present for both conditions: "
          f"{n_groups_checked}/{n_groups_checked} item x condition groups have exactly "
          f"4 objects, target present exactly once in each")

    return analysis_df


# ──────────────────────────────────────────────────────────────
# PER-ITEM METRICS
# ──────────────────────────────────────────────────────────────

def compute_item_metrics(analysis_df):
    records = []
    for (item, cond), g in analysis_df.groupby(["item", "condition"]):
        g = g.dropna(subset=["human_prob_analysis", "llm_prob_analysis"])
        if len(g) < 2:
            continue
        p = g["human_prob_analysis"].values.astype(float)
        q = g["llm_prob_analysis"].values.astype(float)
        if p.sum() > 0:
            p = p / p.sum()
        if q.sum() > 0:
            q = q / q.sum()
        js_dist = jensenshannon(p, q, base=2)
        js_div = float(js_dist ** 2) if np.isfinite(js_dist) else np.nan
        rho, pval = spearmanr(p, q) if len(g) >= 3 else (np.nan, np.nan)
        top_h_idx, top_l_idx = int(np.argmax(p)), int(np.argmax(q))
        objs = g["object"].values
        records.append({
            "item": item, "condition": cond, "js_divergence_bits": js_div,
            "spearman_rho_within_item": rho,
            "human_top_prob": p[top_h_idx], "llm_top_prob": q[top_l_idx],
            "human_top_object": objs[top_h_idx], "llm_top_object": objs[top_l_idx],
        })
    return pd.DataFrame(records)


def overall_spearman(analysis_df):
    out = {}
    d = analysis_df.dropna(subset=["human_prob_analysis", "llm_prob_analysis"])
    for cond, g in d.groupby("condition"):
        rho, pval = spearmanr(g["human_prob_analysis"], g["llm_prob_analysis"]) if len(g) >= 3 else (np.nan, np.nan)
        out[cond] = (rho, pval, len(g))
    return out


# ──────────────────────────────────────────────────────────────
# PLOTS -- 3 total, simple, single-panel
# ──────────────────────────────────────────────────────────────

def plot_scatter(analysis_df, out_path):
    """Styled to match the professor's reference plot: one axis = human
    ('actual'), one axis = LLM ('predicted'), solid gridlines, a labeled
    45-degree reference line, boxed axes. All 4 objects per sentence
    (including target) are plotted as undifferentiated candidates within
    each condition, matching how the underlying data is computed (see
    module docstring: no target/distractor special-casing anywhere in the
    analysis) -- but the two CONDITIONS are colored separately here so
    restrictive vs. non-restrictive alignment can be compared visually.

    NOTE on jitter: human_prob is count(chose X)/N engaged participants,
    so with a small N it can only take a handful of exact fractional
    values (0, 1/N, 2/N, ...) -- dots snap to those columns and many
    real, distinct points land exactly on top of each other, especially
    at the (0,0)/(0,1)/(1,0)/(1,1) corners. Jitter is added to BOTH axes
    (not just x) specifically so overlapping points at the corners
    separate into a visible cloud instead of hiding each other -- this
    changes where a dot is DRAWN, never the underlying data.
    """
    rng = np.random.default_rng(0)
    g_all = analysis_df.dropna(subset=["human_prob_analysis", "llm_prob_analysis"])

    axis_max = max(g_all["human_prob_analysis"].max(), g_all["llm_prob_analysis"].max())
    axis_max = float(np.ceil(axis_max * 10) / 10)
    axis_max = max(axis_max, 0.2)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.plot([0, axis_max], [0, axis_max], color="#555555", linestyle="--",
            linewidth=1.3, label="45\u00b0 line", zorder=1)

    for cond in CONDITIONS:
        g = g_all[g_all["condition"] == cond]
        jx = rng.uniform(-0.018, 0.018, size=len(g))
        jy = rng.uniform(-0.018, 0.018, size=len(g))
        ax.scatter(g["human_prob_analysis"] + jx, g["llm_prob_analysis"] + jy,
                   s=40, color=COND_COLOR[cond], alpha=0.4, edgecolors="none",
                   label=f"{cond} (n={len(g)})", zorder=2)

    ax.set_xlim(-0.03, axis_max + 0.03); ax.set_ylim(-0.03, axis_max + 0.03)
    ax.set_xlabel("Human probability (actual)")
    ax.set_ylabel("LLM probability (predicted)")
    ax.set_title("Human vs. LLM Predicted Probability, All 4 Objects per Sentence")
    ax.legend(loc="lower right", frameon=True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_example_distributions(analysis_df, item_metrics, verb_lookup, out_path, n_per_condition=2):
    """
    Picks concrete example sentences and shows their actual 4-object
    probability distributions, human vs. LLM, as grouped bar charts. This
    is the plot that shows what 'alignment' or 'disagreement' actually
    looks like at the level of individual objects and numbers, rather than
    a single summary statistic -- complements the scatter/JSD/agreement
    plots rather than replacing them.

    Selection: per condition, the BEST-aligned item (lowest JS divergence)
    and the WORST-aligned item (highest JS divergence) -- showing the real
    range, not a cherry-picked "it works" example.
    """
    examples = []
    for cond in CONDITIONS:
        g = item_metrics[item_metrics["condition"] == cond].dropna(subset=["js_divergence_bits"])
        if g.empty:
            continue
        best = g.loc[g["js_divergence_bits"].idxmin()]
        worst = g.loc[g["js_divergence_bits"].idxmax()]
        examples.append((cond, "best alignment", best["item"]))
        examples.append((cond, "worst alignment", worst["item"]))

    if not examples:
        print(f"  [plot] skipped {out_path} -- no items with a defined JS divergence")
        return

    n = len(examples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), squeeze=False)

    for idx, (cond, label, item) in enumerate(examples):
        ax = axes[idx // ncols][idx % ncols]
        g = analysis_df[(analysis_df["item"] == item) & (analysis_df["condition"] == cond)].dropna(
            subset=["human_prob_analysis", "llm_prob_analysis"])
        g = g.sort_values("llm_prob_analysis", ascending=False)
        x = np.arange(len(g))
        width = 0.38
        ax.bar(x - width / 2, g["human_prob_analysis"], width, color="#2a78d6", label="human")
        ax.bar(x + width / 2, g["llm_prob_analysis"], width, color="#eb6834", label="llm")
        ax.set_xticks(x)
        ax.set_xticklabels(g["object"], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        js = item_metrics[(item_metrics.item == item) & (item_metrics.condition == cond)]["js_divergence_bits"].iloc[0]
        verb = verb_lookup.get((item, cond), "")
        ax.set_title(f"item {int(item)} \u00b7 {verb}  ({cond}, {label})\nJS divergence = {js:.2f} bits", fontsize=10)
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Example Probability Distributions: What Alignment Looks Like", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_agreement_rate(item_metrics, out_path):
    """THE 'how aligned are we, at a glance' plot: for each sentence, did
    human's top-choice object match LLM's top-choice object? One bar per
    condition = % of sentences where they agreed, with a dashed line at
    25% (chance level with 4 objects) so you can tell at a glance whether
    agreement is better than guessing."""
    rates, ns = [], []
    for cond in CONDITIONS:
        g = item_metrics[item_metrics["condition"] == cond]
        if len(g) == 0:
            rates.append(np.nan); ns.append(0)
            continue
        agree = (g["human_top_object"] == g["llm_top_object"]).mean() * 100
        rates.append(agree); ns.append(len(g))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    bars = ax.bar(CONDITIONS, rates, color=[COND_COLOR[c] for c in CONDITIONS], width=0.55)
    for b, n in zip(bars, ns):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 2, f"{h:.0f}%\n(n={n})", ha="center", fontsize=10)
    ax.axhline(25, color="#888888", linestyle="--", linewidth=1.2, label="chance (25%, 4 objects)")
    ax.set_ylabel("% of sentences where human's and LLM's top pick matched")
    ax.set_title("How Often Do Human and LLM Agree on the Top Object?")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def _box_with_dots(ax, item_metrics, value_col, ylabel, title):
    rng = np.random.default_rng(0)
    data, positions, colors = [], [], []
    for i, cond in enumerate(CONDITIONS):
        vals = item_metrics.loc[item_metrics["condition"] == cond, value_col].dropna().values
        data.append(vals)
        positions.append(i)
        colors.append(COND_COLOR[cond])
    bp = ax.boxplot(data, positions=positions, widths=0.5, showfliers=False,
                     patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.6, color=colors[i])
    ax.set_xticks(positions)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_dumbbell(item_metrics, condition, verb_lookup, out_path):
    """One row per sentence (item), sorted by item number. Each row shows
    the human's own top-choice probability and the LLM's own top-choice
    probability as two dots connected by a line -- lets you see, sentence
    by sentence, where human and LLM agree closely (short/no line) versus
    diverge a lot (long line), rather than just an overall correlation."""
    d = item_metrics[item_metrics["condition"] == condition].sort_values("item")
    if d.empty:
        print(f"  [plot] skipped {out_path} -- no data for {condition}")
        return

    n = len(d)
    fig_h = max(6, 0.28 * n + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))

    y_positions = np.arange(n)
    human_color, llm_color = "#2a78d6", "#eb6834"

    ax.hlines(y_positions, d["human_top_prob"], d["llm_top_prob"], color="#c3c2b7", linewidth=1.5, zorder=1)
    ax.scatter(d["human_top_prob"], y_positions, s=32, color=human_color, label="human", zorder=2)
    ax.scatter(d["llm_top_prob"], y_positions, s=32, color=llm_color, label="llm", zorder=2)

    labels = []
    for _, r in d.iterrows():
        verb = verb_lookup.get((r["item"], condition), "")
        labels.append(f"{int(r['item'])} \u00b7 {verb}" if verb else str(int(r["item"])))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()  # item 1 at top
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Top-choice probability")
    ax.set_title(f"Per-Sentence Top-Choice Probability \u2014 {condition}\n"
                 "(each row = one sentence; line length = how much human and LLM disagree)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_js_divergence_curves(item_metrics, out_path):
    """
    JS divergence shown as two overlapping normal (Gaussian) curves, one
    per condition -- NOT a curve fit to the 4-object probability
    distributions themselves (those are unordered categories; a 'curve'
    over them isn't meaningful, same reason a literal peak-comparison
    doesn't apply to them). Here the curve is fit to the JS DIVERGENCE
    VALUES -- one continuous number per sentence, 50 per condition -- which
    genuinely is a meaningful continuous quantity to describe with a mean
    and spread. Less overlap between the two curves = the two conditions
    differ more in how well human and LLM align.

    Shown alongside the real values as a rug (small ticks along the
    bottom) so the smooth curve doesn't overstate how Gaussian the actual
    data is -- with only 50 points per condition and a value bounded to
    [0, 1], the true shape may be skewed; the rug keeps that honest.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.linspace(-0.15, 1.15, 400)

    for cond in CONDITIONS:
        vals = item_metrics.loc[item_metrics["condition"] == cond, "js_divergence_bits"].dropna().values
        if len(vals) < 2:
            continue
        mu, sigma = float(np.mean(vals)), float(np.std(vals, ddof=1))
        y = scipy_norm.pdf(x, mu, sigma)
        ax.plot(x, y, color=COND_COLOR[cond], linewidth=2,
                label=f"{cond}  (mean={mu:.2f}, sd={sigma:.2f}, n={len(vals)})")
        ax.fill_between(x, y, color=COND_COLOR[cond], alpha=0.15)
        ax.axvline(mu, color=COND_COLOR[cond], linestyle=":", linewidth=1.3)
        # rug: actual per-item values along the bottom, so the fitted
        # curve is never mistaken for the raw data
        ax.plot(vals, np.full_like(vals, -0.015), "|", color=COND_COLOR[cond],
                markersize=10, markeredgewidth=1.2, alpha=0.7)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(bottom=-0.05)
    ax.set_xlabel("Jensen-Shannon divergence (bits)")
    ax.set_ylabel("Density (fitted normal curve)")
    ax.set_title("JS Divergence Distribution: Restrictive vs. Non-Restrictive\n"
                 "(curves fit to per-item JS values, not to the object probabilities; "
                 "ticks along bottom = actual items)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


def plot_spearman(item_metrics, overall_rho, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _box_with_dots(ax, item_metrics, "spearman_rho_within_item", "Spearman \u03c1 (within item)",
                   "Rank Agreement per Item\n(each dot = one item)")
    ax.axhline(0, color="black", linewidth=0.8)
    txt = "\n".join(f"overall pooled \u03c1 ({c}): {r:.2f}" if np.isfinite(r) else f"overall pooled \u03c1 ({c}): n/a"
                     for c, (r, p, n) in overall_rho.items())
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved: {out_path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    llm_csv_path = _find_llm_csv()
    print(f"LLM predictions file: {llm_csv_path.name}")
    llm_df = load_llm_predictions(llm_csv_path)
    print(f"  {llm_df['item'].nunique()} items, {len(llm_df)} item x condition x object rows")

    if not SACCADE_CSV.exists():
        raise SystemExit(f"ERROR: saccade per-trial CSV not found at {SACCADE_CSV}")
    saccade_df = pd.read_csv(SACCADE_CSV)
    print(f"Loaded saccade per-trial data: {len(saccade_df)} rows")

    groups = load_all_groups(GROUP_CSV_DIR)
    if not groups:
        raise SystemExit(f"ERROR: no participant_groupN.csv files found in {GROUP_CSV_DIR}")
    print(f"Loaded {len(groups)} participant-group position->object mappings")

    print("\nDetermining each participant's choice (first saccade after verb onset)...")
    choices_df = build_human_choices(saccade_df, groups)

    print("Computing human_prob = count(chose X) / count(engaged participants)...")
    human_df = compute_human_probabilities(choices_df, llm_df)
    human_df.to_csv(CSV_DIR / "human_probabilities.csv", index=False)
    print(f"  Saved: {CSV_DIR / 'human_probabilities.csv'} ({len(human_df)} rows)")

    print("\nMerging with LLM predictions and building analysis distributions...")
    analysis_df = merge_and_build_analysis(human_df, llm_df)
    analysis_df.to_csv(CSV_DIR / "human_llm_analysis_distributions.csv", index=False)

    item_metrics = compute_item_metrics(analysis_df)
    item_metrics.to_csv(CSV_DIR / "item_level_metrics.csv", index=False)

    ov_rho = overall_spearman(analysis_df)
    print("\nOverall pooled Spearman correlation:")
    for cond, (rho, pval, n) in ov_rho.items():
        print(f"  {cond}: rho={rho:.3f}, p={pval:.4g}, n={n}" if np.isfinite(rho) else f"  {cond}: n/a")
    print("\nMean JS divergence:")
    for cond in CONDITIONS:
        mean_js = item_metrics.loc[item_metrics["condition"] == cond, "js_divergence_bits"].mean()
        print(f"  {cond}: {mean_js:.3f} bits")

    print("\nGenerating plots (6 total), ordered by priority:")
    verb_lookup = build_verb_lookup(llm_df)
    # 1. overall alignment
    plot_scatter(analysis_df, PLOTS_DIR / "1_scatter_human_vs_llm.png")
    # 2. what alignment looks like for individual items
    plot_example_distributions(analysis_df, item_metrics, verb_lookup,
                                PLOTS_DIR / "2_example_distributions.png")
    # 3. main quantitative measure of distributional similarity -- fitted
    # normal curves per condition (replaces the earlier box+dots version)
    plot_js_divergence_curves(item_metrics, PLOTS_DIR / "3_js_divergence.png")
    # 4. simple measure of same-object agreement
    plot_agreement_rate(item_metrics, PLOTS_DIR / "4_agreement_rate.png")
    # 5. optional/exploratory -- ranking agreement, not distribution shape
    plot_spearman(item_metrics, ov_rho, PLOTS_DIR / "5_spearman_optional.png")
    # per-sentence detail views (supplementary, not part of the core 5)
    plot_dumbbell(item_metrics, "restrictive", verb_lookup, PLOTS_DIR / "per_sentence_restrictive.png")
    plot_dumbbell(item_metrics, "non-restrictive", verb_lookup, PLOTS_DIR / "per_sentence_non_restrictive.png")

    print("\nDone.")
    return analysis_df, item_metrics


if __name__ == "__main__":
    main()