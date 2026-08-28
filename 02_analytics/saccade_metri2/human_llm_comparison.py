"""
Human vs. LLM Noun-Prediction Comparison  (SIMPLIFIED)
========================================================
Compares the human anticipatory-gaze probability distribution (derived from
saccade_analysis_ek.py's per-trial output) against the LLM's GPT-2-surprisal
probability distribution (the vwp50_scene_table.csv in this same folder),
for each of the 50 items, split by verb condition (restrictive / non-restrictive).

WHAT CHANGED IN THIS SIMPLIFIED VERSION:

  1. ONE probability formula, exactly as specified:
         human_prob = (# participants who chose object X)
                       / (total # participants who had that item x condition)
     "Chose" = that participant's first saccade after verb onset landed on
     object X within the critical window (verb onset -> noun onset).
     Participants with no object-directed saccade contribute 0 to every
     object's count (they're still in the denominator) -- NOT excluded.
     This is a genuine change from the earlier version, which defaulted to
     a "conditional on engagement" mean instead and only reported this
     formula as a secondary column. It's now the only human_prob.

     WHY THIS DOESN'T CHANGE THE JS/SPEARMAN NUMBERS: this formula sums to
     LESS than 1 across the 4 objects whenever some participants had no
     object-directed saccade (the gap is the elsewhere rate). Comparing an
     under-1 vector against the LLM's fully-renormalized (sums-to-1)
     P_norm requires renormalizing first -- which this script already does
     right before computing JS divergence / Spearman. Renormalizing
     (count/total) to sum to 1 is mathematically IDENTICAL to computing
     (count/engaged-only) directly -- verified numerically (renormalizing
     your formula reproduces the conditional version exactly, and both
     give the same JS divergence against a test LLM distribution). So the
     switch is about using the formula you specified throughout, with
     nothing hidden behind an alternate default -- it does not change any
     divergence or correlation number from before.

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

  3. PLOTS CUT FROM 9 DOWN TO 3, all single-panel, no per-item x-axis with
     50 tiny labels:
       - One scatter: human probability vs. LLM probability, every object
         point, colored by condition, with the overall Spearman rho for
         each condition in the legend.
       - One box-and-dots plot: JS divergence, one box per condition
         (median/IQR), with each item's actual value shown as a jittered
         dot on top -- so you see both the summary and the real spread,
         without 50 unreadable bars.
       - One box-and-dots plot: same idea for Spearman rho.

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

NON-RESTRICTIVE: target dropped, remaining 3 distractors renormalized to
sum to 1 independently on the human side and the LLM side (see point 2
above for the runtime verification of this).

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
from scipy.stats import spearmanr

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
        "Is_Target": "is_target", "P_norm": "llm_prob",
    })
    df["condition"] = df["condition_raw"].apply(normalize_condition)
    df["object"] = df["object_raw"].apply(norm_obj)
    df["is_target"] = df["is_target"].astype(bool)
    return df[["item", "condition", "object", "is_target", "llm_prob"]]


# ──────────────────────────────────────────────────────────────
# HUMAN SIDE: exactly the formula requested --
# count(participants whose first saccade landed on object X)
# / total participants for that item x condition
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
    human_prob = count(chose object X) / total participants for that
    item x condition -- exactly as requested. Every (item, condition,
    object) combination present in the LLM data gets a row, including
    0.0 where nobody chose that object.
    """
    totals = choices_df.groupby(["item", "condition"])["subject_id"].nunique().rename("n_participants_total")

    counts = (choices_df.dropna(subset=["chosen_object"])
              .groupby(["item", "condition", "chosen_object"])["subject_id"]
              .nunique().rename("n_chose").reset_index()
              .rename(columns={"chosen_object": "object"}))

    # scaffold every (item, condition, object) that exists on the LLM side,
    # so objects nobody chose still get an explicit 0.0 rather than being
    # silently absent
    scaffold = llm_df[["item", "condition", "object"]].drop_duplicates()
    merged = scaffold.merge(counts, on=["item", "condition", "object"], how="left")
    merged["n_chose"] = merged["n_chose"].fillna(0).astype(int)
    merged = merged.merge(totals, on=["item", "condition"], how="left")
    merged["human_prob"] = merged["n_chose"] / merged["n_participants_total"]

    elsewhere = (1 - merged.groupby(["item", "condition"])["human_prob"].transform("sum"))
    merged["pct_elsewhere"] = elsewhere.round(6)
    return merged


# ──────────────────────────────────────────────────────────────
# MERGE + ANALYSIS DISTRIBUTIONS (target excluded + renormalized for
# non-restrictive, VERIFIED at runtime -- concern 1)
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

    frames = []
    for (item, cond), g in merged.groupby(["item", "condition"]):
        g = g.copy()
        if cond == "non-restrictive":
            g = g[~g["is_target"]].copy()
            for col, out_col in [("human_prob", "human_prob_analysis"), ("llm_prob", "llm_prob_analysis")]:
                total = g[col].sum()
                g[out_col] = g[col] / total if (np.isfinite(total) and total > 1e-6) else np.nan
        else:
            g["human_prob_analysis"] = g["human_prob"]
            g["llm_prob_analysis"] = g["llm_prob"]
        frames.append(g)
    analysis_df = pd.concat(frames, ignore_index=True)

    # CONCERN 1, VERIFIED AT RUNTIME: target must be genuinely absent from
    # every non-restrictive item, and exactly 3 objects must remain.
    nonrestr = analysis_df[analysis_df["condition"] == "non-restrictive"]
    n_target_leaked = int(nonrestr["is_target"].sum())
    obj_counts = nonrestr.groupby("item")["object"].count()
    n_wrong_count = int((obj_counts != 3).sum())
    if n_target_leaked > 0 or n_wrong_count > 0:
        raise SystemExit(
            f"ERROR: target-exclusion check FAILED for non-restrictive condition -- "
            f"{n_target_leaked} target rows leaked through, {n_wrong_count} items don't "
            f"have exactly 3 objects. Stopping rather than producing plots from broken data.")
    n_items_checked = obj_counts.shape[0]
    print(f"  [verified] target fully excluded from non-restrictive analysis: "
          f"0/{n_target_leaked + n_items_checked} target rows present, "
          f"exactly 3 objects in all {n_items_checked} non-restrictive items")

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
        records.append({"item": item, "condition": cond, "js_divergence_bits": js_div,
                        "spearman_rho_within_item": rho})
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

def plot_scatter(analysis_df, overall_rho, out_path):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for cond in CONDITIONS:
        g = analysis_df[analysis_df["condition"] == cond].dropna(
            subset=["human_prob_analysis", "llm_prob_analysis"])
        rho, _, n = overall_rho.get(cond, (np.nan, np.nan, 0))
        label = f"{cond} (\u03c1={rho:.2f})" if np.isfinite(rho) else cond
        ax.scatter(g["human_prob_analysis"], g["llm_prob_analysis"], s=35, alpha=0.6,
                   color=COND_COLOR[cond], label=label)
    ax.plot([0, 1], [0, 1], linestyle=":", color="#888888", linewidth=1)
    ax.set_xlabel("Human probability  (chose object X / total participants)")
    ax.set_ylabel("LLM probability (P_norm)")
    ax.set_title("Human vs. LLM Object-Prediction Probability")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
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


def plot_js_divergence(item_metrics, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _box_with_dots(ax, item_metrics, "js_divergence_bits", "JS divergence (bits)",
                   "Jensen-Shannon Divergence per Item\n(each dot = one item; lower = more similar)")
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

    print("Computing human_prob = count(chose X) / total participants...")
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

    print("\nGenerating plots (3 total)...")
    plot_scatter(analysis_df, ov_rho, PLOTS_DIR / "scatter_human_vs_llm.png")
    plot_js_divergence(item_metrics, PLOTS_DIR / "js_divergence.png")
    plot_spearman(item_metrics, ov_rho, PLOTS_DIR / "spearman.png")

    print("\nDone.")
    return analysis_df, item_metrics


if __name__ == "__main__":
    main()