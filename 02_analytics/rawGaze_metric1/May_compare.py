import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, mannwhitneyu

pd.set_option("display.max_rows", 30)
sns.set_theme(style="whitegrid", context="talk")

# ── load & clean ─────────────────────────────────────────────────────────
human_df = pd.read_csv("output_all_withPercentages.csv")
llm_df = pd.read_csv("../llm_predictions/result_vwp50_scene_table.csv")

llm_df = llm_df.rename(columns={"Item": "stimuliId", "Condition": "condition", "Object": "obj"})
llm_df["stimuliId"] = llm_df["stimuliId"].astype(int)
# fix different condition spellings between the human and llm pipelines
llm_df["condition"] = llm_df["condition"].replace({"Restrictive": "restrictive", "Non-restr.": "non-restrictive"})

human_df = human_df.rename(columns={"percentages": "human_percent"})
print(human_df)


#before excluding any trials we analyze the percentage of trials where participants predicted the target
def calculate_human_top_is_target(df):
    """
    For each trial (stimuliId + condition): 
    Is the object with highest human_percent also the target?
    Returns percentage and bar chart.
    """
    results = []
    
    for (stim_id, cond), group in df.groupby(["stimuliId", "condition"]):
        if group["human_percent"].isna().all():
            continue  # Skip trials with no predictions
        
        human_top_obj = group.loc[group["human_percent"].idxmax(), "obj"]
        target_objs = group.loc[group["is_target"] == 1, "obj"].values
        is_target = human_top_obj in target_objs
        
        results.append({
            "condition": cond,
            "human_top_is_target": is_target,
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate percentages
    overall = results_df["human_top_is_target"].mean()
    by_cond = results_df.groupby("condition")["human_top_is_target"].mean()
    
    print(f"Overall: {overall:.1%}")
    print(f"\nBy condition:")
    for cond, pct in by_cond.items():
        print(f"  {cond}: {pct:.1%}")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    by_cond.plot(kind="bar", ax=ax, color=["blue", "red"])
    ax.set_ylabel("% Top Human = Target")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for i, v in enumerate(by_cond):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    plt.tight_layout()
    plt.show()
    
    return overall, by_cond

# Run the analysis
top_is_target_df = calculate_human_top_is_target(human_df)
print(top_is_target_df)

def plot_human_percent_histograms(df):
    """
    Display histograms of human_percent distribution 
    for restrictive vs non-restrictive conditions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, cond in zip(axes, ["restrictive", "non-restrictive"]):
        data = df[df["condition"] == cond]["human_percent"].dropna()
        
        ax.hist(data, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Human Probability (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{cond.capitalize()}")
        ax.grid(axis="y", alpha=0.3)
        
        # Add stats
        stats_text = f"n = {len(data)}\nmean = {data.mean():.2f}\nmedian = {data.median():.2f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.show()
plot_human_percent_histograms(human_df)

# ── exclude trials with no usable human gaze data ───────────────────────
# trials (stimuliId + condition) where all 4 objects got zero gaze samples come out as
# NaN in human_percent (0/0) and carry no signal, so exclude them before comparing to the LLM
excluded_trials = human_df.loc[human_df["human_percent"].isna(), ["stimuliId", "condition"]].drop_duplicates()
print(f"excluding {len(excluded_trials)} stimuli/condition trials with no usable human gaze data")

valid_human_df = human_df.dropna(subset=["human_percent"]).copy()

# ── merge human and llm predictions ─────────────────────────────────────
# inner join + validate="one_to_one" catches silent duplicate-row bugs early
# (each stimuliId/condition/obj combination should appear exactly once on both sides)
shared_df = valid_human_df.merge(
    llm_df[["stimuliId", "condition", "obj", "P_norm"]],
    on=["stimuliId", "condition", "obj"],
    how="inner",
    validate="one_to_one",
)

counts_per_item = shared_df.groupby(["stimuliId", "condition"]).size()
#assert (counts_per_item == 4).all(), "some items don't have all 4 objects after the merge"

print(shared_df)

# ── top-choice agreement rate ────────────────────────────────────────────
# does the human's most-gazed object match the LLM's lowest-surprisal (highest P_norm) object?
agree_df = (
    shared_df.groupby(["stimuliId", "condition"])
    .apply(lambda g: pd.Series({
        "human_top": g.loc[g["human_percent"].idxmax(), "obj"],
        "llm_top": g.loc[g["P_norm"].idxmax(), "obj"],
    }))
    .reset_index()
)
agree_df["agree"] = agree_df["human_top"] == agree_df["llm_top"]

agreement_rate = agree_df.groupby("condition")["agree"].mean().reset_index(name="agreement_rate")

fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(data=agreement_rate, x="condition", y="agreement_rate", ax=ax)
for i, row in agreement_rate.iterrows():
    ax.text(i, row["agreement_rate"] + 0.02, f"{row['agreement_rate']:.0%}", ha="center")
ax.set_ylim(0, 1)
ax.set_ylabel("Human/LLM top-choice agreement rate")
ax.set_title("Do human gaze and LLM surprisal pick the same top object?")
plt.show()

# ── target-object confidence distribution ────────────────────────────────
target_df = shared_df[shared_df["is_target"] == 1][["stimuliId", "condition", "human_percent", "P_norm"]]
long_df = target_df.melt(
    id_vars=["stimuliId", "condition"], value_vars=["human_percent", "P_norm"],
    var_name="source", value_name="probability"
)
long_df["source"] = long_df["source"].replace({"human_percent": "Human", "P_norm": "LLM"})

fig, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(data=long_df, x="condition", y="probability", hue="source", split=True, ax=ax)
ax.set_ylabel("Probability assigned to the correct target object")
ax.set_title("Human vs. LLM confidence in the target object, by verb type")
plt.show()

# ── Jensen-Shannon divergence per item ────────────────────────────────────
def item_jsd(group):
    p = group.sort_values("obj")["human_percent"].to_numpy()
    q = group.sort_values("obj")["P_norm"].to_numpy()
    # base=2 bounds JSD to [0,1] bits; scipy renormalizes p/q internally so they need not sum to exactly 1
    return jensenshannon(p, q, base=2) ** 2

jsd_df = (
    shared_df.groupby(["stimuliId", "condition"])
    .apply(item_jsd)
    .reset_index(name="JSD")
)
print(jsd_df)

# ── Spearman correlation, pooled per condition, as a confirmatory check ──
for cond, sub in shared_df.groupby("condition"):
    r, p = spearmanr(sub["human_percent"], sub["P_norm"])
    print(f"{cond}: Spearman r = {r:.3f}, p = {p:.4g}, n = {len(sub)}")

# ── JSD comparison between conditions (the actual hypothesis test) ────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=jsd_df, x="condition", y="JSD", ax=ax)
sns.stripplot(data=jsd_df, x="condition", y="JSD", color="black", alpha=0.4, ax=ax)
ax.set_title("Human-LLM agreement (JSD) by verb type")
plt.show()

restr = jsd_df.loc[jsd_df["condition"] == "restrictive", "JSD"]
nonrestr = jsd_df.loc[jsd_df["condition"] == "non-restrictive", "JSD"]
u_stat, p_val = mannwhitneyu(restr, nonrestr, alternative="two-sided")
print(f"restrictive median JSD = {restr.median():.3f}, non-restrictive median JSD = {nonrestr.median():.3f}")
print(f"Mann-Whitney U = {u_stat:.1f}, p = {p_val:.4g}")
