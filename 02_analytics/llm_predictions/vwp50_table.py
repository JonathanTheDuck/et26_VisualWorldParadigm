"""
Build the scene surprisal table for the 50-item VWP stimulus set, using the
same logic that produced ak99_scene_table_final.csv for the original 16-item
set:

  - Surprisal (bits) of an object = sum of GPT-2 per-token surprisal for the
    tokens making up that object, i.e. everything after the
    "<subject> will <verb> the" prefix.
  - P_norm = softmax of 2**(-surprisal) across the objects in the same scene
    (same item + condition), so the 4-5 candidate objects sum to 1.
  - Rank = rank of the object within its scene by P_norm (1 = most probable).
  - ΔS (bits) = Surprisal(object | non-restrictive verb) - Surprisal(object |
    restrictive verb). This is a property of the object (not of the row), so
    it is identical on both the restrictive and non-restrictive rows for that
    object.
  - The target object of each item is flagged in the "Is_Target" column.
"""
import json, os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

output = pd.read_csv(os.path.join(SCRIPT_DIR, "output_vwp50_scene.csv"))
meta = {int(k): v for k, v in json.load(open(os.path.join(SCRIPT_DIR, "meta_vwp50_scene.json"))).items()}

# ── 1. Sum surprisal over the object's own tokens for every row ───────────
rows = []
for rid, m in meta.items():
    surp = output[(output["item"] == rid) & (output["idx"] > m["prefix_len"])]["surprisal"].sum()
    rows.append({**m, "surprisal": surp})

df = pd.DataFrame(rows)

# ── 2. P_norm + Rank within each (item_num, condition) scene ──────────────
def add_scene_stats(g):
    p = 2.0 ** (-g["surprisal"])
    g["P_norm"] = p / p.sum()
    g["rank"] = g["P_norm"].rank(ascending=False, method="min").astype(int)
    return g

df = df.groupby(["item_num", "condition"], group_keys=False).apply(add_scene_stats)

# ── 3. ΔS per object: non-restrictive surprisal - restrictive surprisal ───
wide = df.pivot_table(index=["item_num", "object"], columns="condition", values="surprisal")
wide["delta"] = wide["non-restrictive"] - wide["restrictive"]
df = df.merge(wide["delta"].reset_index(), on=["item_num", "object"], how="left")

# ── 4. Format for output ───────────────────────────────────────────────────
condition_label = {"restrictive": "Restrictive", "non-restrictive": "Non-restr."}

out = pd.DataFrame({
    "Item": df["item_num"],
    "Condition": df["condition"].map(condition_label),
    "Verb": df["verb"],
    "Object": df["object"],
    "Is_Target": df["is_target"],
    "Surprisal (bits)": df["surprisal"].round(3),
    "P_norm": df["P_norm"].round(4),
    "Rank": df["rank"],
    "ΔS (bits)": df["delta"].round(3),
})
out = out.sort_values(["Item", "Condition", "Rank"], ascending=[True, False, True])

out.to_csv(os.path.join(SCRIPT_DIR, "result_vwp50_scene_table.csv"), index=False)
print(f"Wrote result_vwp50_scene_table.csv with {len(out)} rows ({out['Item'].nunique()} items)")
print(out.head(10).to_string(index=False))
