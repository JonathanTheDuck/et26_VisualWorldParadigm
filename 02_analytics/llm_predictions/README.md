# LLM surprisal predictions (50-item stimulus set)

Scene-based GPT-2 surprisal for all 50 experiment items, computed the same
way as Titus von der Malsburg's original 16-item Altmann & Kamide (1999)
replication table — for each item x condition (restrictive/non-restrictive
verb) x candidate object (target + 2-4 distractors), surprisal of the
object given `"<subject> will <verb> the"`, normalized within each 4-5
object scene into a P_norm, plus ΔS (bits) = surprisal(non-restrictive) -
surprisal(restrictive) per object.

## Pipeline

1. **`prep_vwp50_scene.py`** — parses
   `01_experiment/stimuli/creatingDataStructure/sentences.csv`, builds one
   sentence per (item, condition, candidate object), and records the GPT-2
   prefix length for each so only the object's own tokens get summed later.
   Outputs `input_vwp50_scene.csv` (batch input) + `meta_vwp50_scene.json`.

   Two non-restrictive verbs in `sentences.csv` used to carry the same kind
   of copy-paste corruption that was fixed for item 49 ("inlatecarry" ->
   "carry", commit 7856f09): item 26 "blow grab" -> "grab", item 29
   "eadturn" -> "turn". Confirmed against `annotation_audiov2.csv` (which
   already had the fixed text) and corrected directly in `sentences.csv`.

2. **`llm_generate.py -i input_vwp50_scene.csv -o output_vwp50_scene.csv`**
   — external tool, not part of this repo:
   <https://github.com/tmalsburg/llm_surprisal>. Computes per-token GPT-2
   surprisal (bits) for every sentence. Clone that repo and run this step
   from inside it (or point `-i`/`-o` at the paths in this folder).

3. **`vwp50_table.py`** — sums each object's own token surprisal, computes
   P_norm (softmax within each scene) and Rank, computes ΔS per object, and
   writes the final **`result_vwp50_scene_table.csv`** (Item, Condition, Verb,
   Object, Surprisal (bits), P_norm, Rank, ΔS (bits)).

Re-running steps 1 and 3 only needs `transformers`/`pandas` (no GPU
needed — just tokenization + arithmetic on top of GPT-2's own surprisal
numbers). Step 2 needs `torch` + `transformers` and the external tool.

## Files

- `input_vwp50_scene.csv`, `meta_vwp50_scene.json` — intermediate (step 1 output / step 3 input)
- `output_vwp50_scene.csv` — raw per-token GPT-2 surprisal (step 2 output)
- `result_vwp50_scene_table.csv` — **final result**, 468 rows (50 items x 2 conditions x 4-5 objects)
