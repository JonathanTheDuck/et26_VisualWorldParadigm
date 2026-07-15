# LLM surprisal predictions (50-item stimulus set)

Scene-based GPT-2 surprisal for all 50 experiment items, computed the same
way as Titus von der Malsburg's original 16-item Altmann & Kamide (1999)
replication table — for each item x condition (restrictive/non-restrictive
verb) x candidate object (target + distractors), surprisal of the
object given `"<subject> will <verb> the"`, normalized within each scene
into a P_norm, plus ΔS (bits) = surprisal(non-restrictive) -
surprisal(restrictive) per object.

## Pipeline

1. **`prep_vwp50_scene.py`** — parses
   `01_experiment/stimuli/creatingDataStructure/sentences_fromGdoc.csv` (the
   corrected/current stimulus list — supersedes the older `sentences.csv`,
   which still had 4-5 objects per scene for ~35 items; `sentences_fromGdoc.csv`
   trims nearly every item down to exactly 4 objects: target + 3 distractors),
   builds one sentence per (item, condition, candidate object), and records
   the GPT-2 prefix length for each so only the object's own tokens get
   summed later. Outputs `input_vwp50_scene.csv` (batch input) +
   `meta_vwp50_scene.json`.

   Fact-checked the resulting object lists against the filenames in
   `01_experiment/stimuli/img_discrete/` (every item has exactly 4 object
   images there). One item didn't match: item 36 (id, 0-indexed; "the kid
   will build/stack the lego") listed 4 distractors in `sentences_fromGdoc.csv`
   ("papers, clips, cards, boxes") but only 3 have images ("cards" has none)
   — "cards" removed directly from `sentences_fromGdoc.csv`. Everything else
   matched (modulo hyphen/spacing formatting on "rocking-horse"/"shuttle-cock",
   and two apparent typos in the image filenames themselves — item 41
   "toothbrish", item 46 "laddle" — neither of which is this script's
   problem to fix).

2. **`llm_generate.py -i input_vwp50_scene.csv -o output_vwp50_scene.csv`**
   — external tool, not part of this repo:
   <https://github.com/tmalsburg/llm_surprisal>. Computes per-token GPT-2
   surprisal (bits) for every sentence. Clone that repo and run this step
   from inside it (or point `-i`/`-o` at the paths in this folder).

3. **`vwp50_table.py`** — sums each object's own token surprisal, computes
   P_norm (softmax within each scene) and Rank, computes ΔS per object, and
   writes the final **`result_vwp50_scene_table.csv`** (Item, Condition, Verb,
   Object, Is_Target, Surprisal (bits), P_norm, Rank, ΔS (bits)).

Re-running steps 1 and 3 only needs `transformers`/`pandas` (no GPU
needed — just tokenization + arithmetic on top of GPT-2's own surprisal
numbers). Step 2 needs `torch` + `transformers` and the external tool.

## Files

- `input_vwp50_scene.csv`, `meta_vwp50_scene.json` — intermediate (step 1 output / step 3 input)
- `output_vwp50_scene.csv` — raw per-token GPT-2 surprisal (step 2 output)
- `result_vwp50_scene_table.csv` — **final result**, 400 rows (50 items x 2 conditions x 4 objects each)
