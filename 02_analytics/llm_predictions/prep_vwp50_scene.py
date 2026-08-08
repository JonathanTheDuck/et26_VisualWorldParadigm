"""
Prepare the 50-item VWP stimulus set (et26_VisualWorldParadigm) for
scene-based surprisal computation, the same way prep_ak99.py / ak99_table.py
did for the original 16-item Altmann & Kamide replication.

Source of truth: sentences_fromGdoc.csv from the experiment repo -- the
corrected/current stimulus list (supersedes the older sentences.csv, which
still had 4-5 objects per scene for ~35 items; sentences_fromGdoc.csv trims
nearly every item down to exactly 4 objects: target + 3 distractors). Each
item gives a restrictive-verb sentence, a non-restrictive-verb sentence, and
a parenthesized distractor list.

For every item x condition x candidate-object combination we build a sentence
"<subject> will <verb> the <object>" and record how many GPT-2 tokens belong
to the prefix "<subject> will <verb> the" (prefix_len), so that later we can
sum only the surprisal of the tokens belonging to the object itself.

Output:
  input_vwp50_scene.csv   -- item,text,n  (batch input for llm_generate.py)
  meta_vwp50_scene.json   -- per-row item_num/condition/verb/object/is_target/prefix_len
"""
import csv, json, os
from transformers import GPT2TokenizerFast

# Resolved relative to this script's location so it works straight after
# cloning the repo, regardless of cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SENTENCES_CSV = os.path.join(
    REPO_ROOT, "01_experiment", "stimuli", "creatingDataStructure", "sentences_fromGdoc.csv"
)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def parse_sentence(sentence):
    """'<Subject> will <verb (possibly multi-word)> the <object>' -> (subject, verb, object)."""
    sentence = sentence.strip()
    subject, rest = sentence.split(" will ", 1)
    verb, obj = rest.rsplit(" the ", 1)
    return subject.strip(), verb.strip(), obj.strip()


def load_items():
    items = []
    with open(SENTENCES_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            item_id = int(row[0])
            sent1, sent2 = row[1], row[2]
            # the "objects" field is an unquoted "(a, b, c)" list, so csv.reader
            # has already split it across row[3:] on the internal commas
            objects_field = ",".join(row[3:]).strip().strip("()")
            distractors = [d.strip() for d in objects_field.split(",") if d.strip()]

            subject, verb_r, target = parse_sentence(sent1)
            _, verb_n, target_n = parse_sentence(sent2)
            assert target.lower() == target_n.lower(), \
                f"item {item_id}: target mismatch {target!r} vs {target_n!r}"

            items.append({
                "item_num": item_id + 1,   # 1-indexed, matches ak99 convention
                "subject": subject,
                "restrictive_verb": verb_r,
                "nonrestrictive_verb": verb_n,
                "target": target,
                "distractors": distractors,
            })
    return items


def build_rows(items):
    rows, meta, rid = [], {}, 1
    for it in items:
        objects = [it["target"]] + it["distractors"]
        for condition, verb in [("restrictive", it["restrictive_verb"]),
                                 ("non-restrictive", it["nonrestrictive_verb"])]:
            prefix = f"{it['subject']} will {verb} the"
            prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
            for obj in objects:
                sentence = f"{prefix} {obj}"
                rows.append({"item": rid, "text": sentence, "n": 0})
                meta[str(rid)] = {
                    "item_num": it["item_num"],
                    "condition": condition,
                    "verb": verb,
                    "object": obj,
                    "is_target": obj == it["target"],
                    "prefix_len": prefix_len,
                }
                rid += 1
    return rows, meta


if __name__ == "__main__":
    items = load_items()
    print(f"Loaded {len(items)} items")
    rows, meta = build_rows(items)
    print(f"Built {len(rows)} (item x condition x object) sentences")

    with open(os.path.join(SCRIPT_DIR, "input_vwp50_scene.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "text", "n"])
        w.writeheader()
        w.writerows(rows)

    json.dump(meta, open(os.path.join(SCRIPT_DIR, "meta_vwp50_scene.json"), "w"), indent=2)
    print("Wrote input_vwp50_scene.csv and meta_vwp50_scene.json")
