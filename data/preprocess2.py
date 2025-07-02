import os
import json
import random
import glob

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# 1) load train split of pg-19
ds = load_dataset("emozilla/pg19", split="train")

# 2) tokenizer (same as before)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", use_fast=False
)

# 3) define our buckets & constants
BUCKETS = {
    "32k":  (32_768,   65_536),
    "64k":  (65_536,  131_072),
    "128k": (131_072, float("inf")),
}
MAX_PER_BUCKET = 20
random.seed(42)

# collector
selected = {name: [] for name in BUCKETS}

# ─── STEP A: scan & collect up to 20 examples per bucket ─────────────────

for ex in tqdm(ds, desc="Scanning pg-19 train"):
    text = ex["text"]
    toks = tokenizer(text, add_special_tokens=False).input_ids
    L = len(toks)
    for name, (low, high) in BUCKETS.items():
        if low <= L < high and len(selected[name]) < MAX_PER_BUCKET:
            selected[name].append((text, toks))
    # stop early if all buckets full
    if all(len(v) >= MAX_PER_BUCKET for v in selected.values()):
        break

# sanity check
for name, items in selected.items():
    if len(items) < MAX_PER_BUCKET:
        raise RuntimeError(f"only found {len(items)} examples in bucket {name}")

# ─── STEP B: truncate & save raw jsonl per bucket ────────────────────────

for name, (low, _) in BUCKETS.items():
    out_path = f"pg19_{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for text, toks in tqdm(selected[name], desc=f"Writing pg19_{name}"):
            truncated = tokenizer.decode(toks[:low], skip_special_tokens=True)
            rec = {
                "truncated_length": low,
                "text": truncated
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected[name])} → {out_path}")

# ─── STEP C: read each raw file & write prompt version ───────────────────

for name in BUCKETS:
    infile  = f"pg19_{name}.jsonl"
    outfile = f"pg19_{name}_prompt.jsonl"
    with open(infile, "r", encoding="utf-8") as fin, \
         open(outfile, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Formatting pg19_{name}_prompt"):
            rec = json.loads(line)
            report = rec["text"]
            prompt = (
                "\n\nSummarize the following text into a summary of less than 800 words.\n"
                "### Text: \n\n"
                f"{report}\n\n"
                "### Summary:\n\n"
            )
            new_rec = {
                "truncated_length": rec["truncated_length"],
                "text": prompt
            }
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
    print(f"Wrote prompts → {outfile}")




######

import os
import glob
import json
from tqdm.auto import tqdm

# match only the RAW truncated files, not any *_prompt.jsonl
INPUT_PATTERN = "pg19_[!]*.jsonl"

for infile in glob.glob("pg19_*.jsonl"):
    # skip already‐prompted files
    if infile.endswith("_prompt.jsonl"):
        continue

    base, _ = os.path.splitext(infile)
    outfile = f"{base}_prompt.jsonl"

    with open(infile, "r", encoding="utf-8") as fin, \
         open(outfile, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Formatting {infile}"):
            rec = json.loads(line)
            text = rec["text"]
            # remove the last 200 characters to leave room for the prompt
            text = text[:-200] if len(text) > 200 else ""

            prompt = (
                "\n\nSummarize the following text into a summary of less than 800 words.\n"
                "### Text: \n\n"
                f"{text}\n\n"
                "### Summary:\n\n"
            )

            new_rec = {
                "truncated_length": rec["truncated_length"],
                "text": prompt
            }
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    print(f"Wrote prompts → {outfile}")