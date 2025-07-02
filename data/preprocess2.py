import os
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# 1) load
ds = load_dataset("ccdv/govreport-summarization", split="train")

# 2) tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", use_fast=False
)

# 3) define our buckets: (inclusive lower, exclusive upper)
BUCKETS = {
    "32k":  (32_768,   65_536),
    "64k":  (65_536,  131_072),
    "128k": (131_072, float("inf")),
}
MAX_PER_BUCKET = 20
random.seed(42)

# collector
selected = {name: [] for name in BUCKETS}

# scan until each bucket has 20
for ex in tqdm(ds, desc="Scanning train split"):
    report = ex.get("report", ex.get("text"))
    toks = tokenizer(report, add_special_tokens=False).input_ids
    L = len(toks)
    for name, (low, high) in BUCKETS.items():
        if low <= L < high and len(selected[name]) < MAX_PER_BUCKET:
            selected[name].append((report, toks))
    if all(len(sel) >= MAX_PER_BUCKET for sel in selected.values()):
        break

# sanity check
for name, items in selected.items():
    if len(items) < MAX_PER_BUCKET:
        raise RuntimeError(f"only found {len(items)} examples in bucket {name}")

# 4) truncate & save per bucket
for name, (low, _) in BUCKETS.items():
    out_path = f"govreport_{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for report, toks in tqdm(
            selected[name],
            desc=f"Writing {name} ({low} tokens) → {out_path}"
        ):
            truncated = tokenizer.decode(toks[:low], skip_special_tokens=True)
            record = {
                "truncated_length": low,
                "text": truncated
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected[name])} records to {out_path}")

#####

import os
import glob
import json
from tqdm.auto import tqdm

# pattern matching the 3 buckets you produced
INPUT_PATTERN = "govreport_*.jsonl"

for infile in glob.glob(INPUT_PATTERN):
    base, _ext = os.path.splitext(infile)
    outfile = f"{base}_prompt.jsonl"

    with open(infile, "r", encoding="utf-8") as fin, \
         open(outfile, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Formatting {infile}"):
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

    print(f"Wrote prompts to {outfile}")
