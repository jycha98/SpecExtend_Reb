#!/usr/bin/env python
import os
import json
import argparse
import time
import numpy as np
from tqdm.auto import tqdm
from termcolor import colored
from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── ARGS ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name",   type=str, required=True, choices=["booksum","pg-19","govreport"])
args = parser.parse_args()

# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
data_folder = os.path.join("data", args.dataset_name)
file_to_data = {}
N_PER_FILE = 20
for fname in os.listdir(data_folder):
    path = os.path.join(data_folder, fname)
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i>=N_PER_FILE: break
            line = line.strip().lstrip("\ufeff")
            if not line: continue
            try:
                rec = json.loads(line)
            except:
                continue
            recs.append(rec)
    if recs:
        length = recs[-1]["trunc_length"]
        file_to_data[length] = recs

# ─── MODEL SETUP ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

accelerator = Accelerator()
model, tokenizer = accelerator.prepare(model, tokenizer)

eos_id = tokenizer.eos_token_id
max_new_tokens = 256

# ─── EXPERIMENT METADATA ────────────────────────────────────────────────────────
exp_base = f"{args.dataset_name}_llama3.1_ar"
out_dir  = os.path.join("results", args.dataset_name)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{exp_base}.json")

summary = {}
lengths = sorted(file_to_data.keys())

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
for length in lengths:
    recs = file_to_data[length]
    print(colored(f"→ length={length}, #samples={len(recs)}", "magenta"))

    acc_tokens_per_sec = []
    acc_times = []

    # for each sample run 2 replicates
    for rec in tqdm(recs[:3], desc=f"Len={length}"):
        prompt_ids = tokenizer(rec["text"], return_tensors="pt").input_ids.to(accelerator.device)

        for run in range(2):
            # manual autoregressive decode
            generated = prompt_ids.clone()
            past_kv   = None
            start = time.time()
            with torch.no_grad():
                for step in range(max_new_tokens):
                    if step == 0:
                        out = model(
                            input_ids=generated,
                            use_cache=True,
                        )
                    else:
                        out = model(
                            input_ids=next_token,
                            past_key_values=past_kv,
                            use_cache=True,
                        )
                    logits = out.logits
                    past_kv = out.past_key_values

                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=-1)

                    if next_token.item() == eos_id:
                        break

            elapsed = time.time() - start
            gen_tokens = generated.size(1) - prompt_ids.size(1)
            tps = gen_tokens / elapsed

            acc_times.append(elapsed)
            acc_tokens_per_sec.append(tps)

            # --- new: decode and print per‐run results ---
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(colored(f"Run {run+1} tokens/sec: {tps:.2f}\n", "yellow"))
            # --------------------------------------------

    # aggregate
    summary[length] = {
        "mean_tokens_per_sec": float(np.mean(acc_tokens_per_sec)),
        "mean_time_per_run":   float(np.mean(acc_times)),
        "num_runs":            len(acc_times),
    }

    # write intermediate
    with open(out_path, "w") as f:
        json.dump({"exp_config": vars(args), "results": summary}, f, indent=2)
print(colored("✅ Experiment complete!", "green"))
