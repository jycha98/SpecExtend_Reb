import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from termcolor import colored
from accelerate import Accelerator
from eagle_3.model_eagle import EaModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─── CONFIGS ────────────────────────────────────────────────────────────────────

DATASETS = ['booksum', 'pg-19', 'govreport']

BASE_MODEL_MAP = {
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
}
DRAFT_MODEL_MAP = {
    "llama3.1": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
}

# ─── ARGS ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, choices=DATASETS)
parser.add_argument("--model_name",   type=str, default="llama3.1")
parser.add_argument("--output_folder", type=str, default="default")
parser.add_argument("--use_specextend", action="store_true")
args = parser.parse_args()

# ─── LOAD DATA ──────────────────────────────────────────────────────────────────

data_folder = os.path.join("data", args.dataset_name)
file_to_data = {}
n = 20
for fname in os.listdir(data_folder):
    full_path = os.path.join(data_folder, fname)
    records = []
    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {fname} line {i}: {e}")
                continue
            records.append(rec)
    if records:
        length = records[-1]["trunc_length"]
        file_to_data[length] = records

# ─── MODEL SETUP ────────────────────────────────────────────────────────────────

base_path  = BASE_MODEL_MAP[args.model_name]
draft_path = DRAFT_MODEL_MAP[args.model_name]

model = EaModel.from_pretrained(
    base_model_path   = base_path,
    ea_model_path     = draft_path,
    torch_dtype       = "auto",
    low_cpu_mem_usage = True,
    device_map        = "auto",
).eval()

tokenizer = model.get_tokenizer()

accelerator = Accelerator()
model, tokenizer = accelerator.prepare(model, tokenizer)

# ─── EXPERIMENT PARAMETERS ─────────────────────────────────────────────────────

temperature            = 0.0
max_new_tokens         = 256
use_specextend         = args.use_specextend
retrieval_chunk_size   = 32
retrieve_top_k         = 64
retrieve_every_n_steps = 4

cache_parameters = None
if use_specextend:
    cache_parameters = {
        "retrieval_chunk_size": retrieval_chunk_size,
        "retrieve_top_k":       retrieve_top_k,
        "retrieve_every_n_steps": retrieve_every_n_steps
    }


# ─── BUILD EXPERIMENT METADATA ─────────────────────────────────────────────────

def after_first_slash(x: str) -> str:
    return x.split("/", 1)[-1]

dataset      = after_first_slash(args.dataset_name)
target_model = after_first_slash(base_path)
draft_model  = after_first_slash(draft_path)

exp_name_base = f"{dataset}_{target_model}_{draft_model}_SpecExtend_{use_specextend}"
exp_config = {
    "exp_name":            exp_name_base,
    "dataset":             dataset,
    "target_model":        target_model,
    "draft_model":         draft_model,
    "use_specextend":      use_specextend,
    "cache_parameters":    cache_parameters
}

# ensure unique filename
i = 2
exp_name = exp_name_base
out_dir  = f"results/EAGLE3"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/{exp_name}.json"
while os.path.exists(out_path):
    exp_name = f"{exp_name_base}_{i}"
    out_path  = f"{out_dir}/{exp_name}.json"
    i += 1

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

summary = {}
sorted_lengths = sorted(file_to_data.keys())
for length in sorted_lengths:
    # collect over all runs for this length
    acc_accept_length = []
    acc_total_generated = 0
    acc_total_time      = 0.0

    # skip front few GovReport samples if too long
    if dataset == "GovReport" and length > 10000:
        records = file_to_data[length][4:]
    else:
        records = file_to_data[length]

    # Warmup
    print(colored("Warming up GPUs...", "yellow"))
    for rec in records[:1]:
        input_ids = tokenizer.encode(rec["text"], return_tensors="pt", add_special_tokens=True)
        input_ids = input_ids.to(accelerator.device)
        for _ in range(3):
            _ = model.eagenerate(
                input_ids,
                temperature            = 0.0,
                max_new_tokens         = 5,
                output_result_line     = False,
                verbose                = False,
                use_specextend         = use_specextend,
                retrieval_chunk_size   = retrieval_chunk_size,
                retrieve_top_k         = retrieve_top_k,
                retrieve_every_n_steps = retrieve_every_n_steps,
                retrieval_verbose      = False,
            )
    print(colored("Warmup complete!", "yellow"))

    for rec in tqdm(records, desc=f"len={length}"):
        for _ in range(2):
            input_ids = tokenizer.encode(rec["text"], return_tensors="pt", add_special_tokens=True)
            input_ids = input_ids.to(accelerator.device)

            results = model.eagenerate(
                input_ids,
                temperature            = temperature,
                max_new_tokens         = max_new_tokens,
                output_result_line     = True,
                verbose                = False,
                use_specextend         = use_specextend,
                retrieval_chunk_size   = retrieval_chunk_size,
                retrieve_top_k         = retrieve_top_k,
                retrieve_every_n_steps = retrieve_every_n_steps,
                retrieval_verbose      = False,
            )

            acc_accept_length.append(results["avg_accept_length"])
            acc_total_generated += results["total_generated"]
            acc_total_time      += results["inference_time"]

    num_runs = len(acc_accept_length)
    if num_runs == 0:
        continue

    avg_accept = round(float(np.mean(acc_accept_length)), 3)
    tokens_per_sec = round(acc_total_generated / acc_total_time, 3) if acc_total_time > 1e-9 else 0.0

    summary[length] = {
        "avg_accept_length":    avg_accept,
        "tokens_per_sec":       tokens_per_sec,
        "total_generated":      acc_total_generated,
        "total_inference_time": round(acc_total_time, 3),
        "num_runs":             num_runs,
    }

    final_data = {
        "exp_config": exp_config,
        "results":    summary,
    }

    with open(out_path, "w") as f:
        json.dump(final_data, f, indent=2)

print(colored(f"Experiment complete!", "green"))
