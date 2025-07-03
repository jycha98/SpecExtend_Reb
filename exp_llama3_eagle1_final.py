import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from termcolor import colored
from accelerate import Accelerator
from eagle.model_eagle import EaModel_L3 as EaModel
import torch

# ─── FORCE GPU 0 ───────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─── CONFIGS ────────────────────────────────────────────────────────────────────
DATASETS = ['booksum', 'pg-19', 'govreport']
BASE_MODEL_MAP = {
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
}
DRAFT_MODEL_MAP = {
    "llama3.1": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
}

# ─── ARGS ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, choices=DATASETS)
parser.add_argument("--model_name",   type=str, default="llama3.1")
parser.add_argument("--output_folder", type=str, default="default")
parser.add_argument("--use_specextend", action="store_true")
parser.add_argument("--start_length_id", type=int, default=0)
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

# ─── EXPERIMENT METADATA ─────────────────────────────────────────────────────────
def after_first_slash(x: str) -> str:
    return x.split("/", 1)[-1]

dataset      = after_first_slash(args.dataset_name)
target_model = after_first_slash(BASE_MODEL_MAP[args.model_name])
draft_model  = after_first_slash(DRAFT_MODEL_MAP[args.model_name])
exp_name_base = f"{dataset}_{target_model}_{draft_model}_SpecExtend_{args.use_specextend}"

out_dir = f"results/EAGLE1"
os.makedirs(out_dir, exist_ok=True)
i = 2
exp_name = exp_name_base
out_path  = f"{out_dir}/{exp_name}.json"
while os.path.exists(out_path):
    exp_name = f"{exp_name_base}_{i}"
    out_path  = f"{out_dir}/{exp_name}.json"
    i += 1

exp_config = {
    "exp_name":         exp_name_base,
    "dataset":          dataset,
    "target_model":     target_model,
    "draft_model":      draft_model,
    "use_specextend":   args.use_specextend,
    "cache_parameters": {
        "retrieval_chunk_size": 32,
        "retrieve_top_k":       64,
        "retrieve_every_n_steps": 4
    } if args.use_specextend else None
}

# ─── EXPERIMENT LOOP ─────────────────────────────────────────────────────────────
summary = {}

sorted_lengths = sorted(file_to_data.keys())
for length in sorted_lengths[args.start_length_id:]:
    records = file_to_data[length]
    if dataset == "govreport":
        # skip first 4 if too long
        if length > 10000:
            records = records[4:]
        if length == 2048:
            # special deletion
            del records[6]

    # ─── FRESH MODEL LOAD ─────────────────────────────────────────────────────────
    print(colored(f"Loading model for length={length}...", "yellow"))
    model = EaModel.from_pretrained(
        base_model_path   = BASE_MODEL_MAP[args.model_name],
        ea_model_path     = DRAFT_MODEL_MAP[args.model_name],
        torch_dtype       = "auto",
        low_cpu_mem_usage = True,
        device_map        = "auto",
    ).eval()
    tokenizer = model.get_tokenizer()
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # ─── WARMUP ───────────────────────────────────────────────────────────────────
    print(colored("Warming up GPUs...", "yellow"))
    sample = records[0]
    input_ids = tokenizer.encode(sample["text"], return_tensors="pt", add_special_tokens=True).to(accelerator.device)
    for _ in range(3):
        _ = model.eagenerate(
            input_ids,
            temperature            = 0.0,
            max_new_tokens         = 5,
            output_result_line     = False,
            verbose                = False,
            use_specextend         = args.use_specextend,
            retrieval_chunk_size   = 32,
            retrieve_top_k         = 64,
            retrieve_every_n_steps = 4,
            retrieval_verbose      = False,
        )
    print(colored("Warmup complete!", "yellow"))

    # ─── RUNS WITH MOVING AVERAGE & CHECKPOINT ────────────────────────────────────
    acc_accepts = []
    total_gen   = 0
    total_time  = 0.0

    for rec in tqdm(records, desc=f"len={length}"):
        for _ in range(2):
            input_ids = tokenizer.encode(rec["text"], return_tensors="pt", add_special_tokens=True).to(accelerator.device)
            results = model.eagenerate(
                input_ids,
                temperature            = 0.0,
                max_new_tokens         = 256,
                output_result_line     = True,
                verbose                = False,
                use_specextend         = args.use_specextend,
                retrieval_chunk_size   = 32,
                retrieve_top_k         = 64,
                retrieve_every_n_steps = 4,
                retrieval_verbose      = False,
            )

            # update accumulators
            acc_accepts.append(results["avg_accept_length"])
            total_gen  += results["total_generated"]
            total_time += results["inference_time"]

            # compute moving averages
            avg_accept = round(float(np.mean(acc_accepts)), 3)
            tokens_sec = round(total_gen / total_time, 3) if total_time > 1e-9 else 0.0

            # update summary and immediately checkpoint
            summary[length] = {
                "avg_accept_length":    avg_accept,
                "tokens_per_sec":       tokens_sec,
                "total_generated":      total_gen,
                "total_inference_time": round(total_time, 3),
                "num_runs":             len(acc_accepts),
            }
            with open(out_path, "w") as f:
                json.dump({"exp_config": exp_config, "results": summary}, f, indent=2)

    # ─── TEARDOWN MODEL & FREE GPU ───────────────────────────────────────────────
    print(colored(f"Done length={length}, unloading model.", "yellow"))
    del model, tokenizer
    torch.cuda.empty_cache()

print(colored("All lengths complete!", "green"))
