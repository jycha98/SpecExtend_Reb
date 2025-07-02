from classic.model_classic import SPModel
from eagle.model_eagle import EaModel_L3

from run_classic import load_texts_from_jsonl
from accelerate import Accelerator
from termcolor import colored

# mappings from run_classic.py:
BASE_MODEL_MAP = {
    "llama3.1":   "meta-llama/Llama-3.1-8B-Instruct",
}
DRAFT_MODEL_MAP = {
    "llama3.1":   "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
}
# ────────────────────────────────────────────────────────────────────────────
# 1) ─── USER-CONFIGURE THESE ────────────────────────────────────────────────
INPUT_FILE        = "data/govreport/govreport_2K.jsonl"
MAX_SAMPLES       = 1
MODEL_NAME        = "llama3.1"  
# MODEL_NAME        = "vicuna_7b"   
MAX_GEN_LEN       = 256

# load texts
texts = load_texts_from_jsonl(INPUT_FILE, max_samples=MAX_SAMPLES)

# build & prepare model
base_path  = BASE_MODEL_MAP[MODEL_NAME]
draft_path = DRAFT_MODEL_MAP[MODEL_NAME]
model = EaModel_L3.from_pretrained(
        base_model_path = base_path,
        ea_model_path= draft_path,
        torch_dtype     = "auto",
        low_cpu_mem_usage=True,
        device_map      = "auto"
        ).eval()

tokenizer = model.tokenizer

accelerator = Accelerator()
model, tokenizer = accelerator.prepare(model, tokenizer)
# no warmup step

# inference loop
text = texts[0]
input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
input_ids = input_ids.to(accelerator.device)

USE_SPECEXTEND    = True
VERBOSE           = True
OUTPUT_RESULT_LINE= True

print('Warming up GPUs...')
for i in range(3):
    out = model.eagenerate(
        input_ids,
        temperature         = 0,
        max_new_tokens      = 3,
        output_result_line  = False,
        verbose             = False,
        use_specextend      = USE_SPECEXTEND,
        retrieval_chunk_size= 32,
        retrieve_top_k      = 64,
        retrieve_every_n_steps= 4,
        retrieval_verbose   = False
    )
print('Warmup complete!')

out = model.eagenerate(
    input_ids,
    temperature         = 0,
    max_new_tokens      = MAX_GEN_LEN,
    output_result_line  = OUTPUT_RESULT_LINE,
    verbose             = VERBOSE,
    use_specextend      = True,
    retrieval_chunk_size= 32,
    retrieve_top_k      = 64,
    retrieve_every_n_steps= 4,
    retrieval_verbose   = False
)