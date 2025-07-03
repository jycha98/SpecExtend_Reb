#!/usr/bin/env bash
set -euo pipefail

### Llama3 + EAGLE-1 experiments
python exp_llama3_eagle1_final.py --dataset_name govreport --use_specextend --start_length_id 1
python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend
python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend

### Llama 3.1 + EAGLE-3
python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend
python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend
python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend

### Llama 3.1 + EAGLE-3, 128K
python exp_llama3_eagle3_128k_final.py --dataset_name pg-19