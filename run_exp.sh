#!/usr/bin/env bash
set -uo pipefail

# ### Llama3 + EAGLE-1 experiments
# python exp_llama3_eagle1_final.py --dataset_name govreport --use_specextend --start_length_id 1
# python exp_llama3_eagle1_final.py --dataset_name govreport --use_specextend --start_length_id 2
# python exp_llama3_eagle1_final.py --dataset_name govreport --use_specextend --start_length_id 3
# python exp_llama3_eagle1_final.py --dataset_name govreport --use_specextend --start_length_id 4
# python exp_llama3_eagle1_final.py --dataset_name govreport --use_specextend --start_length_id 5
# python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend 
# python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend --start_length_id 1
# python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend --start_length_id 2
# python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend --start_length_id 3
# python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend --start_length_id 4
# python exp_llama3_eagle1_final.py --dataset_name pg-19 --use_specextend --start_length_id 5
# python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend
# python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend --start_length_id 1
# python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend --start_length_id 2
# python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend --start_length_id 3
# python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend --start_length_id 4
# python exp_llama3_eagle1_final.py --dataset_name booksum --use_specextend --start_length_id 5

# ### Llama 3.1 + EAGLE-3
# python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend
# python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend --start_length_id 1
# python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend --start_length_id 2
# python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend --start_length_id 3
# python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend --start_length_id 4
# python exp_llama3_eagle3_final.py --dataset_name govreport --use_specextend --start_length_id 5
# python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend
# python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend --start_length_id 1
# python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend --start_length_id 2
# python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend --start_length_id 3
# python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend --start_length_id 4
# python exp_llama3_eagle3_final --dataset_name pg-19 --use_specextend --start_length_id 5
# python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend
# python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend --start_length_id 1
# python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend --start_length_id 2
# python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend --start_length_id 3
# python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend --start_length_id 4
# python exp_llama3_eagle3_final.py --dataset_name booksum --use_specextend --start_length_id 5

# # ### Llama 3.1 + EAGLE-3, 128K
# python exp_llama3_eagle1_128k_final.py --dataset_name pg-19 --use_specextend
# python exp_llama3_eagle1_128k_final.py --dataset_name pg-19 --use_specextend --start_length_id 1
# python exp_llama3_eagle1_128k_final.py --dataset_name pg-19 --use_specextend --start_length_id 2

# AR Baseline
# python exp_llama3_ar_4_43.py --dataset_name govreport
# python exp_llama3_ar_4_43.py --dataset_name pg-19
# python exp_llama3_ar_4_43.py --dataset_name booksum
# pip install transformers -U
# pip install accelerate
# python exp_llama3_ar.py --dataset_name pg-19
# python exp_llama3_ar.py --dataset_name booksum
# python exp_llama3_eagle1_128k_final.py --dataset_name pg-19
# pip install accelerate -U
# pip install transformers - U
# python exp_llama3_ar2.py --dataset_name govreport
# python exp_llama3_ar.py --dataset_name govreport
# pip install accelerate==0.21.0
# pip install transformers==4.43.0
python exp_llama3_ar_4_43.py --dataset_name govreport
