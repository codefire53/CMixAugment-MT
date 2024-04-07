#!/bin/bash

#SBATCH --job-name=llm_eval_nlp804 # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l2-001


echo "Evaluate results"
python llm_eval.py --prediction_file baseline_finetuned_output_test_m2m100_418M.tsv --prompt_template generic --hf_model bigscience/bloomz-7b1 --output_file bloomz_m2m100_finetuned.tsv
