# CMixAugment-MT: An Empirical Study of Leveraging Finetuning of Machine Translation Model To Produce Synthetic Code Mixing Data For Data Augmentation 

This is my NLP804 (Deep Learning For Natural Language Generation) project. The goal here is to evaluate different finetuning strategies on machine translation model for producing better code-mixed synthetic dataset.

## Setup
Setup conda environment
```
conda create -n cmixaugment-mt python=3.10
conda activate cmixaugment-mt
```
Install all dependencies
```
pip install -r requirements.txt
```

## How to run experiment
This project mainly uses `hydra` library to store the configuration. To run the experiment you can run this command
```
python main.py --config-name <adapter_config/baseline_config/sft_config>
```

## Evaluation
We use various evaluation metrics to assess the quality of the generated synthetic data. For downstream task evaluation, you can check the `eval/GLUECoS` on NLI and Sentiment Analysis (note that training file other than baseline, we add sample ratio information before the extension in the filename). While for the code-mixing metrics, you can check the `eval/LID-tool` and the main file is `eval/LID-tool/measure_cmi.py`. For evaluation using LLM-as-a-judge, you can check `eval/LLM-eval` and we use `llm_eval.py` to evaluate the text quality using specific LLM and then merge all the LLMs' judgment using `merge_llm_scores.py`. Lastly, for reference-free metrics you can check both `eval/eval_comet_score.py` and `eval.eval_lbse_sim_score.py`.

## Experiment Assets
You can check out this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/mahardika_ihsani_mbzuai_ac_ae/EsjquNQJfzVEqswxV1ou5awBpSa7eP8Uf4OgW7iiTV_2Aw?e=Fm9HaM)
