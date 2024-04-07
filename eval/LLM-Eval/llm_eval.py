from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, PeftModel
import argparse
import json
import torch
import csv
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_predictions(pred_filepath):
    prediction_pairs = []
    with open(pred_filepath, "r") as pred_file:
        data = pred_file.read()
        rows = data.split("\n")
        for row in rows:
            pair = row.split("\t")
            sentence = pair[0]
            translation = pair[1]
            prediction_pairs.append((sentence, translation))
    return prediction_pairs

def initialize_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def create_instruction_candidates(prompt_template, instruction_template, pred_pair, labels):
    sentence = pred_pair[0]
    translation = pred_pair[1]

    instruction = instruction_template.replace("{{Translation}}", translation).replace("{{Sentence}}", sentence)
    instruction_candidates = []
    for label in labels:
        current_instruction = instruction.replace("{{Score}}", str(label))
        current_instruction = prompt_template.format(instruction=current_instruction)
        instruction_candidates.append(current_instruction)
    return instruction_candidates

def predict_score(input_prompts, model, tokenizer):
    model.eval()
    min_pp = float('inf')
    res = -1
    for idx, input_prompt in tqdm(enumerate(input_prompts)):
        with torch.inference_mode():
            encodings = tokenizer(input_prompt, padding='longest', truncation=True, max_length=2048, return_tensors="pt")
            encodings = encodings.to(device)
            input_ids = encodings["input_ids"]
            loss = model(input_ids, labels=input_ids).loss
        pp = torch.exp(loss).detach().cpu().item()
        print(pp)
        if pp < min_pp:
            min_pp = pp
            res = idx + 1
    return res

def evaluate_llm(args):
    labels = [i for i in range(1,6)]
    
    # load the prompt template
    with open('prompt_template.json', "r") as prompt_file:
        prompt_dict = json.load(prompt_file)
        prompt_template = prompt_dict[args.prompt_template]
    
    # load the prediction file
    assert args.prediction_file.endswith(".tsv")
    assert args.output_file.endswith(".tsv")

    pred_pairs = load_predictions(args.prediction_file)
    llm_scores = []

    # load the instruction file
    instruction_template = open("eval_prompt.txt").read()

    # setup the model
    model, tokenizer = initialize_model_and_tokenizer(args.hf_model)


    for pred_pair in tqdm(pred_pairs):
        instruction_candidates = create_instruction_candidates(prompt_template, instruction_template, pred_pair, labels)
        score = predict_score(instruction_candidates, model, tokenizer)
        llm_scores.append(score)
    
    output_columns = ['sentence', 'translation', 'score']

    with open(args.output_file, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(output_columns)
        for score, pred_tuple in zip(llm_scores, pred_pairs):
            sentence, translation = pred_tuple[0], pred_tuple[1]
            tsv_writer.writerow([sentence, translation, score])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str)
    parser.add_argument('--prompt_template', type=str, default='generic')
    parser.add_argument('--hf_model', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    evaluate_llm(args)