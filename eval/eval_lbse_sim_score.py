from transformers import AutoTokenizer, AutoModelForSequenceClassification
import evaluate
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import argparse
from tqdm import tqdm

def compute_cosine_sim(embds1, embds2):
    sim_matrix = np.dot(embds1, embds2.T)
    norms1 = np.linalg.norm(embds1, axis=1)
    norms2 = np.linalg.norm(embds2, axis=1)
    score = sim_matrix / (np.outer(norms1, norms2) + 1e-9)
    return score

def eval_sim_score(all_sim_scores, sts_model, source_sentences, detox_sentences):
    source_len = len(source_sentences)
    detox_len = len(detox_sentences)
    all_texts = detox_sentences + source_sentences
    embds = sts_model.encode(all_texts)
    detox_embds = embds[:detox_len]
    source_embds = embds[detox_len:]
    scores = []
    sim_scores = compute_cosine_sim(detox_embds, source_embds)
    for i in range(detox_len):
        sim_score = max(sim_scores[i,i], 0)
        scores.append(sim_score)
    print(f"similarity: {scores}")
    all_sim_scores.extend(scores)

def eval_lbse_sim_score(args):
    instances = []
    with open(args.pred_file, 'r') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for idx, line in enumerate(tsv_file):
            if idx == 0:
                continue
            instances.append(line)
    
    tokenizer = AutoTokenizer.from_pretrained('textdetox/xlmr-large-toxicity-classifier')
    sts_model = SentenceTransformer('sentence-transformers/LaBSE')

    sim_scores = []

    og_sents = [row[0] for row in instances]
    pred_sents = [row[1] for row in instances]
    for i in tqdm(range(0, len(instances), args.batch_size)):
        eval_sim_score(sim_scores, sts_model, og_sents[i:i+args.batch_size], pred_sents[i:i+args.batch_size])
    
    overall_sim = sum(sim_scores)/len(sim_scores)

    print(f"similarity: {overall_sim}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    eval_lbse_sim_score(args)