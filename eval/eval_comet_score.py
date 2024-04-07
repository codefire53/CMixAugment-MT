import evaluate
import pandas as pd
import argparse

def evaluate_comet(args):
    df = pd.read_csv(args.prediction_file, sep='\t')
    comet = evaluate.load('comet')
    ref = df['reference']
    translation = df['translation']
    source = df['sentence']
    results = comet.compute(predictions=translation, references=ref, sources=source)
    print(f"Comet score: {results['mean_score']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, required=True)
    args = parser.parse_args()
    evaluate_comet(args)