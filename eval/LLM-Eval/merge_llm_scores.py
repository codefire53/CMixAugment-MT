import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score
import numpy as np


def eval_llm_as_judge_score(args):
    file_paths = args.prediction_files
    dataframes = []

    all_labels = [i for i in range(1, 6)]
    annotators_scores = []

    for i, file_path in enumerate(file_paths, start=1):
        df = pd.read_csv(file_path, delimiter='\t')
        
        annotator_scores = df['score'].tolist()
        annotator_scores = [int(score) for score in annotator_scores]
        annotators_scores.append(annotator_scores)

        df.rename(columns={'score': f'score_{i}'}, inplace=True)
        
        dataframes.append(df)
    final_df = pd.concat(dataframes, axis=1)

    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # List of columns to calculate the mode on
    score_columns = [col for col in final_df.columns if col.startswith('score_')]

    # Calculate the mode across the specified columns for each row
    final_df['final_score'] = final_df[score_columns].mean(axis=1)

    print(f"Average GEval: {final_df['final_score'].mean()}")

    output_file_path = args.output_file
    final_df[['sentence', 'translation', 'final_score']].to_csv(output_file_path, sep='\t', index=False)


    print(f"cohen kappa score: {cohen_kappa_score(annotators_scores[0], annotators_scores[1], labels=[1,2,3,4,5])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_files', type=str, default='[]')
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    eval_llm_as_judge_score(args)