from getLanguage import langIdentify
import csv
import argparse
def compute_cmi(sentence):
    lang_labels = langIdentify(sentence, "classifiers/HiEn.classifier")[0]
    label_freq = {
        "HI":0,
        "EN":0,
        "OTHER":0
    }
    hi_cnt = 0
    en_cnt = 0
    other_cnt = 0
    n = len(lang_labels)
    for lang_label in lang_labels:
        label_freq[lang_label[-1]] += 1
    wi = max(label_freq["HI"], label_freq["EN"])
    u = label_freq["OTHER"]
    if n == u:
        return 0.0
    return 100*(1-(wi/(n-u)))

def evaluate_cmi(args):
    predictions = []
    with open(args.prediction_file, "r") as f:
        data = f.read()
        rows = data.split("\n")
        for idx, row in enumerate(rows):
            if idx == 0:
                continue
            instance = row.split("\t")
            predictions.append(instance[1]) # column format: sentence/source, translation, reference
    cmi = []
    for pred in predictions:
        cmi.append(compute_cmi(pred))
    print(f"Average CMI: {sum(cmi)/len(cmi)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str)
    args = parser.parse_args()
    evaluate_cmi(args)