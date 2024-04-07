def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    labels = [[label] for label in labels]
    return preds, labels

