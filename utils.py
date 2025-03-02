import numpy as np
import evaluate

def transform_labels(example):
    # Create a mapping dictionary for sentiment labels
    sentiment_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    # Get the sentiment value using the map with a default
    # This handles both cases in one step and makes the code more maintainable
    return {'labels': sentiment_map.get(example['Sentiment'].lower(), 0)}

def compute_metrics(eval_pred):
    # load the metrics to use
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculate the mertic using the predicted and true value 
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy, "f1score": f1}