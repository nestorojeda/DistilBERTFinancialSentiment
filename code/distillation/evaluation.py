import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(model, dataloader, device):
    """Evaluates a model on a given dataloader.

    Args:
        model: The model to evaluate.
        dataloader: The DataLoader containing the evaluation data.
        device: The device to run evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary containing accuracy, precision, recall, and f1 score.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            # Ensure all necessary keys are present and move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels') # Use .get for safety
            if labels is None:
                 print("Warning: 'labels' key not found in batch. Skipping batch for evaluation.")
                 continue
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels: # Check if any labels were processed
        print("Warning: No labels processed during evaluation. Returning empty metrics.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 