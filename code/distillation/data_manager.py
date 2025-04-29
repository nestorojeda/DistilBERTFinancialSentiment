from transformers import AutoTokenizer
from datasets import DatasetDict

def tokenize_data(example, tokenizer):
    """Tokenizes a single data example."""
    return tokenizer(example['sentence'], padding='max_length', truncation=True)

def get_tokenized_lang_dataset(tokenizer: AutoTokenizer, dataset: DatasetDict, lang: str) -> DatasetDict:
    """Filters a dataset by language and tokenizes it."""
    raw_dataset = dataset.filter(lambda example: example['lang'] == lang)
    return raw_dataset.map(lambda example: tokenize_data(example, tokenizer), batched=True)

def transform_labels(examples):
    """Transforms sentiment labels from strings to integers."""
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    # Handle both single examples and batches
    if isinstance(examples['sentiment'], list):
        examples['labels'] = [label_map[s.lower()] for s in examples['sentiment']]
    else:
        # Handle potential non-string or unexpected format gracefully
        sentiment_str = str(examples.get('sentiment', '')).lower()
        examples['labels'] = label_map.get(sentiment_str, -1) # Use -1 or some indicator for unknown labels
    return examples 