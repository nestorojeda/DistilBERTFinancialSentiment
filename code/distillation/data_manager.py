from transformers import AutoTokenizer
from datasets import DatasetDict
from toolbox.utils import tokenize_data

def get_tokenized_lang_dataset(tokenizer: AutoTokenizer, dataset: DatasetDict, lang: str) -> DatasetDict:
    """Filters a dataset by language and tokenizes it."""
    raw_dataset = dataset.filter(lambda example: example['lang'] == lang)
    return raw_dataset.map(lambda example: tokenize_data(tokenizer, example), batched=True)
