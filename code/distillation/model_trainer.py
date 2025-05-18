import os
import sys
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from datasets import DatasetDict
from data_manager import get_tokenized_lang_dataset # Import from the new module
from evaluation import compute_metrics # Import from the new module
from transformers import EarlyStoppingCallback
from collections import Counter

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")

from toolbox.utils import get_output_dir

class CustomTrainer(Trainer):
    """Custom Trainer that uses weighted CrossEntropyLoss."""
    def __init__(self, weight_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure weight_tensor is moved to the correct device during loss calculation
        self.weight_tensor = weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Move weight tensor to the same device as logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weight_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def get_hyperparameters(train_dataset_size: int, batch_size=16, number_of_epochs=8):
    """Calculates logging steps and warmup steps based on dataset size and batch size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_dataset_size <= 0:
        print("Warning: train_dataset_size is zero or negative. Steps calculation might be incorrect.")
        logging_steps = 1 # Avoid division by zero
        steps = 0
    else:
        steps_per_epoch = train_dataset_size // batch_size
        if steps_per_epoch == 0:
             print(f"Warning: train_dataset_size ({train_dataset_size}) is smaller than batch_size ({batch_size}). Adjusting steps_per_epoch to 1.")
             steps_per_epoch = 1 # Ensure at least one step per epoch if dataset is very small
        logging_steps = steps_per_epoch
        steps = steps_per_epoch * number_of_epochs

    warmup_steps = int(0.1 * steps)
    print(f"Train size: {train_dataset_size}")
    print(f"Number of training steps: {steps}")
    print(f"Number of warmup steps: {warmup_steps}")
    print(f"Logging steps (per epoch): {logging_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {number_of_epochs}")
    return batch_size, number_of_epochs, logging_steps, warmup_steps

def get_training_args(model_name: str, lang: str, batch_size: int, number_of_epochs: int, logging_steps: int, warmup_steps: int) -> TrainingArguments:
    """Creates TrainingArguments for the Hugging Face Trainer."""
    output_dir_name = f'{model_name.replace("FacebookAI/", "")}-{lang}' # Avoid slashes in dir names
    return TrainingArguments(
        num_train_epochs=number_of_epochs,
        load_best_model_at_end=True,
        eval_strategy='steps', # Changed from evaluation_strategy
        save_strategy='steps',
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        save_steps=1000,
        eval_steps=500,
        output_dir=get_output_dir(output_dir_name),
        report_to="wandb", # Consider making this optional or configurable
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        fp16=True,
        no_cuda=False,
        dataloader_pin_memory=True,
        seed=42,  # Ensures reproducibility
        greater_is_better=True,  # Set according to your metric
        logging_first_step=True,  # Log the first step for early diagnostics
        disable_tqdm=False,  # Show progress bar
        remove_unused_columns=True,
    )

def fine_tune_language(model_name: str, dataset: DatasetDict, lang: str, device: torch.device) -> AutoModelForSequenceClassification:
    """Fine-tunes a model for a specific language."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    lang_dataset = get_tokenized_lang_dataset(tokenizer, dataset, lang)

    # Check if train split exists and has data
    if 'train' not in lang_dataset or len(lang_dataset['train']) == 0:
        print(f"Warning: No training data found for language '{lang}'. Skipping fine-tuning.")
        return None # Or raise an error, depending on desired behavior

    # Ensure test split exists for evaluation, if not, maybe split train?
    if 'test' not in lang_dataset:
        print(f"Warning: No test data found for language '{lang}'. Using a split of the training data for evaluation.")
        # Example: Splitting train data if test doesn't exist
        split_dataset = lang_dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    else:
        train_dataset = lang_dataset['train']
        eval_dataset = lang_dataset['test']

    if len(train_dataset) == 0:
        print(f"Warning: Training dataset for '{lang}' is empty after potential split. Skipping fine-tuning.")
        return None
    if len(eval_dataset) == 0:
        print(f"Warning: Evaluation dataset for '{lang}' is empty. Evaluation might not be meaningful.")
        # Decide how to handle empty eval: skip eval, use train set for eval (not ideal), etc.


    hyperparams = get_hyperparameters(len(train_dataset))
    training_args = get_training_args(model_name, lang, *hyperparams)

    train_dataset = train_dataset.shuffle(seed=42) # Use a fixed seed
    eval_dataset = eval_dataset.shuffle(seed=42)

    labels = train_dataset['labels']
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.tensor([total/counts[i] for i in range(3)], dtype=torch.float)
    weights = weights / weights.sum()  # normalize
    weight_tensor = weights.to(device)

    # Define class weights (consider calculating these based on data distribution)
    # Example: Using predefined weights
    class_weights = torch.tensor([2.2643, 0.6222, 1.0515]) # Adjust weights as needed
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        weight_tensor=weight_tensor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"Starting training for {lang}...")
    trainer.train()
    print(f"Training finished for {lang}.")

    # Re-evaluate using a standard Trainer instance if CustomTrainer interferes
    print(f"Evaluating model for {lang}...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results for {lang}: {eval_results}")

    return model 