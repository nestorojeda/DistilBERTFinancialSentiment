"""
Financial Sentiment Analysis Model Fine-tuning Script.

This script fine-tunes an XLM-RoBERTa model for financial sentiment analysis
using multilingual financial dataset.
"""

import sys
import os
import argparse
import torch
import wandb
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Add the project root directory to the path so Python can find the toolbox package
# Get the absolute path to the project root (two levels up from the script)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")

from toolbox.utils import get_output_dir
from toolbox.logger import Logger


class CustomTrainer(Trainer):
    """Custom trainer with weighted loss function for imbalanced datasets."""
    
    def __init__(self, weight_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the original weight tensor and create loss function with it on the right device
        self.original_weight_tensor = weight_tensor
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.model.device))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure the loss function's weight tensor is on the same device as the model
        if hasattr(self.loss_fct, 'weight') and self.loss_fct.weight is not None:
            device = next(model.parameters()).device  # Get current model device
            if self.loss_fct.weight.device != device:
                # Recreate loss function with weight on the correct device
                self.loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.original_weight_tensor.to(device)
                )
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Ensure both tensors are on the same device
        device = logits.device
        labels = labels.to(device)
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def transform_labels(examples):
    """Transform string sentiment labels to integer labels.
    
    Args:
        examples: Dictionary containing sentiment labels
        
    Returns:
        Dictionary with added 'labels' key containing integer labels
    """
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    if isinstance(examples['sentiment'], list):
        examples['labels'] = [label_map[s.lower()] for s in examples['sentiment']]
    else:
        examples['labels'] = label_map[examples['sentiment'].lower()]
    return examples


def tokenize_data(tokenizer, example):
    """Tokenize the input text data.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        example: Dictionary containing 'sentence' key
        
    Returns:
        Tokenized examples
    """
    return tokenizer(example['sentence'], padding='max_length', truncation=True)


def compute_metrics(eval_pred):
    """Compute evaluation metrics.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        Dictionary of metric scores
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    return metric.compute(predictions=predictions, references=labels)


def setup_model_and_tokenizer(model_name_or_path, num_labels=3):
    """Set up the model and tokenizer.
    
    Args:
        model_name_or_path: Name or path of the model
        num_labels: Number of classification labels
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    return model, tokenizer


def prepare_datasets(tokenizer):
    """Prepare and preprocess the datasets.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        Dictionary containing train and evaluation datasets
    """
    # Load and preprocess the main dataset
    ds = load_dataset("nojedag/financial_phrasebank_multilingual")
    dataset = ds.map(transform_labels, batched=True)
    
    # Tokenize the data
    tokenize_function = lambda example: tokenize_data(tokenizer, example)
    dataset = dataset.map(tokenize_function, batched=True)
    
    # Prepare train and evaluation datasets
    train_dataset = dataset['train'].shuffle(seed=10)
    eval_dataset = dataset['test'].shuffle(seed=10)
    
    # Load and preprocess the synthetic dataset
    synthetic_ds = load_dataset("nojedag/synthetic_financial_sentiment")
    synthetic_ds = synthetic_ds.map(transform_labels, batched=True)
    tokenized_synthetic = synthetic_ds.map(
        lambda ex: tokenize_data(tokenizer, ex), 
        batched=True, 
        remove_columns=['sentence', 'lang']
    )
    synthetic_train_dataset = tokenized_synthetic['train'].shuffle(seed=10)
    synthetic_eval_dataset = tokenized_synthetic['test'].shuffle(seed=10)
    
    return {
        'main': {
            'train': train_dataset,
            'eval': eval_dataset
        },
        'synthetic': {
            'train': synthetic_train_dataset,
            'eval': synthetic_eval_dataset
        }
    }


def setup_training_args(model_name, batch_size, num_epochs, datasets):
    """Set up training arguments.
    
    Args:
        model_name: Name of the model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        datasets: Dictionary containing datasets
        
    Returns:
        TrainingArguments instance
    """
    # Calculate steps for training
    logging_steps = len(datasets['main']['train']) // batch_size
    steps = (len(datasets['main']['train']) / batch_size) * num_epochs
    warmup_steps = int(0.1 * steps)
    
    return TrainingArguments(
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        eval_strategy='steps',
        save_strategy='steps',
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        save_steps=1000,
        eval_steps=500,
        output_dir=get_output_dir(model_name),
        run_name=model_name,
        report_to="wandb",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        fp16=True
    )


def train_and_evaluate(model, tokenizer, datasets, training_args, weight_tensor=None):
    """Train and evaluate the model.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        datasets: Dictionary containing datasets
        training_args: TrainingArguments instance
        weight_tensor: Optional tensor of class weights
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    # Set up data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize the trainer
    if weight_tensor is not None:
        # Ensure weight tensor is on the correct device
        device = next(model.parameters()).device
        weight_tensor_device = weight_tensor.to(device)
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets['main']['train'],
            eval_dataset=datasets['main']['eval'],
            weight_tensor=weight_tensor_device,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['main']['train'],
            eval_dataset=datasets['main']['eval'],
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Evaluate on main dataset
    print("\nEvaluating on main dataset...")
    main_eval_results = trainer.evaluate()
    
    # Evaluate on synthetic dataset
    print("\nEvaluating on synthetic dataset...")
    synthetic_trainer_eval = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['synthetic']['train'],
        eval_dataset=datasets['synthetic']['eval'],
        compute_metrics=compute_metrics
    )
    synthetic_eval_results = synthetic_trainer_eval.evaluate()
    
    return trainer, {
        'main': main_eval_results,
        'synthetic': synthetic_eval_results
    }


def save_and_push_model(model, trainer, model_name):
    """Save and push the trained model to the hub.
    
    Args:
        model: The trained model
        trainer: The trainer instance
        model_name: Name of the model
    """
    print(f"\nSaving model to {get_output_dir(model_name)}...")
    model.save_pretrained(get_output_dir(model_name))
    
    print(f"Pushing model to hub as nojedag/{model_name}...")
    model.push_to_hub(f'nojedag/{model_name}')
    
    print("Pushing trainer artifacts to hub...")
    trainer.push_to_hub()


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune a model for financial sentiment analysis")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="xlm-roberta-finetuned-financial-news-sentiment-analysis-european",
        help="Name for the fine-tuned model",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Base model to fine-tune",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs for training",
    )
    
    parser.add_argument(
        "--use_weights",
        action="store_true",
        help="Use class weights for training",
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default="[2.2643, 0.6222, 1.0515]",
        help="Comma-separated list of class weights",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the fine-tuning process."""
    args = parse_arguments()
    
    # Initialize logger
    logger = Logger(log_dir="../logs", filename=f"finetuning_{args.model_name}.txt")
    logger.log(f"Starting fine-tuning process for model: {args.model_name}", type="INFO")
    logger.log(f"Arguments: {args}", type="INFO")
    
    # Initialize wandb
    logger.log("Initializing Weights & Biases", type="INFO")
    wandb.login()
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}", type="INFO")
    
    # Load metric
    global metric
    metric = evaluate.load("accuracy")
    logger.log("Loaded accuracy metric", type="INFO")
    
    # Set up model and tokenizer
    logger.start_timer("model_setup")
    logger.log(f"Setting up model and tokenizer: {args.base_model}", type="INFO")
    model, tokenizer = setup_model_and_tokenizer(args.base_model)
    logger.end_timer("model_setup")
    
    # Prepare datasets
    logger.start_timer("data_preparation")
    logger.log("Preparing datasets", type="INFO")
    datasets = prepare_datasets(tokenizer)
    logger.log(f"Train dataset size: {len(datasets['main']['train'])}", type="INFO")
    logger.log(f"Evaluation dataset size: {len(datasets['main']['eval'])}", type="INFO")
    logger.end_timer("data_preparation")
    
    # Set up training arguments
    logger.log("Setting up training arguments", type="INFO")
    training_args = setup_training_args(
        args.model_name,
        args.batch_size,
        args.epochs,
        datasets
    )
    
    # Set up weight tensor if needed
    weight_tensor = None
    if args.use_weights:
        weights = [float(w) for w in args.weights.split(',')]
        weight_tensor = torch.tensor(weights)
        logger.log(f"Using class weights: {weights}", type="INFO")
    
    # Train and evaluate the model
    logger.start_timer("training")
    logger.log("Starting model training and evaluation", type="INFO")
    trainer, eval_results = train_and_evaluate(
        model,
        tokenizer,
        datasets,
        training_args,
        weight_tensor
    )
    logger.end_timer("training")
    
    # Print evaluation results
    logger.log("Evaluation Results:", type="RESULTS")
    logger.log(f"Main dataset: {eval_results['main']}", type="RESULTS")
    logger.log(f"Synthetic dataset: {eval_results['synthetic']}", type="RESULTS")
    
    # Save and push the model
    logger.start_timer("model_saving")
    logger.log("Saving and pushing model to hub", type="INFO")
    save_and_push_model(model, trainer, args.model_name)
    logger.end_timer("model_saving")
    
    # Finish wandb session
    logger.log("Finishing Weights & Biases session", type="INFO")
    wandb.finish()
    
    # Log total execution time and dump logs to file
    total_time = logger.get_execution_time()
    logger.log(f"Fine-tuning process completed successfully in {total_time:.2f} seconds!", type="INFO")
    log_file_path = logger.dump_to_file()
    print(f"Logs saved to: {log_file_path}")


if __name__ == "__main__":
    main()


