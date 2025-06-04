# %% Imports
import os
import sys
import torch
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import Adam
from torch.cuda.amp import GradScaler

# Local imports from the refactored modules
from toolbox.utils import transform_labels, tokenize_data
from evaluation import evaluate
from model_trainer import fine_tune_language
from distillation import DistillationTrainer

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")

from toolbox.utils import get_output_dir
from toolbox.logger import Logger 

# %% Parse Command Line Arguments
parser = argparse.ArgumentParser(description='Financial Sentiment Analysis Distillation Training')
parser.add_argument('--epochs', type=int, default=10, help='Number of distillation epochs (default: 10)')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training (default: 16)')
parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate (default: 5e-5)')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for distillation (default: 2.0)')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for distillation (default: 0.5)')
parser.add_argument('--langs', type=str, default='en,de,es,fr', 
                   help='Comma-separated list of language codes (default: en,de,es,fr)')

args = parser.parse_args()

# %% Configuration
BASE_MODEL_NAME = "FacebookAI/xlm-roberta-base"
PROJECT_NAME_PREFIX = "xlm-roberta-base-finetuned-financial-phrasebank"
STUDENT_MODEL_NAME = f'{PROJECT_NAME_PREFIX}-student'
STUDENT_MODEL_SAVE_PATH = f'models/{STUDENT_MODEL_NAME}'

# Set values from command line arguments or use defaults
DISTILLATION_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
TEMPERATURE = args.temperature
ALPHA = args.alpha
LANGS = args.langs.split(',')

LANG_2_ID = {
    "en": 0,
    "fr": 1,
    "de": 2,
    "es": 3,
}

ID_2_LANG = {v: k for k, v in LANG_2_ID.items()}

logger = Logger(log_dir="../logs")
logger.log("Script started", type="INFO")
logger.log("Configuration parameters", type="INFO", 
           distillation_epochs=DISTILLATION_EPOCHS, 
           batch_size=BATCH_SIZE, 
           learning_rate=LEARNING_RATE, 
           temperature=TEMPERATURE, 
           alpha=ALPHA, 
           languages=LANGS)

# %% Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log(f"Using device: {device}", type="INFO", device_type=str(device))
print(f"Using device: {device}")

# %% Load and Prepare Base Dataset
logger.start_timer("dataset_loading")
logger.log("Loading base dataset...", type="INFO")
print("Loading base dataset...")
complete_dataset = load_dataset("nojedag/financial_phrasebank_multilingual_augmented")
logger.log("Base dataset loaded and labels transformed.", type="INFO", 
           dataset_size={"train": len(complete_dataset['train']), "test": len(complete_dataset['test'])})
logger.end_timer("dataset_loading")
print("Base dataset loaded and labels transformed.")

# %% Fine-tune or Load Teacher Models
teacher_models = {}
langs_to_fine_tune = []

logger.log("Checking for existing teacher models...", type="INFO")
print("Checking for existing teacher models...")
for lang in LANGS:
    model_output_dir = get_output_dir(f'{PROJECT_NAME_PREFIX}-{lang}')
    if os.path.exists(os.path.join(model_output_dir, "model.safetensors")): # More specific check
        try:
            logger.start_timer(f"load_model_{lang}")
            logger.log(f"Loading existing teacher model for {lang} from {model_output_dir}", type="INFO")
            teacher_models[lang] = AutoModelForSequenceClassification.from_pretrained(model_output_dir)
            teacher_models[lang].to(device)
            logger.log(f"Teacher model for {lang} loaded successfully.", type="SUCCESS")
            logger.end_timer(f"load_model_{lang}")
        except Exception as e:
            logger.log(f"Error loading model for {lang} from {model_output_dir}: {e}. Will attempt fine-tuning.", 
                      type="ERROR", exception=str(e))
            langs_to_fine_tune.append(LANG_2_ID[lang])
            print(f"Error loading model for {lang} from {model_output_dir}: {e}. Will attempt fine-tuning.")
    else:
        logger.log(f"Model directory for {lang} ({model_output_dir}) does not exist or is incomplete. Scheduling for fine-tuning.",
                  type="INFO")
        langs_to_fine_tune.append(LANG_2_ID[lang])

# Fine-tune models for languages that were not found or failed to load
if langs_to_fine_tune:
    logger.log(f"Starting fine-tuning for languages: {langs_to_fine_tune}", type="INFO")
    for lang in langs_to_fine_tune:
        logger.start_timer(f"finetune_model_{ID_2_LANG[lang]}")
        logger.log(f"--- Training teacher model for {ID_2_LANG[lang]} ---", type="INFO")
        model = fine_tune_language(BASE_MODEL_NAME, complete_dataset, lang, device)
        if model:
            teacher_models[lang] = model.to(device) # Ensure model is on the correct device
            save_dir = get_output_dir(f'{PROJECT_NAME_PREFIX}-{ID_2_LANG[lang]}')
            logger.log(f"Saving fine-tuned model for {ID_2_LANG[lang]} to {save_dir}", type="INFO")
            model.save_pretrained(save_dir)
            # Also save the tokenizer used for this language model
            tokenizer_lang = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            tokenizer_lang.save_pretrained(save_dir)
            logger.log(f"Model and tokenizer for {ID_2_LANG[lang]} saved successfully.", type="SUCCESS")
        else:
            logger.log(f"Fine-tuning failed for language {ID_2_LANG[lang]}. It will be excluded from distillation.", type="ERROR")
        logger.end_timer(f"finetune_model_{ID_2_LANG[lang]}")
    logger.log("Teacher model fine-tuning finished.", type="INFO")
else:
    logger.log("All required teacher models were found and loaded.", type="INFO")
    print("All required teacher models were found and loaded.")

# Remove languages for which training failed from the teacher_models dict
valid_langs = list(teacher_models.keys())
logger.log(f"Using teacher models for languages: {valid_langs}", type="INFO", languages=valid_langs)


# %% Prepare DataLoaders for Distillation
logger.start_timer("prepare_dataloaders")
logger.log("Preparing data for distillation...", type="INFO")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME) # Use base tokenizer for student

# Tokenize the entire dataset once
tokenized_dataset = complete_dataset.map(lambda ex: tokenize_data(tokenizer, ex), batched=True,
                                         remove_columns=['sentence', 'lang']) # Remove unused columns

# Ensure 'labels' column exists after transformations
if 'labels' not in tokenized_dataset['train'].column_names:
    logger.log("Column 'labels' not found in the tokenized training dataset.", type="ERROR")
    raise ValueError("Column 'labels' not found in the tokenized training dataset. Check data processing steps.")


# Set format for PyTorch
try:
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    logger.log("Dataset format set to PyTorch.", type="INFO")
except ValueError as e:
    logger.log(f"Error setting dataset format: {e}", type="ERROR", 
               available_columns=tokenized_dataset['train'].column_names)
    # Handle error appropriately, maybe raise or default columns
    raise

train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False) # No need to shuffle eval
logger.log("DataLoaders created.", type="INFO", 
           train_size=len(train_dataset), eval_size=len(eval_dataset), batch_size=BATCH_SIZE)
logger.end_timer("prepare_dataloaders")


# %% Initialize Student Model
logger.start_timer("init_student_model")
logger.log("Initializing student model...", type="INFO")
student_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=3)
student_model.to(device)
logger.log("Student model initialized.", type="SUCCESS")
logger.end_timer("init_student_model")

# %% Setup Distillation Training
optimizer = Adam(student_model.parameters(), lr=LEARNING_RATE)
# Use torch.amp.GradScaler instead of torch.cuda.amp.GradScaler which is deprecated
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available()) # Enable scaler only if using CUDA

logger.log("Setting up distillation trainer.", type="INFO", 
           learning_rate=LEARNING_RATE, temperature=TEMPERATURE, alpha=ALPHA)

distillation_trainer = DistillationTrainer(
    student_model=student_model,
    teacher_models=teacher_models, # Use only successfully loaded/trained models
    train_loader=train_loader,
    eval_loader=eval_loader,
    optimizer=optimizer,
    scaler=scaler,
    device=device,
    temperature=TEMPERATURE,
    alpha=ALPHA,
    num_epochs=DISTILLATION_EPOCHS,
    logger=logger  # Pass the logger to the trainer
)

# %% Run Distillation Training
logger.start_timer("distillation_training")
logger.log("Starting distillation training...", type="INFO", epochs=DISTILLATION_EPOCHS)
distillation_trainer.train()
logger.end_timer("distillation_training")

# %% Final Evaluation
logger.start_timer("final_evaluation")
logger.log("Performing final evaluation of the student model...", type="INFO")
final_metrics = evaluate(student_model, eval_loader, device)
logger.log("Final Evaluation Metrics", type="RESULTS", metrics=final_metrics)
logger.end_timer("final_evaluation")
# Log final metrics (optional, requires tensorboard or similar)
distillation_trainer.writer.add_text("Final Metrics", str(final_metrics))
distillation_trainer.writer.close()

# %% Save Student Model
logger.start_timer("save_model")
logger.log(f"Saving student model and tokenizer to '{STUDENT_MODEL_SAVE_PATH}'...", type="INFO")
student_model.save_pretrained(STUDENT_MODEL_SAVE_PATH)
tokenizer.save_pretrained(STUDENT_MODEL_SAVE_PATH)
logger.log("Student model and tokenizer saved.", type="SUCCESS")
logger.end_timer("save_model")

# %% Optional: Evaluate on Synthetic Data (if needed)
logger.start_timer("synthetic_evaluation")
logger.log("Loading synthetic dataset...", type="INFO")
synthetic_ds = load_dataset("nojedag/synthetic_financial_sentiment")
synthetic_dataset = synthetic_ds.map(transform_labels, batched=True, remove_columns=['sentiment'])
logger.log("Tokenizing synthetic data...", type="INFO")
tokenized_synthetic = synthetic_dataset.map(lambda ex: tokenize_data(tokenizer, ex), batched=True, remove_columns=['sentence', 'lang'])
tokenized_synthetic.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
synthetic_loader = DataLoader(tokenized_synthetic['train'], batch_size=BATCH_SIZE) # Assuming 'train' split
logger.log("Evaluating student model on synthetic data...", type="INFO")
synthetic_metrics = evaluate(student_model, synthetic_loader, device)
logger.log("Synthetic Data Evaluation Metrics", type="RESULTS", metrics=synthetic_metrics)
print("Synthetic Data Evaluation Metrics:", synthetic_metrics)
logger.end_timer("synthetic_evaluation")

# Push to Hugging Face Hub (if needed)
logger.start_timer("hub_upload")
logger.log("Pushing model to Hugging Face Hub...", type="INFO")
student_model.push_to_hub(f'nojedag/{STUDENT_MODEL_NAME}')
logger.log("Model pushed to Hub successfully.", type="SUCCESS")
logger.end_timer("hub_upload")

# %% Dump all logs to file
log_file_path = logger.dump_to_file()
logger.log(f"All logs saved to: {log_file_path}", type="INFO")

logger.log("Script finished.", type="INFO", total_runtime=f"{logger.get_execution_time():.2f} seconds")