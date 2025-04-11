# %% Imports
import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import Adam
from torch.cuda.amp import GradScaler

# Local imports from the refactored modules
from code.data_manager import transform_labels, tokenize_data
from code.evaluation import evaluate
from code.model_trainer import fine_tune_language
from code.distillation import DistillationTrainer
from utils import get_output_dir # Assuming utils.py and get_output_dir exist

# %% Configuration
BASE_MODEL_NAME = "FacebookAI/xlm-roberta-base"
PROJECT_NAME_PREFIX = "xlm-roberta-base-finetuned-financial-phrasebank"
LANGS = ['en', 'de', 'es', 'fr']
STUDENT_MODEL_SAVE_PATH = "saved_student_model"
DISTILLATION_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
TEMPERATURE = 2.0
ALPHA = 0.5 # Weight for CE loss in distillation

# %% Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Load and Prepare Base Dataset
print("Loading base dataset...")
ds = load_dataset("nojedag/financial_phrasebank_multilingual")
complete_dataset = ds.map(transform_labels, batched=True, remove_columns=['sentiment']) # Remove original sentiment column
print("Base dataset loaded and labels transformed.")

# %% Fine-tune or Load Teacher Models
teacher_models = {}
langs_to_fine_tune = []

print("Checking for existing teacher models...")
for lang in LANGS:
    model_output_dir = get_output_dir(f'{PROJECT_NAME_PREFIX}-{lang}')
    if os.path.exists(os.path.join(model_output_dir, "pytorch_model.bin")): # More specific check
        try:
            print(f"Loading existing teacher model for {lang} from {model_output_dir}")
            teacher_models[lang] = AutoModelForSequenceClassification.from_pretrained(model_output_dir)
            teacher_models[lang].to(device)
            print(f"Teacher model for {lang} loaded successfully.")
        except Exception as e:
            print(f"Error loading model for {lang} from {model_output_dir}: {e}. Will attempt fine-tuning.")
            langs_to_fine_tune.append(lang)
    else:
        print(f"Model directory for {lang} ({model_output_dir}) does not exist or is incomplete. Scheduling for fine-tuning.")
        langs_to_fine_tune.append(lang)

# Fine-tune models for languages that were not found or failed to load
if langs_to_fine_tune:
    print(f"Starting fine-tuning for languages: {langs_to_fine_tune}")
    for lang in langs_to_fine_tune:
        print(f"--- Training teacher model for {lang} ---")
        model = fine_tune_language(BASE_MODEL_NAME, complete_dataset, lang)
        if model:
            teacher_models[lang] = model.to(device) # Ensure model is on the correct device
            save_dir = get_output_dir(f'{PROJECT_NAME_PREFIX}-{lang}')
            print(f"Saving fine-tuned model for {lang} to {save_dir}")
            model.save_pretrained(save_dir)
            # Also save the tokenizer used for this language model
            tokenizer_lang = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            tokenizer_lang.save_pretrained(save_dir)
            print(f"Model and tokenizer for {lang} saved successfully.")
        else:
             print(f"Fine-tuning failed for language {lang}. It will be excluded from distillation.")
    print("Teacher model fine-tuning finished.")
else:
    print("All required teacher models were found and loaded.")

# Remove languages for which training failed from the teacher_models dict
valid_langs = list(teacher_models.keys())
print(f"Using teacher models for languages: {valid_langs}")


# %% Prepare DataLoaders for Distillation
print("Preparing data for distillation...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME) # Use base tokenizer for student

# Tokenize the entire dataset once
tokenized_dataset = complete_dataset.map(lambda ex: tokenize_data(ex, tokenizer), batched=True,
                                         remove_columns=['sentence', 'lang']) # Remove unused columns

# Ensure 'labels' column exists after transformations
if 'labels' not in tokenized_dataset['train'].column_names:
     raise ValueError("Column 'labels' not found in the tokenized training dataset. Check data processing steps.")


# Set format for PyTorch
try:
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
except ValueError as e:
    print(f"Error setting dataset format: {e}")
    print("Available columns:", tokenized_dataset['train'].column_names)
    # Handle error appropriately, maybe raise or default columns
    raise

train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False) # No need to shuffle eval
print("DataLoaders created.")


# %% Initialize Student Model
print("Initializing student model...")
student_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=3)
student_model.to(device)
print("Student model initialized.")

# %% Setup Distillation Training
optimizer = Adam(student_model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(enabled=torch.cuda.is_available()) # Enable scaler only if using CUDA

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
    num_epochs=DISTILLATION_EPOCHS
)

# %% Run Distillation Training
distillation_trainer.train()

# %% Final Evaluation
print("Performing final evaluation of the student model...")
final_metrics = evaluate(student_model, eval_loader, device)
print("Final Evaluation Metrics:", final_metrics)

# Log final metrics (optional, requires tensorboard or similar)
# distillation_trainer.writer.add_text("Final Metrics", str(final_metrics))
# distillation_trainer.writer.close()

# %% Save Student Model
print(f"Saving student model and tokenizer to '{STUDENT_MODEL_SAVE_PATH}'...")
student_model.save_pretrained(STUDENT_MODEL_SAVE_PATH)
tokenizer.save_pretrained(STUDENT_MODEL_SAVE_PATH)
print("Student model and tokenizer saved.")

# %% Optional: Evaluate on Synthetic Data (if needed)
# print("Loading synthetic dataset...")
# synthetic_ds = load_dataset("nojedag/synthetic_financial_sentiment")
# synthetic_dataset = synthetic_ds.map(transform_labels, batched=True, remove_columns=['sentiment'])
# print("Tokenizing synthetic data...")
# tokenized_synthetic = synthetic_dataset.map(lambda ex: tokenize_data(ex, tokenizer), batched=True, remove_columns=['sentence', 'lang'])
# tokenized_synthetic.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# synthetic_loader = DataLoader(tokenized_synthetic['train'], batch_size=BATCH_SIZE) # Assuming 'train' split
# print("Evaluating student model on synthetic data...")
# synthetic_metrics = evaluate(student_model, synthetic_loader, device)
# print("Synthetic Data Evaluation Metrics:", synthetic_metrics)

print("Script finished.")


