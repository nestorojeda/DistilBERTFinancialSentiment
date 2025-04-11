import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification
import os

class DistillationTrainer:
    """Handles the knowledge distillation training process."""
    def __init__(self, student_model, teacher_models, train_loader, eval_loader,
                 optimizer, scaler, device, temperature=2.0, alpha=0.5,
                 num_epochs=5, log_dir="runs/student_model_logs"):
        self.student_model = student_model
        self.teacher_models = teacher_models # Dict of lang -> model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(log_dir)
        self.langs = list(teacher_models.keys())

        # Loss functions
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def _distillation_loss(self, student_logits, teacher_logits):
        """Calculates the Kullback-Leibler divergence loss for distillation."""
        student_log_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        # Ensure target tensor requires no grad
        return self.kl_loss_fn(student_log_probs, teacher_probs.detach()) * (self.temperature ** 2)

    def train_epoch(self, epoch, global_step):
        """Runs a single training epoch."""
        self.student_model.train()
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels') # Use .get for safety
            if labels is None:
                print("Warning: 'labels' key not found in batch. Skipping batch for training.")
                continue
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                # Get average teacher logits
                with torch.no_grad():
                    teacher_logits_sum = None
                    for lang in self.langs:
                        teacher = self.teacher_models[lang]
                        teacher.eval() # Ensure teacher is in eval mode
                        outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                        if teacher_logits_sum is None:
                            teacher_logits_sum = outputs.logits
                        else:
                            teacher_logits_sum += outputs.logits
                    if teacher_logits_sum is None:
                         print("Warning: No teacher logits were generated. Skipping batch.")
                         continue # Skip if no teacher logits found
                    teacher_logits_avg = teacher_logits_sum / len(self.teacher_models)

                # Get student logits
                student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                # Calculate losses
                loss_ce = self.ce_loss_fn(student_logits, labels)
                loss_kd = self._distillation_loss(student_logits, teacher_logits_avg)
                loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Log training loss
            self.writer.add_scalar("Training Loss", loss.item(), global_step)
            global_step += 1

            if global_step % 100 == 0:
                print(f"Epoch: {epoch+1}/{self.num_epochs}, Step: {global_step}, Loss: {loss.item():.4f}")
        return global_step

    def train(self):
        """Runs the full training process."""
        global_step = 0
        print("Starting distillation training...")
        for epoch in range(self.num_epochs):
            print(f"--- Epoch {epoch+1}/{self.num_epochs} ---")
            global_step = self.train_epoch(epoch, global_step)
            # Optional: Add evaluation after each epoch
            # eval_metrics = self.evaluate()
            # print(f"Epoch {epoch+1} Evaluation Metrics: {eval_metrics}")
            # self.writer.add_scalar("Evaluation Accuracy", eval_metrics['accuracy'], global_step)

        print("Distillation training finished.")
        self.writer.close() # Close the TensorBoard writer when done 