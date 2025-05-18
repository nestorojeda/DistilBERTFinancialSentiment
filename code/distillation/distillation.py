import os
import sys
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to sys.path")
from toolbox.logger import Logger  # Updated import from dedicated logger module

class DistillationTrainer:
    """Handles the knowledge distillation training process."""
    def __init__(self, student_model, teacher_models, train_loader, eval_loader,
                 optimizer, scaler, device, temperature=2.0, alpha=0.5,
                 num_epochs=5, log_dir="runs/student_model_logs", logger=None):
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
        
        # Logger setup
        self.logger = logger
        if self.logger:
            self.logger.log("DistillationTrainer initialized", type="INFO", 
                           temperature=temperature, alpha=alpha, 
                           num_epochs=num_epochs, languages=self.langs)

        # Loss functions
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def _log(self, message, type="INFO", **kwargs):
        """Helper method to handle logging both to console and Logger if available"""
        print(message)
        if self.logger:
            self.logger.log(message, type=type, **kwargs)

    def _distillation_loss(self, student_logits, teacher_logits):
        """Calculates the Kullback-Leibler divergence loss for distillation."""
        student_log_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        # Ensure target tensor requires no grad
        return self.kl_loss_fn(student_log_probs, teacher_probs.detach()) * (self.temperature ** 2)

    def train_epoch(self, epoch, global_step):
        """Runs a single training epoch."""
        self.student_model.train()
        epoch_loss = 0
        batch_count = 0
        
        if self.logger:
            self.logger.start_timer(f"epoch_{epoch+1}")
            self.logger.log(f"Starting epoch {epoch+1}/{self.num_epochs}", type="EPOCH_START")
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels') # Use .get for safety
            
            if labels is None:
                self._log("Warning: 'labels' key not found in batch. Skipping batch for training.", type="WARNING")
                continue
                
            labels = labels.to(self.device)
            batch_size = input_ids.size(0)
            batch_count += 1

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
                        self._log("Warning: No teacher logits were generated. Skipping batch.", type="WARNING")
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
            
            # Track losses
            epoch_loss += loss.item()

            # Log training loss to TensorBoard
            self.writer.add_scalar("Training Loss", loss.item(), global_step)
            
            # Log detailed metrics periodically
            if global_step % 100 == 0:
                metrics = {
                    "loss": loss.item(),
                    "ce_loss": loss_ce.item(),
                    "kd_loss": loss_kd.item(),
                    "batch_size": batch_size,
                    "global_step": global_step
                }
                self._log(f"Epoch: {epoch+1}/{self.num_epochs}, Step: {global_step}, Loss: {loss.item():.4f}", 
                         type="TRAINING", **metrics)
                
            global_step += 1
            
        # End of epoch logging
        avg_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')
        if self.logger:
            self.logger.log(f"Epoch {epoch+1} completed", type="EPOCH_END",
                           avg_loss=avg_loss, global_step=global_step)
            self.logger.end_timer(f"epoch_{epoch+1}")
            
        return global_step

    def train(self):
        """Runs the full training process."""
        global_step = 0
        self._log("Starting distillation training...", type="TRAINING_START", epochs=self.num_epochs)
        
        for epoch in range(self.num_epochs):
            self._log(f"--- Epoch {epoch+1}/{self.num_epochs} ---", type="EPOCH")
            global_step = self.train_epoch(epoch, global_step)
            
            # Optional: Add evaluation after each epoch
            from evaluation import evaluate # Import here to avoid circular import
            eval_metrics = evaluate(self.student_model, self.eval_loader, self.device)
            self._log(f"Epoch {epoch+1} Evaluation Metrics", type="EVAL", metrics=eval_metrics)
            self.writer.add_scalar("Evaluation Accuracy", eval_metrics['accuracy'], global_step)

        self._log("Distillation training finished.", type="TRAINING_END", 
                 steps_completed=global_step, final_epoch=self.num_epochs)
        
        self.writer.close() # Close the TensorBoard writer when done