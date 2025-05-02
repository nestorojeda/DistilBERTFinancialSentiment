import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class LRFinder:
    """Learning rate finder class to find optimal learning rate"""
    
    def __init__(self, model, optimizer, criterion, device, weight_tensor=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Store original weight tensor if provided
        self.weight_tensor = weight_tensor.to(device) if weight_tensor is not None else None
        
        # Create criterion on the correct device
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.weight_tensor
            )
        
        # Initialize lists to store learning rates and corresponding losses
        self.lrs = []
        self.losses = []
        
        # Set best loss to infinity
        self.best_loss = float('inf')
        
        # Flag to stop training if diverging
        self.stop_training = False
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=1, num_iter=100, step_mode='exp', smooth_f=0.05, diverge_th=5):
        """Performs the learning rate range test
        
        Arguments:
            train_loader (torch.utils.data.DataLoader): The training data loader
            start_lr (float): The starting learning rate
            end_lr (float): The maximum learning rate
            num_iter (int): Number of iterations over which to test
            step_mode (str): 'exp' or 'linear' learning rate increase
            smooth_f (float): The loss smoothing factor in range [0, 1]
            diverge_th (float): The divergence threshold
            
        Returns:
            list, list: The learning rates and corresponding losses
        """
        # Set model to training mode
        self.model.train()
        
        # Reset state
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')
        self.stop_training = False
        
        # Set initial learning rate
        self._set_learning_rate(start_lr)
        
        # Compute the learning rate multiplier based on step mode
        if step_mode == 'exp':
            lr_multiplier = (end_lr / start_lr) ** (1 / num_iter)
        else:
            lr_multiplier = (end_lr - start_lr) / num_iter
        
        # Initialize data iterator
        iterator = iter(train_loader)
        
        # Run the range test for num_iter iterations or until stop_training is triggered
        for iteration in tqdm(range(num_iter)):
            # Get a batch (try to get a new batch or restart if exhausted)
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs = next(iterator)
            
            # Extract labels first, then prepare inputs
            labels = inputs.pop('labels', None)
            
            # Move inputs and labels to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if labels is not None:
                labels = labels.to(self.device)
            else:
                print("Warning: No labels found in batch. Loss calculation will fail.")
                continue  # Skip this iteration if no labels
            
            # Ensure the model is on the correct device
            self.model.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            
            # Ensure we have labels for loss calculation
            if labels is None:
                print("Skipping iteration due to missing labels")
                continue
            
            # Check if criterion has weights and make sure they're on the same device
            if hasattr(self.criterion, 'weight') and self.criterion.weight is not None:
                if self.criterion.weight.device != self.device:
                    # Recreate criterion with weight on the correct device
                    self.criterion = torch.nn.CrossEntropyLoss(
                        weight=self.weight_tensor.to(self.device)
                    )
                
            loss = self.criterion(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record learning rate and loss
            self.lrs.append(self._get_learning_rate())
            # Apply smoothing if requested
            if iteration == 0:
                self.losses.append(loss.item())
            else:
                self.losses.append(smooth_f * loss.item() + (1 - smooth_f) * self.losses[-1])
            
            # Check if loss is diverging
            if self._is_diverging(diverge_th):
                print(f'Stopping early at iteration {iteration}: Loss {self.losses[-1]:.4f} > {diverge_th} * minimal loss {self.best_loss:.4f}')
                break
            
            # Update learning rate
            if step_mode == 'exp':
                self._set_learning_rate(self.lrs[-1] * lr_multiplier)
            else:
                self._set_learning_rate(self.lrs[-1] + lr_multiplier)
        
        return self.lrs, self.losses
    
    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
        """Plots the learning rate range test
        
        Arguments:
            skip_start (int): Number of batches to skip at the start
            skip_end (int): Number of batches to skip at the end
            log_lr (bool): Whether to use log scale for x-axis
            show_lr (float): The learning rate to show as optimal (if provided)
        """
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
            
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
            
        # Get the data to plot
        lrs = self.lrs[skip_start:] if skip_end == 0 else self.lrs[skip_start:-skip_end]
        losses = self.losses[skip_start:] if skip_end == 0 else self.losses[skip_start:-skip_end]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot loss vs learning rate
        if log_lr:
            plt.semilogx(lrs, losses)
        else:
            plt.plot(lrs, losses)
            
        # Find the learning rate with steepest negative gradient
        if len(lrs) > 1:
            # Calculate derivatives
            derivatives = np.gradient(losses, np.log10(lrs) if log_lr else lrs)
            # Find minimum derivative (steepest descent)
            min_der_idx = np.argmin(derivatives)
            suggested_lr = lrs[min_der_idx]
            
            # Show the suggested learning rate
            plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
            
        # Show provided learning rate if any
        if show_lr is not None:
            plt.axvline(x=show_lr, color='g', linestyle='--', label=f'Current LR: {show_lr:.2e}')
        
        # Add labels and legend
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Loss vs. Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        if len(lrs) > 1:
            return suggested_lr
        else:
            return None
    
    def suggest_lr(self, skip_start=10, skip_end=5, log_lr=True):
        """Suggests an optimal learning rate based on the range test
        
        Arguments:
            skip_start (int): Number of batches to skip at the start
            skip_end (int): Number of batches to skip at the end
            log_lr (bool): Whether to use log scale for x-axis
            
        Returns:
            float: The suggested learning rate
        """
        # Get the data
        lrs = self.lrs[skip_start:] if skip_end == 0 else self.lrs[skip_start:-skip_end]
        losses = self.losses[skip_start:] if skip_end == 0 else self.losses[skip_start:-skip_end]
        
        if len(lrs) > 1:
            # Calculate derivatives
            derivatives = np.gradient(losses, np.log10(lrs) if log_lr else lrs)
            # Find minimum derivative (steepest descent)
            min_der_idx = np.argmin(derivatives)
            return lrs[min_der_idx]
        else:
            return None
    
    def _set_learning_rate(self, lr):
        """Sets the learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_learning_rate(self):
        """Gets the current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def _is_diverging(self, diverge_th):
        """Check if the loss is diverging"""
        # Update best loss
        if self.losses[-1] < self.best_loss:
            self.best_loss = self.losses[-1]
        
        # Check if loss is diverging
        if self.losses[-1] > diverge_th * self.best_loss:
            return True
        
        return False