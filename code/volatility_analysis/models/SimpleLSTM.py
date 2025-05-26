import torch
import torch.nn as nn
import random
import numpy as np

# LSTM Model for Volatility Prediction
class LSTMVolatility(nn.Module):
    """
    LSTM Model for volatility prediction.
    
    Args:
        input_size: Number of input features.
        hidden_size: Size of hidden layers.
        num_layers: Number of LSTM layers.
        output_size: Number of output values.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, output_size: int = 1, seed: int = 42) -> None:
        super(LSTMVolatility, self).__init__()
        
        set_seed(seed)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    def _init_weights(self):
        """Initialize weights for better reproducibility."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for reproducible results
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    