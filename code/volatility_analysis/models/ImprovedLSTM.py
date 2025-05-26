import torch
import torch.nn as nn
import random
import numpy as np

# LSTM Model for Volatility Prediction with Attention and Dropout
class ImprovedLSTMVolatility(nn.Module):
    """Enhanced LSTM with attention and dropout."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2, seed: int = 42) -> None:
        super(ImprovedLSTMVolatility, self).__init__()
        set_seed(seed)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual connection
        
        # Take the last output
        out = attn_out[:, -1, :]
        out = self.dropout(out)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
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
