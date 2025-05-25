import torch
import torch.nn as nn

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
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, output_size: int = 1) -> None:
        super(LSTMVolatility, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out