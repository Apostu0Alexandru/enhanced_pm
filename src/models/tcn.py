import torch.nn as nn

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels=128):
        super().__init__()
        # Placeholder for TCN implementation
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_channels, output_size, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        x = x.transpose(1, 2)  # (batch_size, num_features, seq_len)
        return self.tcn(x).transpose(1, 2)

from pytorch_tcn import TCN

def build_tcn_model(input_shape, num_channels=128, kernel_size=4, dropout=0.1):
    """
    Build a Temporal Convolutional Network model for RUL prediction.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        num_channels: Number of TCN channels (can be a list for multiple layers)
        kernel_size: Size of the convolutional kernel
        dropout: Dropout rate
        
    Returns:
        Compiled TCN model
    """
    # If num_channels is a single value, convert to list for TCN
    if isinstance(num_channels, int):
        num_channels = [num_channels] * 4  # 4 layers of TCN
    
    # Get input dimensions
    seq_len, n_features = input_shape
    
    # Create TCN model
    model = TCN(
        num_inputs=n_features,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        causal=True,
        use_skip_connections=True,
        input_shape='NCL'  # (batch, channels, length)
    )
    
    # Wrap in a custom module for simpler interface
    class TCNModel(nn.Module):
        def __init__(self, tcn, output_size=1):
            super(TCNModel, self).__init__()
            self.tcn = tcn
            self.linear = nn.Linear(num_channels[-1], output_size)
            
        def forward(self, x):
            # TCN expects input shape (batch, channels, length)
            # But our data might be (batch, length, channels)
            if x.shape[1] != n_features:
                x = x.transpose(1, 2)
            
            y = self.tcn(x)
            # Take the last output for sequence prediction
            return self.linear(y[:, :, -1])
    
    return TCNModel(model, output_size=1)
