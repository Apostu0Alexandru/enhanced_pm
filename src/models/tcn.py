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
