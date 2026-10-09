import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ConvWaveformDecoder(nn.Module):
    def __init__(self, hidden_dim=64, out_channels=3, target_length=500):
        super().__init__()
        self.target_length = target_length
        self.fc = nn.Linear(hidden_dim, 128 * 31)  # Expand latent state
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 62
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 124
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),   # -> 248
            nn.ReLU(),
            nn.ConvTranspose1d(16, out_channels, kernel_size=6, stride=2, padding=1), # -> 500
        )

    def forward(self, h):
        # h shape: (N, hidden_dim)
        x = F.relu(self.fc(h)).view(-1, 128, 31)
        out = self.deconv(x)  # Shape: (N, out_channels, 500)
        return out.transpose(1, 2)  # Shape: (N, 500, out_channels)