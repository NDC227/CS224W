import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaselineEncoder(nn.Module):
    """
    Baseline Temporal-Spatial GNN Encoder:
    Processes temporal waveforms and spatial coordinates across all nodes.
    """
    def __init__(self, in_channels=3, time_steps=500, hidden_dim=64, pos_dim=3):
        super(BaselineEncoder, self).__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + 1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.spatial_gcn1 = GCNConv(hidden_dim * 16 + hidden_dim, hidden_dim)
        self.spatial_gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, pos, edge_index, mask_channel):
        """
        Returns:
            h: Node representations of shape (N, hidden_dim) for all nodes.
        """
        N, T, C = x.shape
        
        # Append mask indicator -> Shape: (N, T, C+1)
        mask_expanded = mask_channel.unsqueeze(1).expand(-1, T, -1)
        x_in = torch.cat([x, mask_expanded], dim=-1)
        
        # Temporal Conv expects (N, Channels, T)
        x_in = x_in.transpose(1, 2)
        h_temp = self.temporal_encoder(x_in).view(N, -1) 
        
        h_pos = self.pos_encoder(pos)                     
        h = torch.cat([h_temp, h_pos], dim=-1)            
        
        # Spatial message passing across all nodes
        h = F.relu(self.spatial_gcn1(h, edge_index))
        h = F.relu(self.spatial_gcn2(h, edge_index))      
        
        return h


class BaselineDecoder(nn.Module):
    """
    Baseline Temporal Waveform Decoder:
    Isolates the target node and maps its latent representation back to a single T x C waveform.
    """
    def __init__(self, in_channels=3, time_steps=500, hidden_dim=64):
        super(BaselineDecoder, self).__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, time_steps * in_channels)
        )

    def forward(self, h, mask_channel):
        """
        Args:
            h: Enriched node representations of shape (N, hidden_dim) from the encoder
            mask_channel: Binary indicator tensor of shape (N, 1) where target = 1.0
        Returns:
            out: Reconstructed waveform of shape (T, C) for the target node
        """
        T, C = self.time_steps, self.in_channels
        
        # Automatically find target nodes where mask_channel == 0
        target_indices = torch.where(mask_channel.squeeze(-1) == 0.0)[0]
        
        if len(target_indices) == 0:
            target_indices = torch.where(mask_channel.squeeze(-1) != 1.0)[0]
            
        if len(target_indices) == 0:
            raise ValueError("No masked target nodes found in mask_channel.")
        
        # Isolate only the target node embedding(s) -> Shape: (num_targets, hidden_dim)
        h_target = h[target_indices]                      
        
        # Decode for target nodes -> Shape: (num_targets, T * C)
        out = self.net(h_target)                          
        out = out.view(-1, T, C)                          # Shape: (num_targets, T, C)
        
        # Squeeze to return shape (T, C) for the single target waveform
        if out.size(0) == 1:
            out = out.squeeze(0)                          # Shape: (T, C)
        
        return out


class BaselineSeismicGNN(nn.Module):
    """
    Wrapper class combining the Baseline Encoder and Decoder.
    Outputs a single reconstructed waveform of shape (T, C).
    """
    def __init__(self, in_channels=3, time_steps=500, hidden_dim=64, pos_dim=3):
        super(BaselineSeismicGNN, self).__init__()
        self.encoder = BaselineEncoder(in_channels, time_steps, hidden_dim, pos_dim)
        self.decoder = BaselineDecoder(in_channels, time_steps, hidden_dim)

    def forward(self, x, pos, edge_index, mask_channel, batch=None):
        h = self.encoder(x, pos, edge_index, mask_channel)
        out = self.decoder(h, mask_channel)
        return out