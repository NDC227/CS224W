import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaselineSeismicGNN(nn.Module):
    def __init__(self, in_channels=3, time_steps=500, hidden_dim=64, pos_dim=3):
        super(BaselineSeismicGNN, self).__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels
        
        # 1. Temporal Waveform Encoder (processes observed T x C waveform per station)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + 1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # 2. Static Position & Metadata Embedder
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Spatial Graph Convolutional Layers
        self.spatial_gcn1 = GCNConv(hidden_dim * 16 + hidden_dim, hidden_dim)
        self.spatial_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # 4. Temporal Waveform Decoder (maps latent representation back to T x C)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, time_steps * in_channels)
        )

    def forward(self, x, pos, edge_index, mask_channel):
        """
        Args:
            x: Input waveforms tensor of shape (N, T, C)
            pos: 3D spatial coordinates tensor of shape (N, 3)
            edge_index: Graph topology edge indices tensor of shape (2, E)
            mask_channel: Binary indicator channel tensor of shape (N, 1), 
                          where 1.0 indicates a masked target node to reconstruct.
        """
        N, T, C = x.shape
        
        # Append mask indicator to input channels -> Shape: (N, T, C+1)
        mask_expanded = mask_channel.unsqueeze(1).expand(-1, T, -1)
        x_in = torch.cat([x, mask_expanded], dim=-1)
        
        # Temporal Conv expects (N, Channels, T)
        x_in = x_in.transpose(1, 2)
        h_temp = self.temporal_encoder(x_in).view(N, -1)  # Shape: (N, hidden_dim)
        
        # Process spatial position coordinates
        h_pos = self.pos_encoder(pos)                     # Shape: (N, hidden_dim)
        
        # Combine temporal signal and spatial positional embeddings
        h = torch.cat([h_temp, h_pos], dim=-1)            # Shape: (N, hidden_dim * 2)
        
        # Perform spatial message passing across all nodes
        h = F.relu(self.spatial_gcn1(h, edge_index))
        h = F.relu(self.spatial_gcn2(h, edge_index))

        # Automatically find target nodes where mask_channel == 1
        target_indices = torch.where(mask_channel.squeeze(-1) != 1.0)[0]

        if len(target_indices) == 0:
            raise ValueError("No masked target nodes found in mask_channel (all values are 0).")
        
        # Isolate only the target node embeddings -> Shape: (num_targets, hidden_dim * 2)
        h_target = h[target_indices]                      
        
        # Decode only for the target nodes -> Shape: (num_targets, T * C)
        out = self.decoder(h_target)                      
        out = out.view(-1, T, C)                          # Shape: (num_targets, T, C)
        
        # If exactly one target node was masked, squeeze the batch/node dimension to output (T, C)
        if out.size(0) == 1:
            out = out.squeeze(0)                          # Shape: (T, C)
            
        return out