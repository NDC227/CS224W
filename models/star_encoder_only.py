import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.star.star import TransformerModel
from models.baseline_model import BaselineDecoder

class STAREncoder(nn.Module):
    """
    Adapted STAR (Spatio-Temporal Graph Transformer) Encoder for Seismic Waves.
    Outputs latent node representations of shape (N, hidden_dim) to match the baseline encoder.
    """
    def __init__(self, in_channels=3, time_steps=500, hidden_dim=64, pos_dim=3, dropout_prob=0.1):
        super(STAREncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        
        # 1. Temporal feature extractor for high-frequency waveforms + mask channel
        # Projects input (in_channels + 1) to hidden_dim
        self.temporal_tcn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels + 1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16) # Compresses time dimension into fixed temporal features
        )
        
        # 2. Static spatial attribute embedder (coordinates, V_s30, etc.)
        self.spatial_embedder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. STAR Dual-Encoder Transformer Backbone
        self.temporal_encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout_prob)
        
        # Note: TransformerModel refers to your custom or imported STAR spatial attention module
        # Ensure it accepts and outputs hidden_dim (ninp)
        self.spatial_encoder_1 = TransformerModel(ninp=hidden_dim, nhead=8, nhid=2048, nlayers=2, dropout=dropout_prob)
        self.spatial_encoder_2 = TransformerModel(ninp=hidden_dim, nhead=8, nhid=2048, nlayers=2, dropout=dropout_prob)
        
        self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
        self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer, 1)
        
        # 4. Fusion and Refinement layers matching hidden_dim
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Final projection to ensure exact dimension match with baseline encoder
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(self.dropout_prob)

    def forward(self, x_obs, mask, static_metadata, nei_list):
        """
        Args:
            x_obs: Observed waveforms tensor of shape (N, T, C)
            mask: Binary indicator tensor of shape (N, 1) or (N, T, 1)
            static_metadata: Static attributes tensor of shape (N, pos_dim)
            nei_list: Neighbor index tensor for spatial attention graph connectivity
        Returns:
            final_latent_embeddings: Tensor of shape (N, hidden_dim) matching the baseline format.
        """
        N, T, C = x_obs.shape
        
        # Ensure mask is formatted correctly for concatenation -> Shape: (N, T, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).expand(-1, T, -1)
        if mask.dim() == 3 and mask.size(-1) == 3:
            # If the mask is duplicated across channels, take the first one or reduce
            mask = mask[:, :, :1]  # Shape becomes (6, 6000, 1)
            
        # 1. Extract temporal features via TCN -> Shape: (N, hidden_dim, 16) -> Transpose/Pool to (N, hidden_dim) or keep sequence length
        # (Assuming temporal_tcn output matches sequence dimension expected by temporal encoder)
        x_combined = torch.cat([x_obs, mask], dim=-1).transpose(1, 2) # Shape: (N, C+1, T)
        # print("x_obs", x_obs.shape)
        # print("mask", mask.shape)
        # print("x_combined", x_combined.shape)
        temporal_feats = self.temporal_tcn(x_combined).transpose(1, 2) # Shape: (N, 16, hidden_dim) -> or adjust based on your TCN design
        
        # 2. Embed static spatial attributes -> Shape: (N, hidden_dim)
        static_metadata = static_metadata.float()
        spatial_feats = self.spatial_embedder(static_metadata) 
        
        # 3. Pass through STAR's dual encoder blocks
        spatial_out = self.spatial_encoder_1(spatial_feats.unsqueeze(1), nei_list) # Shape: (N, 1, hidden_dim)
        spatial_out = spatial_out.squeeze(1) # Shape: (N, hidden_dim)
        
        temporal_out = self.temporal_encoder_1(temporal_feats) # Shape: (N, seq_len, hidden_dim)
        temporal_out = temporal_out.mean(dim=1) # Pool across time if needed -> Shape: (N, hidden_dim)
        
        # Fusion
        fusion_feat = self.fusion_layer(torch.cat((temporal_out, spatial_out), dim=-1)) # Shape: (N, hidden_dim)
        
        # Final refinement layers
        spatial_refined = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list).squeeze(1)
        final_latent = self.temporal_encoder_2(spatial_refined.unsqueeze(1)).squeeze(1)
        
        final_latent_embeddings = self.output_projection(final_latent) # Shape: (N, hidden_dim)
        
        return final_latent_embeddings

class STAREncoderSeismicGNN(nn.Module):
    """
    Experimental Model Wrapper:
    Maintains the exact same forward signature as the baseline model 
    (x, pos, edge_index, mask_channel), making it 100% compatible with 
    your unified training loop.
    """
    def __init__(self, star_encoder, decoder):
        super(STAREncoderSeismicGNN, self).__init__()
        self.encoder = star_encoder
        self.decoder = decoder

    def encode(self, x, pos, edge_index, mask_channel):
        """
        Matches the baseline forward signature while translating PyG inputs 
        into what the STAR encoder expects internally.
        """
        N = x.size(0)
        
        # 1. Internal translation: Convert PyG edge_index to nei_list if required by STAR,
        # or construct an attention adjacency mask from edge_index.
        # (Example: converting sparse edge_index to a dense neighbor/adjacency matrix for STAR)
        nei_list = self._edge_index_to_nei_list(edge_index, num_nodes=N)
        
        # 2. Pass through the adapted STAR encoder using standard names
        # - x           -> x_obs
        # - mask_channel -> mask
        # - pos         -> static_metadata
        h = self.encoder(
            x_obs=x, 
            mask=mask_channel, 
            static_metadata=pos, 
            nei_list=nei_list
        )
        
        return h

    def forward(self, x, pos, edge_index, mask_channel, batch=None):
        h = self.encode(x, pos, edge_index, mask_channel)
        return self.decoder(h, mask_channel)

    def _edge_index_to_nei_list(self, edge_index, num_nodes):
        """
        Helper method to bridge PyG's edge_index format to STAR's expected nei_list format.
        (Adjust this based on how STAR's TransformerModel expects nei_list to be structured).
        """
        # If your STAR attention block expects a dense adjacency or padded neighbor tensor,
        # you can build it here from edge_index, or return edge_index directly if adapted.
        device = edge_index.device
        
        # Simple fallback example (create a dense adjacency matrix from edge_index):
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            
        return adj


class STAREncoderGNOSeismicGNN(STAREncoderSeismicGNN):
    """STAR encoder paired with a geometry-aware GNO waveform decoder."""

    def forward(self, x, pos, edge_index, mask_channel, batch=None):
        h = self.encode(x, pos, edge_index, mask_channel)
        return self.decoder(h, pos, mask_channel, batch=batch)