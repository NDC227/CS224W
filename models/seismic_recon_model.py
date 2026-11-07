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

    def forward(self, x, pos, edge_index, mask_channel):
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
        
        # 3. Decode into the target waveform(s)
        out = self.decoder(h, mask_channel)
        return out

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