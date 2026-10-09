import torch

def normalize_data(data, eps=1e-8):
    """
    Normalizes waveform signals and spatial coordinates in a PyG Data object.
    """
    # 1. Normalize waveforms per station across time (mean=0, std=1)
    mean_x = data.x.mean(dim=1, keepdim=True)  # Shape: (N, 1, C)
    std_x = data.x.std(dim=1, keepdim=True)    # Shape: (N, 1, C)
    data.x = (data.x - mean_x) / (std_x + eps)
    
    # 2. Normalize spatial position coordinates to zero mean and unit variance
    mean_pos = data.pos.mean(dim=0, keepdim=True)  # Shape: (1, 3)
    std_pos = data.pos.std(dim=0, keepdim=True)    # Shape: (1, 3)
    data.pos = (data.pos - mean_pos) / (std_pos + eps)
    
    return data

def apply_node_masking(data, mask_ratio=0.25):
    """
    Randomly selects target nodes to mask in a PyG Data / Batch object.
    
    Args:
        data: PyG Data or Batch object with data.x shape (N, T, C) and data.pos (N, 3)
        mask_ratio: Fraction of nodes to mask as targets (default: 0.25)
        
    Returns:
        x_masked: Input waveform tensor with masked nodes zeroed out (N, T, C)
        target_mask: Boolean tensor (N,) where True indicates a masked target station
        mask_channel: Binary indicator feature channel (N, 1) appended to inputs
    """
    num_nodes = data.x.size(0)
    num_masked = max(1, int(num_nodes * mask_ratio))
    
    # Randomly select node indices to mask
    perm = torch.randperm(num_nodes, device=data.x.device)
    masked_indices = perm[:num_masked]
    
    target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    target_mask[masked_indices] = True
    
    # Zero out waveform signals at masked target nodes
    x_masked = data.x.clone()
    x_masked[target_mask] = 0.0
    
    # Binary mask channel indicating observed (1.0) vs target (0.0)
    mask_channel = (~target_mask).float().unsqueeze(-1)  # Shape: (N, 1)
    
    return x_masked, target_mask, mask_channel