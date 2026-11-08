import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseAlignedCrossCorrelationMetric(nn.Module):
    """
    Computes the Maximum Cross-Correlation (CC) between predicted and true waveforms,
    accounting for minor time-shifts/jitter. Values closer to 1.0 indicate better phase alignment.
    """
    def __init__(self, max_lag_samples: int = 20):
        super().__init__()
        self.max_lag_samples = max_lag_samples

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveforms of shape (T, C) or (N, T, C)
            target: Ground truth waveforms of shape (T, C) or (N, T, C)
        Returns:
            Mean cross-correlation score across channels and batches.
        """
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
            
        batch_size, T, C = pred.shape
        cc_scores = []

        for b in range(batch_size):
            for c in range(C):
                p = pred[b, :, c]
                t = target[b, :, c]
                
                # Normalize to zero mean and unit variance to compute normalized cross-correlation
                p = p - p.mean()
                t = t - t.mean()
                
                p_norm = torch.norm(p)
                t_norm = torch.norm(t)
                
                if p_norm < 1e-8 or t_norm < 1e-8:
                    cc_scores.append(torch.tensor(0.0, device=pred.device))
                    continue
                
                p_unit = p / p_norm
                t_unit = t / t_norm
                
                # Compute full cross-correlation using 1D convolution
                # Reshape for conv1d: input (1, 1, T), weight (1, 1, kernel_size)
                p_padded = F.pad(p_unit.unsqueeze(0).unsqueeze(0), (self.max_lag_samples, self.max_lag_samples))
                xcorr = F.conv1d(p_padded, t_unit.unsqueeze(0).unsqueeze(0))
                
                # Max correlation within the allowable lag window
                max_cc = torch.max(torch.abs(xcorr))
                cc_scores.append(max_cc)
                
        return torch.stack(cc_scores).mean()