import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseArrivalTimeErrorMetric(nn.Module):
    """
    Computes the absolute time error (in seconds) between predicted phase arrival 
    sample indices and ground truth pick sample indices.
    """
    def __init__(self, sampling_rate: float = 100.0):
        super().__init__()
        self.sampling_rate = sampling_rate

    def forward(self, pred_arrival: torch.Tensor, target_arrival: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_arrival: Predicted sample indices or normalized times (if normalized, set scale accordingly)
            target_arrival: Ground truth sample indices
        Returns:
            Mean absolute error in seconds.
        """
        # Calculate sample difference
        sample_diff = torch.abs(pred_arrival - target_arrival)
        
        # Convert samples to seconds based on sampling rate (e.g., 100 Hz for STEAD)
        time_error_seconds = sample_diff / self.sampling_rate
        
        return time_error_seconds.mean()