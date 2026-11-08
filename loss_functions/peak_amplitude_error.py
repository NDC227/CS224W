import torch
import torch.nn as nn
import torch.nn.functional as F

class PeakAmplitudeErrorMetric(nn.Module):
    """
    Computes the Peak Amplitude Error (PAE), measuring how accurately 
    the model predicts the maximum shaking intensity (absolute peak ground motion).
    """
    def __init__(self, relative: bool = True):
        super().__init__()
        self.relative = relative

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveforms of shape (T, C) or (N, T, C)
            target: Ground truth waveforms of shape (T, C) or (N, T, C)
        Returns:
            Absolute or relative peak amplitude error.
        """
        # Find peak (maximum absolute value) along the time dimension (dim=-2)
        pred_peaks = torch.max(torch.abs(pred), dim=-2).values  # Shape: (C,) or (N, C)
        target_peaks = torch.max(torch.abs(target), dim=-2).values
        
        if self.relative:
            # Relative percentage error, protected against division by zero
            error = torch.abs(pred_peaks - target_peaks) / (torch.abs(target_peaks) + 1e-6)
        else:
            # Absolute error in amplitude units
            error = torch.abs(pred_peaks - target_peaks)
            
        return error.mean()