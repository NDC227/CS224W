import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, spectral_weight=0.5):
        super(ReconstructionLoss, self).__init__()
        self.spectral_weight = spectral_weight

    def forward(self, pred, target, mask):
        pred_target = pred[mask]
        true_target = target[mask]
        
        # Smooth L1 is less sensitive to phase outliers than MSE
        time_loss = F.smooth_l1_loss(pred_target, true_target)
        
        # STFT Loss
        pred_flat = pred_target.transpose(1, 2).reshape(-1, pred_target.size(1))
        true_flat = true_target.transpose(1, 2).reshape(-1, true_target.size(1))
        
        stft_pred = torch.stft(pred_flat, n_fft=64, hop_length=16, return_complex=True)
        stft_true = torch.stft(true_flat, n_fft=64, hop_length=16, return_complex=True)
        
        spectral_loss = F.l1_loss(torch.abs(stft_pred), torch.abs(stft_true))
        
        return time_loss + self.spectral_weight * spectral_loss