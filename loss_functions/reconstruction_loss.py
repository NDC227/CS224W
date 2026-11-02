import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, spectral_weight=0.5, envelope_weight=0.2, n_fft=64, hop_length=16):
        super(ReconstructionLoss, self).__init__()
        self.spectral_weight = spectral_weight
        self.envelope_weight = envelope_weight
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted target waveforms of shape (num_targets, T, C) or (T, C)
            target: Ground truth target waveforms of shape (num_targets, T, C) or (T, C)
        """
        # Ensure consistent shapes (handle single-target / unbatched or batch dimensions cleanly)
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
            
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 1. Time-Domain Loss (Smooth L1 is robust to minor phase shifts)
        time_loss = F.smooth_l1_loss(pred, target)
        
        # Flatten batch and channel dimensions for STFT: (num_targets * C, T)
        pred_flat = pred.transpose(1, 2).reshape(-1, pred.size(1))
        true_flat = target.transpose(1, 2).reshape(-1, target.size(1))
        
        # 2. Log-STFT Magnitude Loss (better balances low and high frequencies)
        stft_pred = torch.stft(pred_flat, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        stft_true = torch.stft(true_flat, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        
        mag_pred = torch.abs(stft_pred)
        mag_true = torch.abs(stft_true)
        
        # Log compression prevents high-amplitude frequencies from dominating the loss
        log_mag_pred = torch.log(mag_pred + 1e-5)
        log_mag_true = torch.log(mag_true + 1e-5)
        spectral_loss = F.l1_loss(log_mag_pred, log_mag_true)
        
        total_loss = time_loss + self.spectral_weight * spectral_loss
        
        # 3. Optional Amplitude Envelope Loss (combats energy collapse/attenuation)
        if self.envelope_weight > 0.0:
            kernel_size = 33
            pad = kernel_size // 2
            
            pred_env = F.avg_pool1d(torch.abs(pred_flat).unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)
            true_env = F.avg_pool1d(torch.abs(true_flat).unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)
            
            env_loss = F.l1_loss(pred_env, true_env)
            total_loss += self.envelope_weight * env_loss
            
        return total_loss