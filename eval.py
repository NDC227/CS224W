import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your model architectures
from models.baseline_model import BaselineDecoder, BaselineSeismicGNN
from models.star_encoder_only import STAREncoder, STAREncoderSeismicGNN, STAREncoderGNOSeismicGNN
from models.gno import GNO_Decoder

# ==========================================
# 1. Evaluation Metric Classes
# ==========================================

from loss_functions.phase_arrival_time_error import PhaseArrivalTimeErrorMetric
from loss_functions.phase_correlation import PhaseAlignedCrossCorrelationMetric
from loss_functions.peak_amplitude_error import PeakAmplitudeErrorMetric

# ==========================================
# 2. Evaluation Runner Script
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Seismic GNN Metrics on Test Set")
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "star_only", "star_gno"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model weights.")
    parser.add_argument("--test_data_path", type=str, default="data/test_seismic_graphs_processed.pt", help="Path to test dataset.")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="Ratio of nodes to mask for evaluation.")
    parser.add_argument("--sampling_rate", type=float, default=100.0, help="Sampling frequency in Hz (default: 100 Hz).")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Evaluating Model: {args.model_type.upper()}")

    # Load test dataset
    test_data_list = torch.load(args.test_data_path, weights_only=False)
    print(f"Loaded {len(test_data_list)} test graphs from {args.test_data_path}")

    # Initialize model based on first sample dimensions
    sample_0 = test_data_list[0]
    time_steps = sample_0.x.size(1)
    in_channels = sample_0.x.size(2)
    pos_dim = sample_0.pos.size(-1)

    if args.model_type == "baseline":
        model = BaselineSeismicGNN(in_channels=in_channels, time_steps=time_steps, hidden_dim=256).to(device)
    elif args.model_type == "star_only":
        encoder = STAREncoder(in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim)
        decoder = BaselineDecoder(in_channels, time_steps, hidden_dim=256)
        model = STAREncoderSeismicGNN(star_encoder=encoder, decoder=decoder).to(device)
    elif args.model_type == "star_gno":
        encoder = STAREncoder(
            in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim
        )
        decoder = GNO_Decoder(
            in_channels=in_channels,
            time_steps=time_steps,
            hidden_dim=256,
            pos_dim=pos_dim,
        )
        model = STAREncoderGNOSeismicGNN(
            star_encoder=encoder, decoder=decoder
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    print("Model weights loaded successfully.\n")

    # Initialize metrics
    cc_metric = PhaseAlignedCrossCorrelationMetric(max_lag_samples=20).to(device)
    pae_metric = PeakAmplitudeErrorMetric(relative=True).to(device)
    arrival_metric = PhaseArrivalTimeErrorMetric(sampling_rate=args.sampling_rate).to(device)

    # Accumulators for overall test statistics
    all_cc, all_pae, all_arrival_err = [], [], []

    with torch.no_grad():
        for idx, data in enumerate(test_data_list):
            data = data.to(device)

            # 1. Extract pre-masked attributes directly from the PyG Data object
            x_masked = data.x                   # Input waveforms with target nodes zeroed out
            target_mask = data.target_mask.bool() # Boolean tensor indicating target nodes
            mask_channel = data.mask_channel    # Binary indicator feature channel (N, 1)

            # 2. Check if any nodes were actually masked
            if not target_mask.any():
                continue

            # Extract target node indices from the boolean mask
            target_indices = torch.where(target_mask)[0]

            # 3. Forward pass: Handle both standard models (tensor output) and models with auxiliary heads (dict output)
            outputs = model(x_masked, data.pos, data.edge_index, mask_channel)

            if isinstance(outputs, dict):
                pred_targets = outputs["waveform"]
            else:
                pred_targets = outputs

            # 4. Ground truth target waveforms from the pre-processed data
            true_targets = data.y

            # Ensure consistent batched/multi-target dimensions
            if pred_targets.ndim == 2:
                pred_targets = pred_targets.unsqueeze(0)
            if true_targets.ndim == 2:
                true_targets = true_targets.unsqueeze(0)

            # 5. Compute Waveform Metrics
            cc_score = cc_metric(pred_targets, true_targets)
            pae_score = pae_metric(pred_targets, true_targets)

            all_cc.append(cc_score.item())
            all_pae.append(pae_score.item())

            # 6. Optional: Evaluate Auxiliary Arrival Time Errors (if model outputs a dict and ground truth exists)
            if isinstance(outputs, dict) and hasattr(data, 'y_p_arrival') and data.y_p_arrival is not None:
                # Extract predictions for the target nodes
                pred_p_norm = outputs['p_arrival'][target_indices]
                
                # Convert normalized prediction back to sample indices (since model outputs [0, 1])
                T_samples = data.x.size(1)
                pred_p_samples = pred_p_norm * T_samples
                true_p_samples = data.y_p_arrival[target_indices].float()
                
                # Compute arrival time error in seconds using your metric
                arr_err = arrival_metric(pred_p_samples, true_p_samples)
                all_arrival_err.append(arr_err.item())

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_data_list):
                mean_cc = torch.tensor(all_cc).mean().item()
                print(f"Processed {idx + 1}/{len(test_data_list)} graphs... | Current Mean CC: {mean_cc:.4f}")

    # Print Final Aggregated Results
    print("\n" + "="*40)
    print(f"FINAL TEST SET EVALUATION ({args.model_type.upper()})")
    print("="*40)
    print(f"Phase-Aligned Cross-Correlation (CC) [Higher is better]: {torch.tensor(all_cc).mean():.4f} ± {torch.tensor(all_cc).std():.4f}")
    print(f"Relative Peak Amplitude Error (PAE) [Lower is better]:  {torch.tensor(all_pae).mean():.4f} ± {torch.tensor(all_pae).std():.4f}")
    if all_arrival_err:
        print(f"Phase Arrival Time Error (Seconds) [Lower is better]: {torch.tensor(all_arrival_err).mean():.4f}s")
    print("="*40)

# python eval_metrics.py --model_type baseline --model_path trained_models/seismic_baseline_model.pth --test_data_path data/test_seismic_graphs_processed.pt
# python eval_metrics.py --model_type star_only --model_path trained_models/star_only_model.pth --test_data_path data/test_seismic_graphs_processed.pt
if __name__ == '__main__':
    evaluate()