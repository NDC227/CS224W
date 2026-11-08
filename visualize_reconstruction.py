import argparse
import torch
import matplotlib.pyplot as plt

# Import all possible model architectures and your normalization utility
from models.baseline_model import BaselineDecoder, BaselineSeismicGNN
from models.gno import GNO_Decoder
from models.star_encoder_only import STAREncoder, STAREncoderSeismicGNN, STAREncoderGNOSeismicGNN
# from utils import normalize_data

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Seismic Waveform Reconstruction Evaluation")
    parser.add_argument("--model_type", type=str, required=True, choices=["baseline", "star_only", "star_gno"],
                        help="Select which model architecture to evaluate.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth weights file.")
    parser.add_argument("--test_data_path", type=str, default="data/test_seismic_graphs_processed.pt", help="Path to test dataset.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the graph sample in the test set.")
    return parser.parse_args()

def visualize_reconstruction():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Evaluating Model: {args.model_type.upper()}")
    
    # 1. Load test dataset and select sample graph
    test_data = torch.load(args.test_data_path, weights_only=False)
    sample_data = test_data[args.sample_idx].to(device)
    print(sample_data)
    
    # 2. Inspect dimensions and dynamically instantiate the chosen model
    time_steps = sample_data.x.size(1)   # e.g., 6000
    in_channels = sample_data.x.size(2)  # e.g., 3
    pos_dim = sample_data.pos.size(-1)
    
    if args.model_type == "baseline":
        model = BaselineSeismicGNN(in_channels=in_channels, time_steps=time_steps, hidden_dim=256).to(device)
    elif args.model_type == "star_only":
        #in_channels, time_steps, hidden_dim, pos_dim
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
        
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    print("Model weights loaded successfully.")
    
    # 3. Extract pre-masked inputs stored in the Data object
    x_masked = sample_data.x
    target_mask = sample_data.target_mask.bool()
    mask_channel = sample_data.mask_channel
    
    masked_indices = torch.where(target_mask)[0]
    if len(masked_indices) == 0:
        print("Warning: No target nodes found via target_mask in this sample.")
        return
        
    target_node_idx = masked_indices[0].item()
    print(f"Visualizing reconstruction for masked target station node index: {target_node_idx}")
    
    # 4. Forward pass: Model automatically isolates target nodes and returns (num_targets, T, C) or (T, C)
    with torch.no_grad():
        graph_batch = torch.zeros(
            sample_data.num_nodes, dtype=torch.long, device=device
        )
        pred = model(
            x_masked,
            sample_data.pos,
            sample_data.edge_index,
            mask_channel,
            batch=graph_batch,
        )
        
    # Handle matching prediction to our specific target_node_idx
    if pred.ndim == 3:
        # Multiple targets: find which position in masked_indices corresponds to target_node_idx
        target_pos = (masked_indices == target_node_idx).nonzero(as_tuple=True)[0].item()
        pred_target = pred[target_pos]  # Shape: (T, C)
    else:
        pred_target = pred              # Shape: (T, C)
        
    pred_waveform = pred_target.cpu().numpy()  # Shape: (T, C)
    
    # 5. Determine ground truth waveform
    if hasattr(sample_data, 'y') and sample_data.y is not None:
        if sample_data.y.shape == sample_data.x.shape:
            # y contains all nodes: shape (N, T, C)
            true_waveform = sample_data.y[target_node_idx].cpu().numpy()
        elif pred.ndim == 3 and sample_data.y.shape == pred.shape:
            # y contains only targets with matching shape
            target_pos = (masked_indices == target_node_idx).nonzero(as_tuple=True)[0].item()
            true_waveform = sample_data.y[target_pos].cpu().numpy()
        else:
            true_waveform = sample_data.y[0].cpu().numpy() if sample_data.y.ndim == 3 else sample_data.y.cpu().numpy()
    else:
        # Fallback if y isn't available
        true_waveform = sample_data.x[target_node_idx].cpu().numpy()
        
    # 6. Plot the 3 components (East, North, Vertical)
    channel_names = ['East Component (E)', 'North Component (N)', 'Vertical Component (Z)']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    for c in range(3):
        axes[c].plot(true_waveform[:, c], label='Ground Truth', color='black', alpha=0.8, linewidth=1.5)
        axes[c].plot(pred_waveform[:, c], label=f'{args.model_type.capitalize()} Reconstructed', color='crimson', linestyle='--', alpha=0.8, linewidth=1.5)
        axes[c].set_ylabel(channel_names[c], fontsize=11)
        axes[c].legend(loc='upper right', frameon=True)
        axes[c].grid(True, linestyle=':', alpha=0.6)
        
    axes[-1].set_xlabel('Time Steps', fontsize=11)
    plt.suptitle(f'[{args.model_type.upper()}] Waveform Reconstruction at Target Node {target_node_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_filename = f'{args.model_type}_reconstruction_result.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved as '{output_filename}'")
    plt.show()


# python eval.py --model_type baseline --model_path trained_models/seismic_baseline_model.pth --sample_idx 0
if __name__ == '__main__':
    visualize_reconstruction()