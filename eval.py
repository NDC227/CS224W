import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from train import BaselineSeismicGNN, apply_node_masking
from utils import normalize_data  # Import normalization function


def visualize_reconstruction(
    model_path, test_data_path, sample_idx=0, mask_ratio=0.25
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load test dataset and select a sample graph
    test_data = torch.load(test_data_path, weights_only=False)
    if isinstance(test_data, list):
        sample_data = test_data[sample_idx]
    else:
        sample_data = test_data[sample_idx]

    sample_data = sample_data.to(device)

    # 2. Apply Z-score normalization prior to masking and model forward pass
    sample_data = normalize_data(sample_data.clone())

    # 3. Inspect dimensions and initialize model
    time_steps = sample_data.x.size(1)
    in_channels = sample_data.x.size(2)

    model = BaselineSeismicGNN(
        in_channels=in_channels, time_steps=time_steps, hidden_dim=64
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")

    # 4. Apply node masking on normalized input
    with torch.no_grad():
        x_masked, target_mask, mask_channel = apply_node_masking(
            sample_data, mask_ratio=mask_ratio
        )

        # Forward pass with normalized x and pos
        pred = model(
            x_masked, sample_data.pos, sample_data.edge_index, mask_channel
        )

    # 5. Extract a masked target node to visualize
    masked_indices = torch.where(target_mask)[0]
    if len(masked_indices) == 0:
        print(
            "Warning: No nodes were masked with the current random seed. Try"
            " running again."
        )
        return

    target_node_idx = masked_indices[0].item()
    print(
        "Visualizing reconstruction for masked target node index:"
        f" {target_node_idx}"
    )

    # Move normalized waveforms to CPU and convert to NumPy for plotting
    true_waveform = sample_data.x[target_node_idx].cpu().numpy()  # Shape: (T, C)
    pred_waveform = pred[target_node_idx].cpu().numpy()  # Shape: (T, C)

    # 6. Plot the 3 components (East, North, Vertical)
    channel_names = [
        "East Component (E)",
        "North Component (N)",
        "Vertical Component (Z)",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for c in range(3):
        axes[c].plot(
            true_waveform[:, c],
            label="Ground Truth (Normalized)",
            color="black",
            alpha=0.8,
            linewidth=1.5,
        )
        axes[c].plot(
            pred_waveform[:, c],
            label="GNN Reconstructed",
            color="crimson",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
        )
        axes[c].set_ylabel(channel_names[c], fontsize=11)
        axes[c].legend(loc="upper right", frameon=True)
        axes[c].grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Time Steps", fontsize=11)
    plt.suptitle(
        "Seismic Waveform Reconstruction at Unobserved Station (Node"
        f" {target_node_idx})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save figure and show plot
    plt.savefig("waveform_reconstruction_result.png", dpi=300)
    print("Plot saved as 'waveform_reconstruction_result.png'")
    plt.show()


if __name__ == "__main__":
    visualize_reconstruction(
        model_path="trained_models/seismic_baseline_model.pth",
        test_data_path="data/test_seismic_graphs.pt",
        sample_idx=0,
        mask_ratio=0.25,
    )