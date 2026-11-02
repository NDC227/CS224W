import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.baseline_model import BaselineSeismicGNN
from loss_functions.reconstruction_loss import ReconstructionLoss

def train_epoch(model, loader, optimizer, criterion, device):
    """Executes a single training epoch using pre-masked data attributes."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 1. Extract pre-existing attributes from the batch object
        # batch.x: Input waveforms (with masked nodes already zeroed out)
        # batch.target_mask: Boolean tensor indicating target nodes to reconstruct
        # batch.mask_channel: Binary indicator feature channel (N, 1)
        x_masked = batch.x
        target_mask = batch.target_mask.bool()
        mask_channel = batch.mask_channel
        
        # 2. Forward pass through BaselineSeismicGNN
        pred = model(x_masked, batch.pos, batch.edge_index, mask_channel)
        
        # 3. Compute loss against the ground truth
        # Note: Ensure batch.y (or your ground truth tensor) holds the unmasked target signals 
        # for the loss calculation. If batch.y contains the full unmasked waveforms:
        true_waveforms = batch.y if hasattr(batch, 'y') and batch.y.shape == pred.shape else batch.x 
        # (Alternatively, if batch.x was modified in-place, make sure you have the true signals accessible)
        
        loss = criterion(pred, true_waveforms, target_mask)
        
        # 4. Backpropagation & optimization step
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluates the model on validation or test sets without gradient updates."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            x_masked = batch.x
            target_mask = batch.target_mask.bool()
            mask_channel = batch.mask_channel
            
            pred = model(x_masked, batch.pos, batch.edge_index, mask_channel)
            
            true_waveforms = batch.y if hasattr(batch, 'y') and batch.y.shape == pred.shape else batch.x
            loss = criterion(pred, true_waveforms, target_mask)
            
            total_loss += loss.item() * batch.num_graphs
            
    return total_loss / len(loader.dataset)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load your dataset splits saved as .pt files
    # (Ensure these match your saved data structure, e.g., a list of PyG Data objects)
    train_data = torch.load("data/train_seismic_graphs_processed.pt", weights_only=False)
    val_data = torch.load("data/val_seismic_graphs_processed.pt", weights_only=False)
    test_data = torch.load("data/test_seismic_graphs_processed.pt", weights_only=False)
    
    # Create PyG DataLoaders (batch_size governs how many graph samples are processed in parallel)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # 2. Inspect dimensions from a sample graph
    sample_data = train_data[0]
    time_steps = sample_data.x.size(1)   # e.g., T time steps
    in_channels = sample_data.x.size(2)  # e.g., C = 3 channels (E, N, Z)
    
    # 3. Initialize Model, Optimizer, and Loss Function
    model = BaselineSeismicGNN(
        in_channels=in_channels,
        time_steps=time_steps,
        hidden_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = ReconstructionLoss(spectral_weight=0.5)
    
    # 4. Training Loop Execution
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, mask_ratio=0.25)
        val_loss = evaluate(model, val_loader, criterion, device, mask_ratio=0.25)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save the best baseline model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "trained_models/seismic_baseline_model.pth")
            print("  --> Saved new best baseline checkpoint.")
            
    # 5. Final Evaluation on Test Set using Best Weights
    model.load_state_dict(torch.load("trained_models/seismic_baseline_model.pth", map_location=device))
    test_loss = evaluate(model, test_loader, criterion, device, mask_ratio=0.25)
    print(f"\nFinal Test Masked Reconstruction Loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()