import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from models.baseline_model import BaselineSeismicGNN
from loss_functions.reconstruction_loss import ReconstructionLoss
from utils import *

from torch_geometric.loader import DataLoader

def train_epoch(model, loader, optimizer, criterion, device, mask_ratio=0.25):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Apply random masking
        x_masked, target_mask, mask_channel = apply_node_masking(batch, mask_ratio=mask_ratio)
        
        # Forward pass
        pred = model(x_masked, batch.pos, batch.edge_index, mask_channel)
        
        # Compute loss on target nodes
        loss = criterion(pred, batch.x, target_mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
        
    return total_loss / total_graphs

@torch.no_grad()
def evaluate(model, loader, criterion, device, mask_ratio=0.25):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Apply deterministic or random node masking
        x_masked, target_mask, mask_channel = apply_node_masking(batch, mask_ratio=mask_ratio)
        
        pred = model(x_masked, batch.pos, batch.edge_index, mask_channel)
        loss = criterion(pred, batch.x, target_mask)
        
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
        
    return total_loss / total_graphs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load saved dataset splits from .pt files
    train_data = torch.load("data/train_seismic_graphs.pt", weights_only=False)
    val_data = torch.load("data/val_seismic_graphs.pt", weights_only=False)
    test_data = torch.load("data/test_seismic_graphs.pt", weights_only=False)

    train_data = [normalize_data(d) for d in train_data]
    val_data = [normalize_data(d) for d in val_data]
    test_data = [normalize_data(d) for d in test_data]
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Inspect dataset dimensions from first object
    sample_data = train_data[0]
    time_steps = sample_data.x.size(1)   # e.g., T time steps
    in_channels = sample_data.x.size(2)  # e.g., C=3 channels (E, N, Z)
    
    # 2. Instantiate Model, Optimizer, and Loss
    model = BaselineSeismicGNN(
        in_channels=in_channels,
        time_steps=time_steps,
        hidden_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = ReconstructionLoss(spectral_weight=0.1)
    
    # 3. Training Loop
    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, mask_ratio=0.25)
        val_loss = evaluate(model, val_loader, criterion, device, mask_ratio=0.25)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
    # 4. Final Evaluation on Test Set
    test_loss = evaluate(model, test_loader, criterion, device, mask_ratio=0.25)
    print(f"\nFinal Test Masked Reconstruction Loss: {test_loss:.6f}")

    # 5. Save the trained model
    torch.save(model.state_dict(), "trained_models/seismic_baseline_model.pth")
    print("Model saved to trained_models/seismic_baseline_model.pth")

if __name__ == "__main__":
    main()