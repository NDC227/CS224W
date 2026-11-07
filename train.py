import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Import both architectures and your custom loss function
from models.baseline_model import BaselineEncoder, BaselineDecoder, BaselineSeismicGNN
from models.star_encoder_only import STAREncoder, STAREncoderSeismicGNN
from loss_functions.reconstruction_loss import ReconstructionLoss

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes a single training epoch using pre-masked data attributes
    where the model output shape matches the ground truth targets (batch.y).
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        # print("batch", batch)
        
        # 1. Extract inputs from the pre-masked PyG batch object
        # batch.x: Input waveforms (with target nodes already zeroed out) -> Shape: (sum(N), T, 3)
        # batch.mask_channel: Binary indicator feature channel -> Shape: (sum(N), 1)
        x_masked = batch.x
        mask_channel = batch.mask_channel
        
        # Ground truth waveforms for the masked targets -> Shape: (num_targets_in_batch, T, 3)
        true_waveforms = batch.y 
        
        # 2. Forward pass: Model automatically isolates target nodes and returns (num_targets, T, C)
        pred = model(x_masked, batch.pos, batch.edge_index, mask_channel)

        # print(pred.shape, true_waveforms.shape)
        # 3. Compute loss directly between predictions and ground truth targets
        # (Note: Since `pred` and `true_waveforms` are already aligned to the target nodes, 
        # you no longer need a boolean mask index in your criterion call!)
        loss = criterion(pred, true_waveforms)
        
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
            mask_channel = batch.mask_channel
            true_waveforms = batch.y
            
            pred = model(x_masked, batch.pos, batch.edge_index, mask_channel)
            loss = criterion(pred, true_waveforms)
            
            total_loss += loss.item() * batch.num_graphs
            
    return total_loss / len(loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Seismic Wavefield Reconstruction Training")
    parser.add_argument("--model_type",
                        type=str, 
                        choices=["baseline", "star_only", "experimental"], 
                        default="baseline",
                        help="Select whether to train the baseline or experimental model.")
    parser.add_argument("--epochs", 
                        type=int, 
                        default=20)
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1)
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Training Model: {args.model_type.upper()}")
    
    # 1. Load Data
    train_data = torch.load("data/train_seismic_graphs_processed.pt", weights_only=False)
    val_data = torch.load("data/val_seismic_graphs_processed.pt", weights_only=False)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    sample_data = train_data[0]
    time_steps = sample_data.x.size(1)
    in_channels = sample_data.x.size(2)
    
    # 2. Dynamically Instantiate Model based on argument
    if args.model_type == "baseline":
        model = BaselineSeismicGNN(in_channels=in_channels, time_steps=time_steps, hidden_dim=256).to(device)
        model_save_path = "trained_models/baseline_model.pth"
    if args.model_type == "star_only":
        #in_channels, time_steps, hidden_dim, pos_dim
        encoder = STAREncoder(in_channels, time_steps, hidden_dim=256, pos_dim=3)
        decoder = BaselineDecoder(in_channels, time_steps, hidden_dim=256)
        model = STAREncoderSeismicGNN(star_encoder=encoder, decoder=decoder).to(device)
        model_save_path = "trained_models/star_only_model.pth"
    elif args.model_type == "experimental":
        pass
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ReconstructionLoss(spectral_weight=0.5, envelope_weight=1.0)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best checkpoint to {model_save_path}")

# python train.py --model_type baseline --epochs 20
# python train.py --model_type star_only --epochs 20
if __name__ == "__main__":
    main()