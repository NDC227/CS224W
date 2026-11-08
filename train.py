import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Import architectures and custom loss function
from models.baseline_model import (
    BaselineEncoder, 
    BaselineDecoder, 
    BaselineSeismicGNN, 
    BaselineSeismicGNNWithAux
)
from models.gno import GNO_Decoder
from models.star_encoder_only import (
    STAREncoder,
    STAREncoderGNOSeismicGNN,
    STAREncoderSeismicGNN,
)
from loss_functions.reconstruction_loss import ReconstructionLoss

def train_epoch(model, loader, optimizer, criterion, device, is_aux_model=False):
    """
    Executes a single training epoch supporting both standard waveform reconstruction
    and multi-task auxiliary supervision.
    """
    model.train()
    total_loss = 0.0

    batch_i = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 1. Extract inputs from the pre-masked PyG batch object
        x_masked = batch.x
        mask_channel = batch.mask_channel
        true_waveforms = batch.y 
        
        # 2. Forward pass
        # If using auxiliary model, `pred` is a dictionary: {"waveform", "p_arrival", "s_arrival", "baz"}
        pred = model(
            x_masked, batch.pos, batch.edge_index, mask_channel, batch=batch.batch
        )

        if is_aux_model:
            waveform_pred = pred["waveform"]
            
            # Primary waveform reconstruction loss
            loss_wave = criterion(waveform_pred, true_waveforms)
            
            # Auxiliary losses (computed over observed or applicable nodes using dataset attributes)
            # Adjust batch attribute names (e.g., batch.p_arrival, batch.obs_mask) to match your dataset schema
            obs_mask = getattr(batch, "obs_mask", torch.ones_like(pred["p_arrival"], dtype=torch.bool))
            
            loss_p = F.mse_loss(pred["p_arrival"][obs_mask], batch.y_p_arrival_norm[obs_mask])
            loss_s = F.mse_loss(pred["s_arrival"][obs_mask], batch.y_s_arrival_norm[obs_mask])
            loss_baz = F.mse_loss(pred["baz"][obs_mask], batch.y_baz_norm[obs_mask])
            
            # Weighted multi-task loss combination
            lambda_p, lambda_s, lambda_baz = 0.5, 0.5, 0.2
            loss = loss_wave + (lambda_p * loss_p) + (lambda_s * loss_s) + (lambda_baz * loss_baz)
        else:
            # Standard single-task reconstruction loss
            loss = criterion(pred, true_waveforms)
        
        # 3. Backpropagation & optimization step
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs

        # batch_i += 1
        # if batch_i % 10 == 0:
        #     print(loss.item())
        
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, is_aux_model=False):
    """Evaluates the model on validation or test sets without gradient updates."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            x_masked = batch.x
            mask_channel = batch.mask_channel
            true_waveforms = batch.y
            
            pred = model(
                x_masked, batch.pos, batch.edge_index, mask_channel, batch=batch.batch
            )
            
            if is_aux_model:
                waveform_pred = pred["waveform"]
                loss_wave = criterion(waveform_pred, true_waveforms)
                
                obs_mask = getattr(batch, "obs_mask", torch.ones_like(pred["p_arrival"], dtype=torch.bool))
                loss_p = F.mse_loss(pred["p_arrival"][obs_mask], batch.y_p_arrival_norm[obs_mask])
                loss_s = F.mse_loss(pred["s_arrival"][obs_mask], batch.y_s_arrival_norm[obs_mask])
                loss_baz = F.mse_loss(pred["baz"][obs_mask], batch.y_baz_norm[obs_mask])
                
                lambda_p, lambda_s, lambda_baz = 0.5, 0.5, 0.2
                loss = loss_wave + (lambda_p * loss_p) + (lambda_s * loss_s) + (lambda_baz * loss_baz)
            else:
                loss = criterion(pred, true_waveforms)
            
            total_loss += loss.item() * batch.num_graphs
            
    return total_loss / len(loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Seismic Wavefield Reconstruction Training")
    parser.add_argument("--model_type",
                        type=str, 
                        choices=["baseline", "baseline_aux", "star_only", "star_gno", "experimental"],
                        default="baseline",
                        help="Select whether to train the baseline, baseline with aux heads, or experimental model.")
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
    pos_dim = sample_data.pos.size(-1) if hasattr(sample_data, "pos") and sample_data.pos is not None else 3
    
    is_aux_model = False
    
    # 2. Dynamically Instantiate Model based on argument
    if args.model_type == "baseline":
        model = BaselineSeismicGNN(in_channels=in_channels, time_steps=time_steps, hidden_dim=256, pos_dim=pos_dim).to(device)
        model_save_path = "trained_models/baseline_model.pth"
    elif args.model_type == "baseline_aux":
        model = BaselineSeismicGNNWithAux(in_channels=in_channels, time_steps=time_steps, hidden_dim=256, pos_dim=pos_dim).to(device)
        model_save_path = "trained_models/baseline_aux_model.pth"
        is_aux_model = True
    elif args.model_type == "star_only":
        encoder = STAREncoder(in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim)
        decoder = BaselineDecoder(in_channels, time_steps, hidden_dim=256)
        model = STAREncoderSeismicGNN(star_encoder=encoder, decoder=decoder).to(device)
        model_save_path = "trained_models/star_only_model.pth"
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
        model_save_path = "trained_models/star_gno_model.pth"
    elif args.model_type == "experimental":
        pass
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ReconstructionLoss(spectral_weight=0.5, envelope_weight=1.0)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_aux_model=is_aux_model)
        val_loss = evaluate(model, val_loader, criterion, device, is_aux_model=is_aux_model)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best checkpoint to {model_save_path}")

# python train.py --model_type baseline --epochs 20
# python train.py --model_type star_only --epochs 20
# python train.py --model_type star_gno --epochs 20
# python train.py --model_type baseline_aux --epochs 20
if __name__ == "__main__":
    main()