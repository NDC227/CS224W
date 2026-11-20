import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# Import architectures and custom loss function
from models.baseline_model import (
    BaselineEncoder, 
    BaselineLinearDecoder, 
    BaselineConvDecoder,
    BaselineSeismicGNN, 
    BaselineSeismicGNNWithAux
)
from models.gno import GNO_Decoder, GNOSeismicGNN
from models.star_encoder_only import (
    STAREncoder,
    STAREncoderGNOSeismicGNN,
    STAREncoderSeismicGNN,
)
from loss_functions.reconstruction_loss import ReconstructionLoss

def _get_target_indices(mask_channel):
    mask_1d = mask_channel.squeeze(-1)
    target_idx = torch.where(mask_1d == 0.0)[0]
    if target_idx.numel() == 0:
        target_idx = torch.where(mask_1d != 1.0)[0]
    return target_idx


def _align_target_to_pred(target, pred, batch):
    if target.shape == pred.shape:
        return target
    if target.numel() == pred.numel():
        return target.reshape_as(pred)
    if target.size(0) == batch.x.size(0):
        target_idx = _get_target_indices(batch.mask_channel)
        sliced = target[target_idx]
        if sliced.numel() == pred.numel():
            return sliced.reshape_as(pred)
        return sliced
    return target


def _compute_batch_loss(pred, batch, criterion, is_aux_model=False):
    true_waveforms = batch.y

    if is_aux_model:
        waveform_pred = pred["waveform"]
        aligned_waveforms = _align_target_to_pred(true_waveforms, waveform_pred, batch)
        assert waveform_pred.shape == aligned_waveforms.shape, (
            f"Shape mismatch: pred {waveform_pred.shape} vs target {aligned_waveforms.shape}"
        )
        loss_wave = criterion(waveform_pred, aligned_waveforms)

        obs_mask = getattr(
            batch, "obs_mask", torch.ones_like(pred["p_arrival"], dtype=torch.bool)
        )
        if obs_mask.shape != pred["p_arrival"].shape and obs_mask.numel() == pred["p_arrival"].numel():
            obs_mask = obs_mask.reshape_as(pred["p_arrival"])

        p_target = batch.y_p_arrival_norm.reshape_as(pred["p_arrival"])
        s_target = batch.y_s_arrival_norm.reshape_as(pred["s_arrival"])
        baz_target = batch.y_baz_norm.reshape_as(pred["baz"])

        loss_p = F.mse_loss(pred["p_arrival"][obs_mask], p_target[obs_mask])
        loss_s = F.mse_loss(pred["s_arrival"][obs_mask], s_target[obs_mask])
        loss_baz = F.mse_loss(pred["baz"][obs_mask], baz_target[obs_mask])

        lambda_p, lambda_s, lambda_baz = 0.5, 0.5, 0.2
        return (
            loss_wave
            + (lambda_p * loss_p)
            + (lambda_s * loss_s)
            + (lambda_baz * loss_baz)
        )

    aligned_waveforms = _align_target_to_pred(true_waveforms, pred, batch)
    assert pred.shape == aligned_waveforms.shape, (
        f"Shape mismatch: pred {pred.shape} vs target {aligned_waveforms.shape}"
    )
    return criterion(pred, aligned_waveforms)


def train_epoch(model, loader, optimizer, criterion, device, is_aux_model=False):
    """
    Executes a single training epoch supporting both standard waveform reconstruction
    and multi-task auxiliary supervision.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(
            batch.x, batch.pos, batch.edge_index, batch.mask_channel, batch=batch.batch
        )
        loss = _compute_batch_loss(pred, batch, criterion, is_aux_model=is_aux_model)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, is_aux_model=False):
    """Evaluates the model on validation or test sets without gradient updates."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(
                batch.x, batch.pos, batch.edge_index, batch.mask_channel, batch=batch.batch
            )
            loss = _compute_batch_loss(pred, batch, criterion, is_aux_model=is_aux_model)

            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Seismic Wavefield Reconstruction Training")
    parser.add_argument("--model_type",
                        type=str, 
                        choices=["baseline", "baseline_aux", "star_only", "gno_only", "star_gno", "experimental"],
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
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0)
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
        decoder = BaselineConvDecoder(in_channels, time_steps, hidden_dim=256)
        model = STAREncoderSeismicGNN(star_encoder=encoder, decoder=decoder).to(device)
        model_save_path = "trained_models/star_only_model.pth"
    elif args.model_type == "gno_only":
        encoder = BaselineEncoder(in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim)
        decoder = GNO_Decoder(
                    in_channels=in_channels,
                    time_steps=time_steps,
                    hidden_dim=256,
                    pos_dim=pos_dim,
                )
        model = GNOSeismicGNN(encoder, decoder)
        model_save_path = "trained_models/gno_only_model.pth"
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
        
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ReconstructionLoss(time_weight=1.0,
                                   spectral_weight=0.2,
                                   envelope_weight=1.0,
                                   n_fft=256,
                                   hop_length=64)
    
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

# python train.py --model_type baseline --epochs 20 --lr 1e-3
# python train.py --model_type star_only --epochs 20 --lr 1e-3 --weight_decay 0.0 --batch_size 2
# python train.py --model_type gno_only --epochs 20
# python train.py --model_type star_gno --epochs 20
# python train.py --model_type baseline_aux --epochs 20
if __name__ == "__main__":
    main()