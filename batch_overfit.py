import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam

# Import architectures and custom loss function
from models.baseline_model import (
    BaselineEncoder,
    BaselineDecoder,
    BaselineConvDecoder,
    BaselineSeismicGNN,
    BaselineSeismicGNNWithAux,
)
from models.gno import GNO_Decoder, GNOSeismicGNN
from models.star_encoder_only import (
    STAREncoder,
    STAREncoderGNOSeismicGNN,
    STAREncoderSeismicGNN,
)
from loss_functions.reconstruction_loss import ReconstructionLoss


def compute_loss(pred, batch, criterion, is_aux_model=False):
    """Computes reconstruction (and optional auxiliary) loss for a batch."""
    true_waveforms = batch.y
    # print("pred", pred[0,:])
    # print("gt", true_waveforms[0,:])

    if is_aux_model:
        waveform_pred = pred["waveform"]
        loss_wave = criterion(waveform_pred, true_waveforms)

        obs_mask = getattr(
            batch, "obs_mask", torch.ones_like(pred["p_arrival"], dtype=torch.bool)
        )
        loss_p = F.mse_loss(
            pred["p_arrival"][obs_mask], batch.y_p_arrival_norm[obs_mask]
        )
        loss_s = F.mse_loss(
            pred["s_arrival"][obs_mask], batch.y_s_arrival_norm[obs_mask]
        )
        loss_baz = F.mse_loss(
            pred["baz"][obs_mask], batch.y_baz_norm[obs_mask]
        )

        lambda_p, lambda_s, lambda_baz = 0.5, 0.5, 0.2
        return (
            loss_wave
            + (lambda_p * loss_p)
            + (lambda_s * loss_s)
            + (lambda_baz * loss_baz)
        )

    else:
        loss = criterion(pred, true_waveforms)
        return loss


def compute_grad_norm(parameters, norm_type=2.0):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].grad.device
    norms = torch.stack(
        [torch.linalg.vector_norm(p.grad.detach(), ord=norm_type).to(device) for p in params]
    )
    return torch.linalg.vector_norm(norms, ord=norm_type).item()

def overfit_step(model, fixed_batch, optimizer, criterion, is_aux_model=False):
    """Runs one optimization step on the same fixed batch."""
    model.train()
    optimizer.zero_grad()

    pred = model(
        fixed_batch.x,
        fixed_batch.pos,
        fixed_batch.edge_index,
        fixed_batch.mask_channel,
        batch=fixed_batch.batch,
    )

    # print("pred min/max/std", pred.min().item(), pred.max().item(), pred.std().item())

    loss = compute_loss(pred, fixed_batch, criterion, is_aux_model=is_aux_model)
    loss.backward()

    # Optional: clip gradients (use a large max_norm like 10.0+ first if you only want to observe raw norms)
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1).item()

    # Optional: inspect encoder vs. decoder separately
    enc_params = model.star_encoder.parameters() if hasattr(model, "star_encoder") else model.encoder.parameters()
    enc_grad_norm = compute_grad_norm(enc_params)
    dec_grad_norm = compute_grad_norm(model.decoder.parameters())
    # print("grad_norms", enc_grad_norm, dec_grad_norm)

    optimizer.step()

    return loss.item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-Batch Overfit Sanity Check for Seismic Reconstruction"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "baseline",
            "baseline_aux",
            "star_only",
            "gno_only",
            "star_gno",
            "experimental",
        ],
        default="star_only",
        help="Model architecture to overfit on a single batch.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_batch", type=int, default=2, help="Number of fixed batches to overfit on.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_idx", type=int, default=0, help="Index of sample/batch to overfit on.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Default 0.0 for overfitting checks.")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def build_model(args, time_steps, in_channels, pos_dim, device):
    is_aux_model = False

    if args.model_type == "baseline":
        model = BaselineSeismicGNN(
            in_channels=in_channels,
            time_steps=time_steps,
            hidden_dim=256,
            pos_dim=pos_dim,
        ).to(device)
        save_path = "trained_models/overfit_baseline_model.pth"
    elif args.model_type == "baseline_aux":
        model = BaselineSeismicGNNWithAux(
            in_channels=in_channels,
            time_steps=time_steps,
            hidden_dim=256,
            pos_dim=pos_dim,
        ).to(device)
        save_path = "trained_models/overfit_baseline_aux_model.pth"
        is_aux_model = True
    elif args.model_type == "star_only":
        encoder = STAREncoder(in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim)
        decoder = BaselineConvDecoder(in_channels, time_steps, hidden_dim=256)
        model = STAREncoderSeismicGNN(star_encoder=encoder, decoder=decoder).to(device)
        save_path = "trained_models/overfit_star_only_model.pth"
    elif args.model_type == "gno_only":
        encoder = BaselineEncoder(in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim)
        decoder = GNO_Decoder(
            in_channels=in_channels,
            time_steps=time_steps,
            hidden_dim=256,
            pos_dim=pos_dim,
        )
        model = GNOSeismicGNN(encoder, decoder).to(device)
        save_path = "trained_models/overfit_gno_only_model.pth"
    elif args.model_type == "star_gno":
        encoder = STAREncoder(in_channels, time_steps, hidden_dim=256, pos_dim=pos_dim)
        decoder = GNO_Decoder(
            in_channels=in_channels,
            time_steps=time_steps,
            hidden_dim=256,
            pos_dim=pos_dim,
        )
        model = STAREncoderGNOSeismicGNN(star_encoder=encoder, decoder=decoder).to(device)
        save_path = "trained_models/overfit_star_gno_model.pth"
    elif args.model_type == "experimental":
        raise NotImplementedError("Experimental model is not defined yet.")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model, save_path, is_aux_model


def main():
    args = parse_args()
    num_batch = args.num_batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} | Overfitting Model: {args.model_type.upper()} | "
        f"Batches: {num_batch} (batch_size={args.batch_size}, start_idx={args.batch_idx})"
    )

    # 1. Load dataset and pin a small set of fixed batches
    train_data = torch.load("data/train_seismic_graphs_processed.pt", weights_only=False)
    batch_loader = DataLoader(
        train_data[args.batch_idx : args.batch_idx + num_batch * args.batch_size],
        batch_size=args.batch_size,
        shuffle=False,
    )
    fixed_batches = [batch.to(device) for batch in batch_loader]

    sample_data = train_data[1]
    time_steps = sample_data.x.size(1)
    in_channels = sample_data.x.size(2)
    pos_dim = (
        sample_data.pos.size(-1)
        if hasattr(sample_data, "pos") and sample_data.pos is not None
        else 3
    )

    # 2. Instantiate model, optimizer, and criterion
    model, model_save_path, is_aux_model = build_model(
        args, time_steps, in_channels, pos_dim, device
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ReconstructionLoss(
        time_weight=1.0,
        spectral_weight=0.2,
        envelope_weight=1.0,
        n_fft=256,
        hop_length=64,
    )

    # 3. Multi-batch overfitting loop
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in fixed_batches:
            epoch_loss += overfit_step(
                model, batch, optimizer, criterion, is_aux_model=is_aux_model
            )
        avg_loss = epoch_loss / len(fixed_batches)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"{len(fixed_batches)}-Batch Avg Loss: {avg_loss:.6f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(best_state_dict, model_save_path)
    print(f"--> Saved best overfit checkpoint (loss={best_loss:.6f}) to {model_save_path}")

# python batch_overfit.py --model_type baseline --epochs 200 --lr 1e-3
# python batch_overfit.py --model_type star_only --epochs 200 --lr 1e-3 --num_batch 4
if __name__ == "__main__":
    main()