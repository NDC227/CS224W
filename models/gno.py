import torch
import torch.nn as nn


class GNO_Decoder(nn.Module):
    """Decode node latents into waveforms with a coordinate-conditioned integral operator."""

    def __init__(
        self,
        in_channels=3,
        time_steps=500,
        hidden_dim=64,
        pos_dim=3,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels

        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.kernel = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, time_steps * in_channels),
        )

    def forward(self, h, pos, mask_channel, batch=None):
        """
        Args:
            h: Node latent features with shape (N, hidden_dim).
            pos: Node coordinates with shape (N, pos_dim).
            mask_channel: (N, 1), with 1 for observed stations and 0 for targets.
            batch: Optional graph index for each node, used by PyG batches.

        Returns:
            Reconstructed waveforms with shape (num_targets, time_steps, in_channels),
            or (time_steps, in_channels) for a single target.
        """
        if h.ndim != 2 or pos.ndim != 2 or h.size(0) != pos.size(0):
            raise ValueError("h and pos must be rank-2 tensors with the same node count.")
        if mask_channel.numel() != h.size(0):
            raise ValueError("mask_channel must contain exactly one value per node.")

        observed = mask_channel.reshape(-1) == 1
        target_indices = torch.where(~observed)[0]
        if target_indices.numel() == 0:
            raise ValueError("No masked target nodes found in mask_channel.")

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        elif batch.ndim != 1 or batch.numel() != h.size(0):
            raise ValueError("batch must contain exactly one graph index per node.")

        source_indices = torch.where(observed)[0]
        graph_outputs = []
        graph_target_indices = []

        for graph_id in torch.unique(batch[target_indices], sorted=True):
            graph_targets = target_indices[batch[target_indices] == graph_id]
            graph_sources = source_indices[batch[source_indices] == graph_id]
            if graph_sources.numel() == 0:
                raise ValueError(
                    f"Graph {graph_id.item()} has target nodes but no observed stations."
                )

            relative_pos = (
                pos[graph_targets, None, :] - pos[graph_sources][None, :, :]
            )
            kernel_values = self.kernel(relative_pos)
            source_values = self.value_projection(h[graph_sources])
            integral = (kernel_values * source_values.unsqueeze(0)).mean(dim=1)

            query_features = h[graph_targets]
            decoded = self.output_projection(
                torch.cat((query_features, integral), dim=-1)
            )
            graph_outputs.append(decoded)
            graph_target_indices.append(graph_targets)

        target_order = torch.argsort(torch.cat(graph_target_indices))
        out = torch.cat(graph_outputs)[target_order]
        out = out.reshape(-1, self.time_steps, self.in_channels)
        if out.size(0) == 1:
            out = out.squeeze(0)
        return out
