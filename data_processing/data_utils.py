import numpy as np
import torch
from torch_geometric.data import Data
import obspy
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import networkx as nx

def geodetic_to_ecef(lat, lon, elevation_m=0.0):
    """
    Converts Latitude, Longitude, Elevation to 3D ECEF Cartesian coordinates (in km).
    """
    a = 6378137.0  # WGS-84 Earth semi-major axis (meters)
    e_sq = 0.00669437999014  # WGS-84 eccentricity squared
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
    
    x = (N + elevation_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + elevation_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e_sq) + elevation_m) * np.sin(lat_rad)
    
    return np.array([x, y, z], dtype=np.float32) / 1000.0  # Shape: (3,)

def process_stead_trace(hdf5_file, trace_name):
    dataset = hdf5_file.get(f"data/{trace_name}")
    raw_data = np.array(dataset, dtype=np.float32) # (6000, 3) -> E, N, Z
    
    stream = obspy.Stream()
    for ch_idx, comp in enumerate(['E', 'N', 'Z']):
        tr = obspy.Trace(data=raw_data[:, ch_idx])
        tr.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
        tr.stats.delta = 0.01  # 100 Hz sampling rate
        tr.stats.channel = dataset.attrs['receiver_type'] + comp
        stream.append(tr)
        
    # Detrend and bandpass filter
    stream.detrend("demean")
    stream.detrend("linear")
    stream.filter("bandpass", freqmin=0.5, freqmax=20.0)
    
    return np.stack([tr.data for tr in stream], axis=-1)  # (6000, 3)

def align_processed_waves(row):
    processed_waves = row["processed_waves"]
    relative_times = row["relative_time"]

    # Align processed_waves based on relative_times
    sample_rate_hz = 100.0
    ref_time = min(relative_times)

    # Shift each waveform by its relative delay in samples.
    # The earliest event (relative_time == 0) stays at the start of the aligned array.
    max_shift = int(round((max(relative_times) - ref_time).total_seconds() * sample_rate_hz))
    aligned_waves = []

    for wave, rel_t in zip(processed_waves, relative_times):
        shift = int(round((rel_t - ref_time).total_seconds() * sample_rate_hz))
        aligned_wave = np.zeros((wave.shape[0] + max_shift, wave.shape[1]), dtype=wave.dtype)
        aligned_wave[shift:shift + wave.shape[0]] = wave
        aligned_waves.append(aligned_wave)

    return aligned_waves

def align_arrival_times(row, sample_rate_hz=100.0):
    # Convert inputs to numpy arrays to handle scalars, lists, or Series
    p_arrival = np.array(row["p_arrival_sample"], dtype=float)
    s_arrival = np.array(row["s_arrival_sample"], dtype=float)
    relative_time = np.array([time.total_seconds() for time in row["relative_time"]], dtype=float)

    # 1. Convert relative time offset (seconds) into integer sample count shift
    sample_shift = np.round(relative_time * sample_rate_hz)

    # 2. Shift arrival sample indices to the aligned global t_0 timeline
    p_aligned = p_arrival + sample_shift
    s_aligned = s_arrival + sample_shift

    # 3. Preserve NaNs for missing phase picks, otherwise cast to int
    row["p_arrival_aligned"] = np.where(
        np.isnan(p_arrival), np.nan, np.round(p_aligned)
    )
    row["s_arrival_aligned"] = np.where(
        np.isnan(s_arrival), np.nan, np.round(s_aligned)
    )

    return row

def build_event_edges(row, k_neighbors=6, max_radius_km=200.0, v_phase=6.0):
    # Ensure coords is a 2D NumPy array with shape (N, 3)
    coords = np.asarray(row["station_coords_ecef"], dtype=np.float32)
    N = len(coords)
    
    edge_src, edge_dst, edge_attrs = [], [], []
    
    # Compute relative displacement vectors and pairwise distances
    # diff[i, j] = coords[j] - coords[i] (Vector pointing from station i to station j)
    diff = coords[None, :, :] - coords[:, None, :]  # Shape: (N, N, 3)
    dist_matrix = np.linalg.norm(diff, axis=-1)      # Shape: (N, N)
    
    for i in range(N):
        # Find neighbor stations within max_radius_km excluding self (distance > 0)
        neighbors = np.where((dist_matrix[i] > 0) & (dist_matrix[i] <= max_radius_km))[0]
        
        # If there are more than k_neighbors, select the k closest ones
        if len(neighbors) > k_neighbors:
            closest_k_idx = np.argsort(dist_matrix[i][neighbors])[:k_neighbors]
            neighbors = neighbors[closest_k_idx]
            
        for j in neighbors:
            d_ij = dist_matrix[i, j]
            delta_p = diff[i, j]      # (dx, dy, dz) vector from station i to station j
            delta_t = d_ij / v_phase  # Theoretical P-wave travel delay in seconds
            
            edge_src.append(j)  # Source node (neighbor)
            edge_dst.append(i)  # Target node (receiver)
            
            # Feature vector: [dx, dy, dz, distance_km, phase_delay_sec]
            # edge_attr = np.hstack([delta_p, d_ij, delta_t])
            edge_attr = np.hstack([delta_p])
            edge_attrs.append(edge_attr)
            
    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
    else:
        # Fallback tensor structure for isolated nodes
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)
        
    return edge_index, edge_attr

def row_to_pyg_data(row, aux_targets=False):
    # 1. Waveform signals: shape (N, T, 3)
    waves = np.array(row["aligned_waves"], dtype=np.float32)
    x = torch.tensor(waves, dtype=torch.float32)
    
    # 2. Station 3D Cartesian coordinates: shape (N, 3)
    coords = np.asarray(row["station_coords_ecef"], dtype=np.float32)
    if coords.ndim > 2:
        coords = coords.squeeze()
    pos = torch.tensor(coords, dtype=torch.float32)
    
    # 3. Edge topology (2, num_edges) and Edge attributes (num_edges, 5)
    edge_index = row["edge_index"].long() if isinstance(row["edge_index"], torch.Tensor) else torch.tensor(row["edge_index"], dtype=torch.long)
    edge_attr = row["edge_attr"].float() if isinstance(row["edge_attr"], torch.Tensor) else torch.tensor(row["edge_attr"], dtype=torch.float32)

    if aux_targets:
        p_arrivals = torch.tensor(row["p_arrival_aligned"], dtype=torch.float32) # (N,) 
        s_arrivals = torch.tensor(row["s_arrival_aligned"], dtype=torch.float32) # (N,) 
        back_azimuth = torch.tensor(row["back_azimuth_deg"], dtype=torch.float32) # (N,)
    
    # 4. Construct PyG Data object
    if aux_targets:
        data = Data(
                x=x,                      # Observed station waveforms (N, T, 3)
                pos=pos,                  # Station 3D coordinates (N, 3)
                edge_index=edge_index,    # Graph connectivity (2, E)
                edge_attr=edge_attr,      # Edge features (E, 5)
                y_p_arrival=p_arrivals, 
                y_s_arrival=s_arrivals, 
                y_baz=back_azimuth,
                source_id=str(row.name)   # Event identifier
            )
    else:
        data = Data(
            x=x,                      # Observed station waveforms (N, T, 3)
            pos=pos,                  # Station 3D coordinates (N, 3)
            edge_index=edge_index,    # Graph connectivity (2, E)
            edge_attr=edge_attr,      # Edge features (E, 5)
            source_id=str(row.name)   # Event identifier
        )
    return data

def visualize_pyg_spatial_graph(
    data,
    station_names: list = None,
    dist_feature_idx: int = 3,  # Index 3 corresponds to d_ij in a 5D edge vector
    dt_feature_idx: int = 4,   # Index 4 corresponds to dt_ij in a 5D edge vector
    title: str = "STEAD Spatial Sensor Graph"
):
    """Plots a PyG spatial graph with exact physical node coordinates,

    node position labels, and physical edge attribute callouts.
    """
    # 1. Extract PyG tensors
    edge_index = data.edge_index.cpu().numpy()
    pos = data.pos.cpu().numpy()  # Shape: (N, 2) or (N, 3)
    edge_attr = (
        data.edge_attr.cpu().numpy()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None
        else None
    )
    
    num_nodes = pos.shape[0]
    names = station_names if station_names else [f"STA_{i}" for i in range(num_nodes)]

    # 2. Build NetworkX DiGraph & Assign Positions
    G = nx.DiGraph()
    pos_dict = {}
    for i in range(num_nodes):
        G.add_node(i)
        # Use X=Longitude/ECEF-X and Y=Latitude/ECEF-Y for 2D plane
        pos_dict[i] = (pos[i, 0], pos[i, 1])

    # 3. Add Edges and Format Labels
    edge_labels = {}
    for k in range(edge_index.shape[1]):
        u, v = edge_index[0, k], edge_index[1, k]
        G.add_edge(u, v)

        if edge_attr is not None and edge_attr.ndim > 1 and edge_attr.shape[1] > dist_feature_idx:
            d_km = edge_attr[k, dist_feature_idx]
            if edge_attr.shape[1] > dt_feature_idx:
                dt_sec = edge_attr[k, dt_feature_idx]
                label_str = f"{d_km:.1f}km\n({dt_sec:.2f}s)"
            else:
                label_str = f"{d_km:.1f}km"
            edge_labels[(u, v)] = label_str

    # 4. Create Matplotlib Canvas
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw Nodes
    nx.draw_networkx_nodes(
        G,
        pos_dict,
        ax=ax,
        node_color='skyblue',
        node_size=700,
        edgecolors='navy',
        linewidths=1.5
    )

    # Draw Directed Curved Edges (Arc radius prevents bi-directional overlap)
    nx.draw_networkx_edges(
        G,
        pos_dict,
        ax=ax,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw Custom Node Labels (Station ID + Spatial Coordinates)
    node_text_labels = {
        i: f"{names[i]}\n({pos[i, 0]:.2f}, {pos[i, 1]:.2f})"
        for i in range(num_nodes)
    }
    nx.draw_networkx_labels(
        G,
        pos_dict,
        labels=node_text_labels,
        ax=ax,
        font_size=8,
        font_weight='bold',
        font_color='black'
    )

    # Draw Edge Attribute Labels
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos_dict,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
            font_color='darkred',
            rotate=False
        )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Longitude / ECEF X (km)", fontsize=11)
    ax.set_ylabel("Latitude / ECEF Y (km)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

def plot_event_diagnostic_dashboard(
    data, station_names=None, sample_rate_hz=100.0
):
    """Side-by-side diagnostic dashboard:

    - Left: Spatial Graph Geometry, Edge Distances, Coordinates, and Phase Annotations.
    - Right: Synchronized Waveforms with P/S Onset Markers.
    """
    fig = plt.figure(figsize=(13, 6))
    ax_graph = fig.add_subplot(1, 2, 1)

    # 1. Safely extract core graph tensors
    edge_index = (
        data.edge_index.cpu().numpy()
        if hasattr(data, "edge_index") and data.edge_index is not None
        else np.empty((2, 0))
    )
    pos = (
        data.pos.cpu().numpy()
        if hasattr(data, "pos") and data.pos is not None
        else None
    )
    edge_attr = (
        data.edge_attr.cpu().numpy()
        if hasattr(data, "edge_attr") and data.edge_attr is not None
        else None
    )
    x_waves = (
        data.x.cpu().numpy()
        if hasattr(data, "x") and data.x is not None
        else None
    )

    N = pos.shape[0]
    names = (
        station_names
        if station_names
        else [f"Sta {i}" for i in range(N)]
    )

    # 2. Extract Optional Auxiliary Targets dynamically
    p_arrivals = None
    if hasattr(data, "y_p_arrival") and getattr(data, "y_p_arrival") is not None:
        val = getattr(data, "y_p_arrival")
        p_arrivals = (
            val.cpu().numpy()
            if isinstance(val, torch.Tensor)
            else np.array(val)
        )

    s_arrivals = None
    if hasattr(data, "y_s_arrival") and getattr(data, "y_s_arrival") is not None:
        val = getattr(data, "y_s_arrival")
        s_arrivals = (
            val.cpu().numpy()
            if isinstance(val, torch.Tensor)
            else np.array(val)
        )

    baz_vals = None
    if hasattr(data, "y_baz") and getattr(data, "y_baz") is not None:
        val = getattr(data, "y_baz")
        baz_vals = (
            val.cpu().numpy()
            if isinstance(val, torch.Tensor)
            else np.array(val)
        )

    # 3. Panel 1: Graph Topology & Multi-Line Node Labels
    G = nx.DiGraph()
    pos_dict = {i: (pos[i, 0], pos[i, 1]) for i in range(N)}
    for i in range(N):
        G.add_node(i)

    edge_labels = {}
    for k in range(edge_index.shape[1]):
        u, v = edge_index[0, k], edge_index[1, k]
        G.add_edge(u, v)
        if (
            edge_attr is not None
            and edge_attr.ndim > 1
            and edge_attr.shape[1] >= 4
        ):
            edge_labels[(u, v)] = f"{edge_attr[k, 3]:.1f}km"

    # 1. Determine node colors based on whether seismic wave readings are zeroed out
    node_colors = []
    for i in range(N):
        # Check if the node's waveform data is zeroed out
        if x_waves is not None and np.abs(np.sum(x_waves[i])) < 1e-6:
            node_colors.append("crimson")  # Color for zeroed-out / target masked nodes
        else:
            node_colors.append("teal")  # Color for active context stations

    # 2. Draw nodes using the dynamic color list
    nx.draw_networkx_nodes(
        G,
        pos_dict,
        ax=ax_graph,
        node_color=node_colors,  # Pass the per-node color list
        node_size=900,
        edgecolors="black",
    )

    nx.draw_networkx_edges(
        G,
        pos_dict,
        ax=ax_graph,
        edge_color="gray",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        connectionstyle="arc3,rad=0.1",
    )

    # Construct multi-line text for each node
    node_labels = {}
    for i in range(N):
        lines = [f"{names[i]}", f"({pos[i, 0]:.1f}, {pos[i, 1]:.1f})"]

        # Format P and S arrivals
        aux_time_str = []
        if p_arrivals is not None and i < len(p_arrivals):
            val_p = float(p_arrivals[i])
            p_sec = (
                val_p / sample_rate_hz if val_p > 100 else val_p
            )
            aux_time_str.append(f"P:{p_sec:.2f}s")
        if s_arrivals is not None and i < len(s_arrivals):
            val_s = float(s_arrivals[i])
            s_sec = (
                val_s / sample_rate_hz if val_s > 100 else val_s
            )
            aux_time_str.append(f"S:{s_sec:.2f}s")

        if aux_time_str:
            lines.append(" | ".join(aux_time_str))

        # Format Back-Azimuth
        if baz_vals is not None and i < len(baz_vals):
            lines.append(f"BAZ:{float(baz_vals[i]):.1f}°")

        node_labels[i] = "\n".join(lines)

    nx.draw_networkx_labels(
        G,
        pos_dict,
        labels=node_labels,
        ax=ax_graph,
        font_size=7,
        font_weight="bold",
    )

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos_dict,
            edge_labels=edge_labels,
            ax=ax_graph,
            font_size=7,
            font_color="darkred",
            rotate=False,
        )

    ax_graph.set_title(
        "Station Geometry, Topology & Phase Annotations",
        fontsize=12,
        fontweight="bold",
    )
    ax_graph.set_xlabel("Longitude / ECEF X (km)", fontsize=10)
    ax_graph.set_ylabel("Latitude / ECEF Y (km)", fontsize=10)
    ax_graph.grid(True, linestyle=":", alpha=0.6)

    # 4. Panel 2: Stacked Seismic Traces with Arrival Onset Markers
    if x_waves is not None:
        ax_waves = fig.add_subplot(1, 2, 2)
        T = x_waves.shape[1]
        t_axis = np.linspace(0, T / sample_rate_hz, T)
        offset_step = 2.5  # Vertical spacing for stacked station traces

        # 1. Identify target masked nodes (boolean array where True = target node)
        target_mask = np.zeros(N, dtype=bool)
        if hasattr(data, "target_mask") and getattr(data, "target_mask") is not None:
            val = getattr(data, "target_mask")
            target_mask = (
                val.cpu().numpy()
                if isinstance(val, torch.Tensor)
                else np.array(val, dtype=bool)
            )
        elif hasattr(data, "mask_channel") and getattr(data, "mask_channel") is not None:
            val = getattr(data, "mask_channel")
            mask_ch = (
                val.cpu().numpy()
                if isinstance(val, torch.Tensor)
                else np.array(val)
            )
            target_mask = mask_ch.squeeze() == 0.0
        else:
            # Fallback: flag node as target if trace is flat zero or constant
            for idx in range(N):
                target_mask[idx] = (np.abs(x_waves[idx]).sum() < 1e-6) or (
                    np.std(x_waves[idx]) < 1e-6
                )

        masked_indices = np.where(target_mask)[0]

        # 2. Safely extract ground-truth target waveform tensor from data.y or data.target
        target_data = None
        if hasattr(data, "y") and getattr(data, "y") is not None:
            val = getattr(data, "y")
            target_data = (
                val.cpu().numpy()
                if isinstance(val, torch.Tensor)
                else np.array(val)
            )
        elif hasattr(data, "target") and getattr(data, "target") is not None:
            val = getattr(data, "target")
            target_data = (
                val.cpu().numpy()
                if isinstance(val, torch.Tensor)
                else np.array(val)
            )

        # 3. Plot stacked waveforms row-by-row
        for i in range(N):
            y_base = i * offset_step
            is_masked = i in masked_indices

            # Check if ground truth exists for this masked node
            has_gt = False
            gt_norm = None
            if is_masked and target_data is not None:
                mask_pos = np.where(masked_indices == i)[0][0]
                # Handle single-target shape (T, C) vs multi-target shape (num_masked, T, C)
                if len(masked_indices) == 1 and target_data.ndim <= 2:
                    gt_z = target_data[:, 2] if target_data.ndim == 2 else target_data
                else:
                    gt_z = (
                        target_data[mask_pos, :, 2]
                        if target_data.ndim == 3
                        else target_data[mask_pos]
                    )

                # Normalize ground-truth target channel for display offset
                max_amp = np.max(np.abs(gt_z))
                gt_norm = gt_z / (max_amp + 1e-6) if max_amp > 0 else gt_z
                has_gt = True

            # Extract input channel (z-component) from data.x
            z_channel = x_waves[i, :, 2] if x_waves.ndim == 3 else x_waves[i]
            max_z = np.max(np.abs(z_channel))
            z_norm = z_channel / (max_z + 1e-6) if max_z > 0 else z_channel

            if is_masked:
                if has_gt:
                    # Plot ground truth target waveform from data.y in crimson dashed style
                    ax_waves.plot(
                        t_axis,
                        gt_norm + y_base,
                        label=f"{names[i]}",
                        linewidth=0.8,
                        color="crimson",
                    )
                    ax_waves.text(
                        -1.5,
                        y_base + 0.3,
                        names[i],
                        fontweight="bold",
                        fontsize=8,
                        color="crimson",
                    )
            else:
                # Plot active context station trace in solid navy
                ax_waves.plot(
                    t_axis,
                    z_norm + y_base,
                    label=names[i],
                    linewidth=0.8,
                    color="navy",
                )
                ax_waves.text(
                    -1.5,
                    y_base + 0.3,
                    names[i],
                    fontweight="bold",
                    fontsize=8,
                    color="navy",
                )

            # P-wave onset marker
            if p_arrivals is not None and i < len(p_arrivals):
                val_p = float(p_arrivals[i])
                p_t = (
                    val_p * (3000.0 / sample_rate_hz)
                    if val_p <= 1.0
                    else (val_p / sample_rate_hz if val_p > 100 else val_p)
                )
                if 0 <= p_t <= t_axis[-1]:
                    ax_waves.vlines(
                        p_t,
                        y_base - 0.8,
                        y_base + 0.8,
                        colors="darkred" if is_masked else "crimson",
                        linestyles="--",
                        linewidth=1.2,
                    )
                    ax_waves.text(
                        p_t,
                        y_base + 0.9,
                        "P",
                        color="crimson",
                        fontweight="bold",
                        fontsize=7,
                        ha="center",
                    )

            # S-wave onset marker
            if s_arrivals is not None and i < len(s_arrivals):
                val_s = float(s_arrivals[i])
                s_t = (
                    val_s * (3000.0 / sample_rate_hz)
                    if val_s <= 1.0
                    else (val_s / sample_rate_hz if val_s > 100 else val_s)
                )
                if 0 <= s_t <= t_axis[-1]:
                    ax_waves.vlines(
                        s_t,
                        y_base - 0.8,
                        y_base + 0.8,
                        colors="darkgreen",
                        linestyles=":",
                        linewidth=1.2,
                    )
                    ax_waves.text(
                        s_t,
                        y_base + 0.9,
                        "S",
                        color="darkgreen",
                        fontweight="bold",
                        fontsize=7,
                        ha="center",
                    )

        ax_waves.set_title(
            "Z-Component Waveforms & Ground-Truth Targets",
            fontsize=12,
            fontweight="bold",
        )
        ax_waves.set_xlabel("Time (seconds)", fontsize=10)
        ax_waves.set_ylabel("Stacked Stations", fontsize=10)
        ax_waves.set_yticks([])
        ax_waves.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

def normalize_event_graph(
    data: Data,
    max_radius_km: float = 200.0,
    max_delay_sec: float = 33.0,
    eps: float = 1e-8,
) -> Data:
    """Normalizes all graph components in a PyG Data object at the event level.

    Normalizations Applied:
    1. Seismic Waveforms (data.x): Z-score scaled using event-level statistics computed ONLY over active context nodes (excluding masked target nodes). Saves `data.mean_event` and `data.std_event` for test-time denormalization.
    2. Station Coordinates (data.pos): Centroid Z-score scaled relative to event context stations. Saves `data.pos_mean` and `data.pos_std`.
    3. Edge Attributes (data.edge_attr): 5D vectors [dx, dy, dz, d_ij, dt_ij] scaled by max_radius_km and max_delay_sec into clean unit ranges.
    4. Auxiliary P/S Arrival Times (Optional): Scaled relative to waveform window length T into [0.0, 1.0].
    5. Auxiliary Back-Azimuth (Optional): Converted from degrees into continuous 2D [sin(rad), cos(rad)] vectors.
    """
    # Clone data object to prevent mutating the input in-place
    data = data.clone()

    # Identify active observed context nodes vs. masked target nodes
    if hasattr(data, "mask_channel") and data.mask_channel is not None:
        obs_mask = data.mask_channel.squeeze(-1) == 1.0
    elif hasattr(data, "target_mask") and data.target_mask is not None:
        obs_mask = ~data.target_mask
    else:
        obs_mask = torch.ones(
            data.x.size(0), dtype=torch.bool, device=data.x.device
        )

    # -------------------------------------------------------------------------
    # 1. Seismic Waveforms (x) - Event-Level Normalization
    # -------------------------------------------------------------------------
    if hasattr(data, "x") and data.x is not None:
        obs_waveforms = data.x[obs_mask] if obs_mask.any() else data.x
        # Calculate event-level scalar mean and std from context stations
        mean_event = obs_waveforms.mean()
        std_event = obs_waveforms.std()

        # Save statistics directly to the PyG Data object for denormalization
        data.mean_event = mean_event
        data.std_event = std_event

        # Apply event-level Z-score scaling across ALL nodes (context + target)
        data.x = (data.x - mean_event) / (std_event + eps)

    # -------------------------------------------------------------------------
    # 2. Station 3D ECEF Coordinates (pos) - Centroid Normalization
    # -------------------------------------------------------------------------
    if hasattr(data, "pos") and data.pos is not None:
        obs_pos = data.pos[obs_mask] if obs_mask.any() else data.pos
        pos_mean = obs_pos.mean(dim=0, keepdim=True)  # Shape: (1, 3)
        pos_std = obs_pos.std(dim=0, keepdim=True)  # Shape: (1, 3)
        pos_std = torch.where(
            pos_std < eps, torch.ones_like(pos_std), pos_std
        )

        data.pos_mean = pos_mean
        data.pos_std = pos_std
        data.pos = (data.pos - pos_mean) / (pos_std + eps)

    # -------------------------------------------------------------------------
    # 3. Edge Attributes (edge_attr) - Physical Max Bounds Scaling
    # -------------------------------------------------------------------------
    # Expects 5D edge vector: [dx, dy, dz, d_ij, dt_ij]
    if (
        hasattr(data, "edge_attr")
        and data.edge_attr is not None
        and data.edge_attr.size(1) >= 5
    ):
        edge_attr_norm = data.edge_attr.clone()
        # Scale 3D relative displacements (dx, dy, dz) and distance d_ij by max radius
        edge_attr_norm[:, 0:4] = edge_attr_norm[:, 0:4] / max_radius_km
        # Scale theoretical travel delay dt_ij by max expected delay
        edge_attr_norm[:, 4] = edge_attr_norm[:, 4] / max_delay_sec
        data.edge_attr = edge_attr_norm

    # -------------------------------------------------------------------------
    # 4. Optional Auxiliary Targets: P and S Phase Arrival Times
    # -------------------------------------------------------------------------
    T = (
        float(data.x.size(1))
        if hasattr(data, "x") and data.x is not None
        else 3000.0
    )

    # Search for P-wave arrival field
    p_attr = "y_p_arrival" if hasattr(data, "y_p_arrival") else None
    if p_attr is not None:
        p_val = getattr(data, p_attr).float()
        # Scale sample indices relative to fixed window length T -> [0.0, 1.0]
        setattr(data, "y_p_arrival_norm", p_val / T)

    # Search for S-wave arrival field
    s_attr = "y_s_arrival" if hasattr(data, "y_s_arrival") else None
    if s_attr is not None:
        s_val = getattr(data, s_attr).float()
        setattr(data, "y_s_arrival_norm", s_val / T)

    # -------------------------------------------------------------------------
    # 5. Optional Auxiliary Targets: Back-Azimuth (Circular 2D Vector)
    # -------------------------------------------------------------------------
    baz_attr = "y_baz" if hasattr(data, "y_baz") else None
    if baz_attr is not None:
        baz_deg = getattr(data, baz_attr).float()
        baz_rad = torch.deg2rad(baz_deg)
        # Convert 0-360 degree angle into continuous 2D [sin(rad), cos(rad)] vector in [-1.0, 1.0]^2
        baz_vector = torch.stack(
            [torch.sin(baz_rad), torch.cos(baz_rad)], dim=-1
        )
        setattr(data, "y_baz_norm", baz_vector)

    return data

def apply_node_masking(data, mask_ratio=None, num_mask=None):
    """Randomly selects target nodes to mask in a PyG Data object.

    Args:
        data: PyG Data object with data.x of shape (N, T, C)
        mask_ratio: Fraction of nodes to mask as targets (e.g., 0.25)
        num_mask: Fixed number of nodes to mask

    Returns:
        data: Updated PyG Data object with:
          - data.x: Waveform tensor with masked node(s) zeroed out (N, T, C)
          - data.y: Ground-truth wave reading(s) for the masked node(s)
            Shape: (T, C) if 1 node, or (num_masked, T, C) if multiple
          - data.target_mask: Boolean tensor (N,) where True indicates a masked station
          - data.mask_channel: Binary indicator feature channel (N, 1) [1.0 = context, 0.0 = target]
    """
    assert (
        mask_ratio is not None or num_mask is not None
    ), "Either mask_ratio or num_mask must be set"

    num_nodes = data.x.size(0)
    if mask_ratio is not None:
        num_masked = max(1, int(num_nodes * mask_ratio))
    else:
        num_masked = min(num_mask, num_nodes)

    # 1. Randomly select node indices to mask
    perm = torch.randperm(num_nodes, device=data.x.device)
    masked_indices = perm[:num_masked]

    target_mask = torch.zeros(
        num_nodes, dtype=torch.bool, device=data.x.device
    )
    target_mask[masked_indices] = True

    # 2. Extract ground-truth target waveforms BEFORE zeroing out
    targets = data.x[target_mask]  # Shape: (num_masked, T, C)
    if num_masked == 1:
        data.y = targets.squeeze(0)  # Shape: (T, C)
    else:
        data.y = targets  # Shape: (num_masked, T, C)

    # Note: If you explicitly prefer a Python list of tensors:
    # data.y = [t for t in targets]

    # 3. Clone data.x and zero out waveform signals at target nodes
    data.x = data.x.clone()
    data.x[target_mask] = 0.0

    # 4. Attach indicator metadata to the PyG Data object
    data.target_mask = target_mask
    data.mask_channel = (~target_mask).float().unsqueeze(-1)  # Shape: (N, 1)

    return data