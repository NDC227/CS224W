import pandas as pd
import matplotlib.pyplot as plt

file_name = "data/chunk2.hdf5"
csv_file = "data/chunk2.csv"

# Load index metadata
df = pd.read_csv(csv_file)

import re

df_eq = df[df["trace_category"] == "earthquake_local"]
temp = df["snr_db"][0]
temp.split("[")[1].split("]")[0].split(" ")
# def process(x):
#     processed = re.findall(r"\d+.\d*", x)
#     return processed
df_eq = df_eq[df_eq["snr_db"].apply(lambda s: min(map(float, re.findall(r"\d+.\d*", s)))) >= 10.0]
print(len(df_eq))
df_eq.head()

event_counts = df_eq.groupby("source_id").size()
valid_events = event_counts[event_counts >= 5].index
filtered_df_eq = df_eq[df_eq["source_id"].isin(valid_events)]
filtered_df_eq.head()

# Convert trace_start_times into dates using strptime
from datetime import datetime
filtered_df_eq["trace_start_time"] = pd.to_datetime(filtered_df_eq["trace_start_time"], format='mixed')
filtered_df_eq

import numpy as np

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

def process_row_coords(row):
    return geodetic_to_ecef(
        lat=row["receiver_latitude"],
        lon=row["receiver_longitude"],
        elevation_m=row["receiver_elevation_m"]
    )

# Assign 3D coordinate vector [x, y, z] to a new column
filtered_df_eq["station_coords_ecef"] = filtered_df_eq.apply(process_row_coords, axis=1)
filtered_df_eq.head()


import h5py
import obspy
from obspy import UTCDateTime
import numpy as np

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


from tqdm import tqdm
dtf2 = h5py.File(file_name, 'r')

processed_waves = []
for i, row in tqdm(filtered_df_eq.iterrows()):
    processed_wave = process_stead_trace(dtf2, row["trace_name"])
    processed_waves.append(processed_wave)
filtered_df_eq["processed_waves"] = processed_waves
filtered_df_eq.head()


grouped_filtered_df_eq = filtered_df_eq.groupby("source_id").agg(list)
grouped_filtered_df_eq.head()


def valid_time_span(times, max_span_sec=60):
    return (max(times) - min(times)).total_seconds() <= max_span_sec

valid_events_mask = grouped_filtered_df_eq["trace_start_time"].apply(valid_time_span)
grouped_clean = grouped_filtered_df_eq[valid_events_mask].copy()
grouped_filtered_df_eq = grouped_clean
grouped_filtered_df_eq.head()


sample_trace_name = grouped_filtered_df_eq.iloc[0]["trace_name"][0]
dtf2 = h5py.File(file_name, 'r')
waves = process_stead_trace(dtf2, sample_trace_name)
waves


fig = plt.figure()
ax = fig.add_subplot(311)         
plt.plot(waves[:,0], 'k')
plt.rcParams["figure.figsize"] = (8, 5)
legend_properties = {'weight':'bold'}    
plt.tight_layout()
ymin, ymax = ax.get_ylim()      
plt.ylabel('Amplitude counts', fontsize=12) 
ax.set_xticklabels([]);


min_start_times = []
for i, row in grouped_filtered_df_eq.iterrows():
    min_start_time = min(row["trace_start_time"])
    min_start_times.append(min_start_time)
grouped_filtered_df_eq["min_start_time"] = min_start_times

relative_times = []
for i, row in grouped_filtered_df_eq.iterrows():
    relative_time_i = []
    for start_time in row["trace_start_time"]:
        relative_time_i.append(start_time - row["min_start_time"])
    relative_times.append(relative_time_i)
grouped_filtered_df_eq["relative_time"] = relative_times
grouped_filtered_df_eq.head()


# Within each row, align the processed_waves according to the relative_time.
# The processed_wave correpsonding to relative time 0 should not change.
# All other waves should be shifted in time accordingly
# processed_waves is at 100 Hz

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

grouped_filtered_df_eq["aligned_waves"] = grouped_filtered_df_eq.apply(align_processed_waves, axis=1)
grouped_filtered_df_eq.head()


# Limit aligned_waves to be 6000 samples long
grouped_filtered_df_eq["aligned_waves"] = grouped_filtered_df_eq["aligned_waves"].apply(lambda x: np.asarray(x)[:, :6000, :])
grouped_filtered_df_eq


import numpy as np
import torch

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
            edge_attr = np.hstack([delta_p, d_ij, delta_t])
            edge_attrs.append(edge_attr)
            
    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
    else:
        # Fallback tensor structure for isolated nodes
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)
        
    return edge_index, edge_attr


# Compute edge topologies and feature tensors
edges_and_attrs = grouped_filtered_df_eq.apply(build_event_edges, axis=1)

# Store results in separate DataFrame columns
grouped_filtered_df_eq["edge_index"] = [item[0] for item in edges_and_attrs]
grouped_filtered_df_eq["edge_attr"] = [item[1] for item in edges_and_attrs]


has_edges = grouped_filtered_df_eq["edge_index"].apply(lambda e: e.shape[1] > 0)
grouped_filtered_df_eq = grouped_filtered_df_eq[has_edges]
grouped_filtered_df_eq["edge_index"]


import torch
import numpy as np
from torch_geometric.data import Data

def row_to_pyg_data(row):
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
    
    # 4. Construct PyG Data object
    data = Data(
        x=x,                      # Observed station waveforms (N, T, 3)
        pos=pos,                  # Station 3D coordinates (N, 3)
        edge_index=edge_index,    # Graph connectivity (2, E)
        edge_attr=edge_attr,      # Edge features (E, 5)
        source_id=str(row.name)   # Event identifier
    )
    return data

# Convert all valid events into a list of PyG Data objects
pyg_dataset = [row_to_pyg_data(row) for _, row in grouped_filtered_df_eq.iterrows()]


pyg_dataset


sample_data = pyg_dataset[0]

print("PyG Data Object Summary:")
print("x (Waveforms):    ", sample_data.x.shape)          # Expected: (N, T, 3)
print("pos (Coordinates):", sample_data.pos.shape)        # Expected: (N, 3)
print("edge_index:       ", sample_data.edge_index.shape) # Expected: (2, E)
print("edge_attr:        ", sample_data.edge_attr.shape)  # Expected: (E, 5)
print("source_id:        ", sample_data.source_id)


import random

# 1. Extract all unique event IDs from the PyG dataset
unique_events = list(set([data.source_id for data in pyg_dataset]))

# 2. Shuffle event IDs deterministically
random.seed(42)
random.shuffle(unique_events)

# 3. Calculate split index boundaries (70% train, 15% val, 15% test)
num_events = len(unique_events)
train_cutoff = int(0.70 * num_events)
val_cutoff = int(0.85 * num_events)

train_event_ids = set(unique_events[:train_cutoff])
val_event_ids = set(unique_events[train_cutoff:val_cutoff])
test_event_ids = set(unique_events[val_cutoff:])

# 4. Filter the PyG Data objects into their respective splits
train_dataset = [d for d in pyg_dataset if d.source_id in train_event_ids]
val_dataset = [d for d in pyg_dataset if d.source_id in val_event_ids]
test_dataset = [d for d in pyg_dataset if d.source_id in test_event_ids]

print(f"Total Events: {num_events}")
print(f"Train Graphs: {len(train_dataset)}")
print(f"Val Graphs:   {len(val_dataset)}")
print(f"Test Graphs:  {len(test_dataset)}")


import torch

torch.save(train_dataset, "data/train_seismic_graphs.pt")
torch.save(val_dataset, "data/val_seismic_graphs.pt")
torch.save(test_dataset, "data/test_seismic_graphs.pt")


