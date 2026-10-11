import pandas as pd
import h5py
import numpy as np
import re
import torch
from tqdm import tqdm
import random

from data_utils import *

file_name = "raw_data/chunk2.hdf5"
csv_file = "raw_data/chunk2.csv"

# Load index metadata
df = pd.read_csv(csv_file)

df_eq = df[df["trace_category"] == "earthquake_local"]
temp = df["snr_db"][0]
temp.split("[")[1].split("]")[0].split(" ")

df_eq = df_eq[df_eq["snr_db"].apply(lambda s: min(map(float, re.findall(r"\d+.\d*", s)))) >= 10.0]
print(len(df_eq))

# deduplicate based on trace_name
df_eq = df_eq.drop_duplicates(subset=["receiver_latitude", "receiver_longitude", "source_id"])

event_counts = df_eq.groupby("source_id").size()
valid_events = event_counts[event_counts >= 5].index
filtered_df_eq = df_eq[df_eq["source_id"].isin(valid_events)]

# Convert trace_start_times into dates using strptime
filtered_df_eq["trace_start_time"] = pd.to_datetime(filtered_df_eq["trace_start_time"], format='mixed')

def process_row_coords(row):
    return geodetic_to_ecef(
        lat=row["receiver_latitude"],
        lon=row["receiver_longitude"],
        elevation_m=row["receiver_elevation_m"]
    )

# Assign 3D coordinate vector [x, y, z] to a new column
filtered_df_eq["station_coords_ecef"] = filtered_df_eq.apply(process_row_coords, axis=1)

dtf2 = h5py.File(file_name, 'r')

processed_waves = []
for i, row in tqdm(filtered_df_eq.iterrows()):
    processed_wave = process_stead_trace(dtf2, row["trace_name"])
    processed_waves.append(processed_wave)
filtered_df_eq["processed_waves"] = processed_waves

# =============================
# Group by source_id and aggregate lists of relevant columns
# =============================

grouped_filtered_df_eq = filtered_df_eq.groupby("source_id").agg(list)

def valid_time_span(times, max_span_sec=60):
    return (max(times) - min(times)).total_seconds() <= max_span_sec

valid_events_mask = grouped_filtered_df_eq["trace_start_time"].apply(valid_time_span)
grouped_clean = grouped_filtered_df_eq[valid_events_mask].copy()
grouped_filtered_df_eq = grouped_clean

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

# Within each row, align the processed_waves according to the relative_time.
# The processed_wave correpsonding to relative time 0 should not change.
# All other waves should be shifted in time accordingly
# processed_waves is at 100 Hz

grouped_filtered_df_eq["aligned_waves"] = grouped_filtered_df_eq.apply(align_processed_waves, axis=1)

# Limit aligned_waves to be 6000 samples long
grouped_filtered_df_eq["aligned_waves"] = grouped_filtered_df_eq["aligned_waves"].apply(lambda x: np.asarray(x)[:, :6000, :])

# Align p/s arrival times
grouped_filtered_df_eq = grouped_filtered_df_eq.apply(align_arrival_times, axis=1)

# Compute edge topologies and feature tensors
edges_and_attrs = grouped_filtered_df_eq.apply(build_event_edges, axis=1)

# Store results in separate DataFrame columns
grouped_filtered_df_eq["edge_index"] = [item[0] for item in edges_and_attrs]
grouped_filtered_df_eq["edge_attr"] = [item[1] for item in edges_and_attrs]

has_edges = grouped_filtered_df_eq["edge_index"].apply(lambda e: e.shape[1] > 0)
grouped_filtered_df_eq = grouped_filtered_df_eq[has_edges]
grouped_filtered_df_eq["edge_index"]

# =============================
# Save the PyG dataset into train, val, and test splits based on unique event IDs
# =============================

# Convert all valid events into a list of PyG Data objects
pyg_dataset = [row_to_pyg_data(row, aux_targets=True) for _, row in grouped_filtered_df_eq.iterrows()]

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

torch.save(train_dataset, "data/train_seismic_graphs.pt")
torch.save(val_dataset, "data/val_seismic_graphs.pt")
torch.save(test_dataset, "data/test_seismic_graphs.pt")