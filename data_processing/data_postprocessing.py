import torch
from tqdm import tqdm

from data_utils import *

for dataset_path in ["data/train_seismic_graphs.pt", "data/val_seismic_graphs.pt", "data/test_seismic_graphs.pt"]:
    dataset = torch.load(dataset_path, weights_only=False)

    normalized_dataset = [normalize_event_graph(data) for data in tqdm(dataset)]
    masked_dataset = [apply_node_masking(data, num_mask=1) for data in tqdm(normalized_dataset)]

    torch.save(masked_dataset, dataset_path.split(".")[0] + "_processed.pt")