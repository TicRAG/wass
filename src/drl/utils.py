# src/drl/utils.py
import json
import torch
import numpy as np
from torch_geometric.data import Data

def workflow_json_to_pyg_data(json_file_path: str, scaler=None) -> Data:
    """
    Converts a WRENCH workflow JSON file to a PyTorch Geometric Data object.
    Adds a 4th dimension for task status and correctly handles scaling.
    Status codes: 0=WAITING, 1=READY
    """
    with open(json_file_path, 'r') as f:
        wf_data = json.load(f)

    tasks = wf_data['workflow']['tasks']
    
    node_features = []
    for task in tasks:
        # Always create a list with 4 features
        features = [
            float(task.get('runtime', 0.0)),
            float(task.get('flops', 0.0)),
            float(task.get('memory', 0.0)),
            1.0 if not task.get('dependencies') else 0.0 # Status feature
        ]
        node_features.append(features)

    node_features_np = np.array(node_features, dtype=np.float32)

    # If a scaler is provided, scale only the first 3 features (columns)
    if scaler:
        # Extract the first 3 columns, transform them, and place them back
        # This ensures the 4th column (status) is preserved
        node_features_np[:, :3] = scaler.transform(node_features_np[:, :3])

    x = torch.tensor(node_features_np, dtype=torch.float)

    # Task ID to index mapping and edge list creation remain the same
    task_id_to_idx = {task['id']: i for i, task in enumerate(tasks)}
    edge_list = []
    for i, task in enumerate(tasks):
        for parent_id in task.get('dependencies', []):
             if parent_id in task_id_to_idx:
                parent_idx = task_id_to_idx[parent_id]
                edge_list.append([parent_idx, i])

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data