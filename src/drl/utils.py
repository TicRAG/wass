# src/drl/utils.py
import json
import torch
import numpy as np
from torch_geometric.data import Data

def workflow_json_to_pyg_data(json_file_path: str, scaler=None) -> Data:
    """
    Converts a WRENCH workflow JSON file to a PyTorch Geometric Data object.
    If a scaler is provided, it will be used to transform the node features.
    """
    with open(json_file_path, 'r') as f:
        wf_data = json.load(f)

    tasks = wf_data['workflow']['tasks']
    
    node_features = []
    for task in tasks:
        features = [
            float(task.get('runtime', 0.0)),
            float(task.get('flops', 0.0)),
            float(task.get('memory', 0.0))
        ]
        node_features.append(features)

    # Apply scaling if a scaler is provided
    if scaler:
        node_features = scaler.transform(node_features)

    x = torch.tensor(node_features, dtype=torch.float)

    task_name_to_id = {task['name']: i for i, task in enumerate(tasks)}
    edge_list = []
    for i, task in enumerate(tasks):
        for parent_id in task.get('dependencies', []):
             parent_task = next((t for t in tasks if t['id'] == parent_id), None)
             if parent_task and parent_task['name'] in task_name_to_id:
                parent_idx = task_name_to_id[parent_task['name']]
                edge_list.append([parent_idx, i])

    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data