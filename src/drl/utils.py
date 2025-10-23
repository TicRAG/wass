# src/drl/utils.py
import json
import torch
import numpy as np
from torch_geometric.data import Data


def _extract_tasks(workflow_json: dict):
    """Return a normalized list of task dicts supporting both internal and wfcommons formats.

    Supported formats:
      1. Project-generated: { 'workflow': { 'tasks': [ { 'id': ..., 'dependencies': [...], 'flops': ..., 'memory': ... } ] } }
      2. wfcommons (converted): { 'workflow': { 'specification': { 'tasks': [ { 'id': ..., 'parents': [...], 'flops': ..., 'memory': ... } ] } } }

    This function will:
      * Prefer workflow['tasks'] if present.
      * Else, attempt workflow['specification']['tasks'].
      * Map 'parents' -> 'dependencies' for downstream logic.
      * Ensure missing numeric fields default to 0.0.
    """
    if not isinstance(workflow_json, dict):
        return []

    wf_section = workflow_json.get('workflow', {})
    # Build execution runtime mapping if available (wfcommons format)
    exec_tasks = wf_section.get('execution', {}).get('tasks', [])
    runtime_map = {}
    for et in exec_tasks:
        if isinstance(et, dict) and 'id' in et:
            runtime_map[et['id']] = float(et.get('runtimeInSeconds', 0.0))

    # Case 1: existing project format
    if 'tasks' in wf_section and isinstance(wf_section['tasks'], list):
        tasks = wf_section['tasks']
    else:
        # Case 2: wfcommons format
        spec = wf_section.get('specification', {})
        tasks = spec.get('tasks', []) if isinstance(spec.get('tasks', []), list) else []

    normalized = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        # Copy shallow
        nt = dict(t)
        # Map parents -> dependencies if dependencies missing
        if 'dependencies' not in nt:
            parents = nt.get('parents', [])
            if isinstance(parents, list):
                nt['dependencies'] = parents
            else:
                nt['dependencies'] = []
        # Guarantee numeric fields
        nt['flops'] = float(nt.get('flops', 0.0))
        nt['memory'] = float(nt.get('memory', 0.0))
        # Populate runtime: prefer existing 'runtime', else execution runtime map
        if 'runtime' in nt:
            nt['runtime'] = float(nt.get('runtime', 0.0))
        else:
            nt['runtime'] = runtime_map.get(nt.get('id'), 0.0)
        normalized.append(nt)
    return normalized

def workflow_json_to_pyg_data(json_file_path: str, scaler=None) -> Data:
    """
    Converts a WRENCH workflow JSON file to a PyTorch Geometric Data object.
    Adds a 4th dimension for task status and correctly handles scaling.
    Status codes: 0=WAITING, 1=READY
    """
    with open(json_file_path, 'r') as f:
        wf_data = json.load(f)

    tasks = _extract_tasks(wf_data)
    
    node_features = []
    for task in tasks:
        # Always create a list with 4 features: runtime, flops, memory, status
        features = [
            float(task.get('runtime', 0.0)),
            float(task.get('flops', 0.0)),
            float(task.get('memory', 0.0)),
            1.0 if not task.get('dependencies') else 0.0  # Status feature
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
    task_id_to_idx = {task['id']: i for i, task in enumerate(tasks) if 'id' in task}
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