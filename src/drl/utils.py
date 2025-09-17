# src/drl/utils.py
import json
import torch
from torch_geometric.data import Data

def workflow_json_to_pyg_data(json_file_path: str) -> Data:
    """
    将WRENCH工作流的JSON文件转换为PyTorch Geometric的Data对象。

    Args:
        json_file_path (str): 工作流JSON文件的路径。

    Returns:
        torch_geometric.data.Data: GNN可以处理的图数据对象。
    """
    with open(json_file_path, 'r') as f:
        wf_data = json.load(f)

    tasks = wf_data['workflow']['tasks']
    
    # 1. 节点特征 (x): 
    # --- 这是修正的部分 ---
    # 使用 .get('runtime', 0.0) 来安全地获取值。
    # 如果 'runtime' 键不存在，则使用默认值 0.0，避免程序崩溃。
    node_features = [float(task.get('runtime', 0.0)) for task in tasks]
    # --- 修正结束 ---
    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)

    # 2. 边索引 (edge_index): 构建任务依赖关系图。
    task_name_to_id = {task['name']: i for i, task in enumerate(tasks)}
    edge_list = []
    for i, task in enumerate(tasks):
        # 同样，安全地获取 parents 列表
        for parent_name in task.get('parents', []):
            if parent_name in task_name_to_id:
                parent_id = task_name_to_id[parent_name]
                edge_list.append([parent_id, i])
    
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data