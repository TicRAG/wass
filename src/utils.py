"""工具函数和日志设置."""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Generator
from pathlib import Path
import sys 

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """设置统一的日志器."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件handler（如果指定）
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

@contextmanager
def time_stage(stage_name: str, logger: logging.Logger = None) -> Generator[Dict[str, Any], None, None]:
    """计时上下文管理器."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    stage_info = {"name": stage_name, "start_time": start_time}
    
    logger.info(f"[Stage] {stage_name} 开始")
    try:
        yield stage_info
        end_time = time.time()
        elapsed = end_time - start_time
        stage_info["end_time"] = end_time
        stage_info["elapsed_seconds"] = elapsed
        logger.info(f"[Stage] {stage_name} 完成, 耗时: {elapsed:.2f}s")
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        stage_info["end_time"] = end_time
        stage_info["elapsed_seconds"] = elapsed
        stage_info["error"] = str(e)
        logger.error(f"[Stage] {stage_name} 失败, 耗时: {elapsed:.2f}s, 错误: {e}")
        raise

def calculate_conflict_rate(L) -> float:
    """计算标签矩阵中的冲突率."""
    import numpy as np
    
    if hasattr(L, 'shape'):
        # numpy array
        n_samples, n_lfs = L.shape
        if n_samples == 0 or n_lfs <= 1:
            return 0.0
        
        conflicts = 0
        total_pairs = 0
        
        for i in range(n_samples):
            row = L[i]
            # 获取非abstain的标签
            valid_labels = row[row != -1]
            if len(valid_labels) <= 1:
                continue
            
            # 检查是否有冲突
            unique_labels = np.unique(valid_labels)
            if len(unique_labels) > 1:
                conflicts += 1
            total_pairs += 1
        
        return conflicts / max(1, total_pairs)
    else:
        # 其他格式，简单返回0
        return 0.0

def get_logger(name, level=logging.INFO):
    """
    Returns a configured logger.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    # This check prevents adding duplicate handlers if the function is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

# -------------------------------------------------------------
# Knowledge Base Feature Extraction (used by generate_kb_dataset)
# -------------------------------------------------------------
def extract_features_for_kb(task, node_id: str, platform, workflow) -> Dict[str, Any]:
    """Extract a minimal, model-agnostic feature dict for a (task,node) decision.

    Parameters
    ----------
    task : wrench.workflows.Task
        The workflow task just scheduled or considered.
    node_id : str
        Target node name.
    platform : wrench.platforms.Platform
        Platform object (provides node characteristics / bandwidth).
    workflow : wrench.workflows.Workflow
        Workflow DAG (to query edges, parents, children etc.).

    Returns
    -------
    Dict[str, Any]
        Flat dictionary of numeric / categorical features.
    """
    try:
        node = platform.get_node(node_id)
    except Exception:
        node = None

    # Basic structural features
    num_parents = len(task.parents)
    num_children = len(task.children)

    # Aggregate input data size (if API available)
    total_input_data = 0.0
    for p in task.parents:
        try:
            total_input_data += workflow.get_edge_data_size(p.id, task.id)
        except Exception:
            pass

    features: Dict[str, Any] = {
        "task_id": getattr(task, 'id', None),
        "task_comp_size": getattr(task, 'computation_size', 0.0),
        "num_parents": num_parents,
        "num_children": num_children,
        "total_input_data": total_input_data,
        "node_id": node_id,
    }

    if node is not None:
        # Common attributes; guard with getattr for API robustness
        for attr, key in [
            ("speed", "node_speed"),
            ("core_count", "node_cores"),
            ("memory", "node_memory"),
        ]:
            features[key] = getattr(node, attr, None)

    return features