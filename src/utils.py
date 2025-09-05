"""工具函数和日志设置."""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Generator
from pathlib import Path

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
