import os
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn

class SimplePerformancePredictor(nn.Module):
    """
    修正后的简单性能预测器。
    确保与阶段二训练的模型结构和加载逻辑完全一致。
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def load_model(self, model_path: str):
        """健壮的模型加载方法"""
        if not os.path.exists(model_path):
            print(f"[致命错误] 预测器模型文件未找到: {model_path}")
            raise FileNotFoundError(f"Predictor model not found at {model_path}")
        
        try:
            # 使用 weights_only=False 以兼容旧的 PyTorch 保存格式
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            
            # 检查 state_dict 是否被包裹在另一个字典中
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'performance_predictor' in state_dict:
                state_dict = state_dict['performance_predictor']

            self.load_state_dict(state_dict)
            print(f"✅ 性能预测器模型成功加载: {model_path}")

        except Exception as e:
            print(f"[致命错误] 加载性能预测器 state_dict 失败。错误: {e}")
            print("请确保阶段二训练的模型与阶段三使用的 SimplePerformancePredictor 结构匹配。")
            raise e
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)
    
    def predict(self, x: np.ndarray) -> float:
        """预测接口"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float().unsqueeze(0)
            else:
                x_tensor = x.float().unsqueeze(0)
            out = self.forward(x_tensor).item()
        return out

# 保留 PerformancePredictor 以防万一，但 DRL 训练器不再使用它
class PerformancePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        print("[警告] GNN 版 PerformancePredictor 未被 DRL 训练器使用。")
    def load_model(self, model_path: str):
        pass