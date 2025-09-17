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
    
    def extract_graph_features(self, dag, node_features, focus_task_id=None):
        """
        提取图特征用于GNN预测
        这是一个兼容性方法，将图数据转换为预测器可处理的特征向量
        """
        # 简化的图特征提取：将节点特征展平并添加一些图结构信息
        if isinstance(node_features, dict):
            # 如果是字典格式，提取数值特征
            feature_list = []
            for node_name, features in node_features.items():
                if isinstance(features, dict):
                    # 提取关键特征
                    feature_values = [
                        features.get('speed', 0.0),
                        features.get('available_time', 0.0),
                        features.get('queue_length', 0.0)
                    ]
                    feature_list.extend(feature_values)
                else:
                    # 如果是数组或列表
                    if hasattr(features, '__iter__') and not isinstance(features, str):
                        feature_list.extend(list(features))
                    else:
                        feature_list.append(float(features))
            
            # 限制特征维度，避免过大
            max_features = 50  # 根据预测器输入维度调整
            if len(feature_list) > max_features:
                feature_list = feature_list[:max_features]
            elif len(feature_list) < max_features:
                # 填充到期望维度
                feature_list.extend([0.0] * (max_features - len(feature_list)))
            
            return np.array(feature_list, dtype=np.float32)
        elif isinstance(node_features, np.ndarray):
            return node_features.astype(np.float32)
        else:
            # 默认返回一个简单的特征向量
            return np.zeros(50, dtype=np.float32)

# 保留 PerformancePredictor 以防万一，但 DRL 训练器不再使用它
class PerformancePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        print("[警告] GNN 版 PerformancePredictor 未被 DRL 训练器使用。")
    def load_model(self, model_path: str):
        pass