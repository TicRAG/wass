import os
from typing import Optional, List
import torch
import torch.nn as nn
import numpy as np

class SimplePerformancePredictor(nn.Module):
    """
    一个统一的、标准化的性能预测器模型。
    所有训练和使用此模型的脚本都应从这里导入此类。
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # 直接定义层，而不是将它们包裹在 nn.Sequential 中
        # 这很可能与您已保存的 .pth 文件的结构相匹配
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)

        self.output_layer = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.dropout3(self.relu3(self.layer3(x)))
        return self.output_layer(x)

    def load_model(self, model_path: str):
        """健壮的模型加载方法"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"预测器模型文件未找到: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.load_state_dict(state_dict)
            print(f"✅ 性能预测器模型成功加载: {model_path}")
        except Exception as e:
            print(f"[致命错误] 加载性能预测器 state_dict 失败。错误: {e}")
            raise e

    def predict(self, x: np.ndarray) -> float:
        """预测接口"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().unsqueeze(0)
            out = self.forward(x_tensor).item()
        return out