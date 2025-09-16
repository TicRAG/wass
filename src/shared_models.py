import os
from typing import Optional, List
import torch
import torch.nn as nn
import numpy as np

class SimplePerformancePredictor(nn.Module):
    """
    A unified, standardized performance predictor model.
    All scripts that train or use this model should import this class.
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
        """A robust method for loading the model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Predictor model file not found: {model_path}")
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.load_state_dict(state_dict)
            print(f"✅ Performance predictor model loaded successfully: {model_path}")
        except Exception as e:
            print(f"[FATAL] Failed to load performance predictor state_dict. Error: {e}")
            raise e
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict(self, x: np.ndarray) -> float:
        """Prediction interface."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().unsqueeze(0)
            out = self.forward(x_tensor).item()
        return out