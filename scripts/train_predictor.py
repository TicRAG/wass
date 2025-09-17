# scripts/train_predictor.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# --- 新的、更简单的预测器模型 ---
class SimplePredictor(nn.Module):
    """一个简单的MLP，输入是手工设计的统计特征。"""
    def __init__(self, input_dim: int):
        super(SimplePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

def extract_statistical_features(workflow_file: str) -> list:
    """从工作流JSON文件中提取一组统计特征。"""
    with open(workflow_file, 'r') as f:
        data = json.load(f)
    
    tasks = data['workflow']['tasks']
    if not tasks:
        return [0.0] * 5 # 返回默认值

    num_tasks = len(tasks)
    total_flops = sum(float(t.get('flops', 0.0)) for t in tasks)
    total_memory = sum(float(t.get('memory', 0.0)) for t in tasks)
    avg_flops = total_flops / num_tasks if num_tasks > 0 else 0.0
    
    # 估算工作流深度（关键路径长度）
    task_depth = {}
    for task in tasks:
        if not task.get('dependencies'):
            task_depth[task['id']] = 1
        else:
            max_parent_depth = 0
            for parent_id in task.get('dependencies'):
                if parent_id in task_depth:
                    max_parent_depth = max(max_parent_depth, task_depth[parent_id])
            task_depth[task['id']] = max_parent_depth + 1
            
    critical_path_length = max(task_depth.values()) if task_depth else 0

    return [num_tasks, total_flops, total_memory, avg_flops, critical_path_length]

# --- 配置 ---
KB_METADATA_PATH = "data/knowledge_base/workflow_metadata.csv"
WORKFLOW_DIR = "data/workflows"
MODEL_SAVE_DIR = "models/saved_models"
PREDICTOR_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "performance_predictor.pth")
FEATURE_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "feature_scaler.joblib")
MAKESPAN_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "makespan_scaler.joblib")

EPOCHS = 250
BATCH_SIZE = 8
LEARNING_RATE = 0.001

def main():
    print("🚀 [Phase 2.2] Starting Performance Predictor Training (Statistical Features Version)...")
    
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # 1. 加载元数据
    print("\n[Step 1/5] Loading Knowledge Base metadata...")
    metadata = pd.read_csv(KB_METADATA_PATH)
    print(f"✅ Loaded metadata for {len(metadata)} workflows.")

    # 2. 提取统计特征
    print("\n[Step 2/5] Extracting statistical features from JSON files...")
    features_list = []
    for filename in metadata['workflow_file']:
        # 构造完整路径
        filepath = os.path.join(WORKFLOW_DIR, filename)
        if os.path.exists(filepath):
            features_list.append(extract_statistical_features(filepath))
        else:
            print(f"  [Warning] File not found, skipping: {filepath}")

    if not features_list:
        print("❌ No features were extracted. Aborting.")
        return

    # 3. 归一化输入和输出
    print("\n[Step 3/5] Normalizing input features and output makespans...")
    feature_scaler = StandardScaler()
    makespan_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(features_list)
    y_scaled = makespan_scaler.fit_transform(metadata['makespan'].values.reshape(-1, 1))
    
    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_scaled, dtype=torch.float32)
    print("✅ Features and targets normalized.")
    
    # 4. 初始化模型
    input_dim = X_train.shape[1]
    model = SimplePredictor(input_dim=input_dim)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"✅ Model initialized with {input_dim} input features.")
    
    # 5. 训练
    print("\n[Step 4/5] Starting training...")
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 25 == 0:
            # 打印一些样本的预测结果来监控训练过程
            with torch.no_grad():
                sample_preds_scaled = model(X_train[:5])
                sample_preds_real = makespan_scaler.inverse_transform(sample_preds_scaled.numpy())
                sample_targets_real = makespan_scaler.inverse_transform(y_train[:5].numpy())
                errors = np.mean(np.abs(sample_preds_real - sample_targets_real) / sample_targets_real) * 100
                print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, Avg Prediction Error (on 5 samples): {errors:.2f}%")

    print("✅ Training finished.")

    # 6. 保存所有东西
    print("\n[Step 5/5] Saving model and scalers...")
    torch.save(model.state_dict(), PREDICTOR_MODEL_PATH)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(makespan_scaler, MAKESPAN_SCALER_PATH)

    print(f"💾 Predictor model saved to: {PREDICTOR_MODEL_PATH}")
    print(f"💾 Feature scaler saved to: {FEATURE_SCALER_PATH}")
    print(f"💾 Makespan scaler saved to: {MAKESPAN_SCALER_PATH}")
    print("\n🎉 [Phase 2.2] Completed Successfully! 🎉")

if __name__ == "__main__":
    main()