import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path
import sys

# 添加项目路径以导入本地模块
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.knowledge_base.json_kb import JSONKnowledgeBase
# [FIX] 从共享模块导入统一的模型定义
from src.shared_models import SimplePerformancePredictor

def load_data(kb_path):
    kb = JSONKnowledgeBase.load_json(kb_path)
    features, labels = [], []
    for case in kb.cases:
        # 假设 task_features 是一个扁平的向量
        if case.task_features and case.workflow_makespan > 0:
            # 将字典特征转换为列表
            if isinstance(case.task_features, dict):
                 features.append(list(case.task_features.values()))
            else:
                 features.append(case.task_features)
            labels.append(case.workflow_makespan)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)

def main():
    parser = argparse.ArgumentParser(description="从知识库训练性能预测器")
    parser.add_argument('config', help='预测器配置文件路径')
    parser.add_argument('--kb-path', required=True, help='知识库JSON文件路径')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    features, labels = load_data(args.kb_path)
    if len(features) == 0:
        print("[错误] 未从知识库中加载任何有效数据。")
        return

    input_dim = features.shape[1]
    print(f"数据加载完成。特征维度: {input_dim}, 样本数: {len(features)}")

    # [FIX] 使用统一的模型
    model = SimplePerformancePredictor(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [128, 64, 32])
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = torch.nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)

    print("开始训练...")
    for epoch in range(config.get('epochs', 50)):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.get('epochs', 50)}], Loss: {loss.item():.4f}")

    output_path = Path(config.get('output_model_path', 'models/performance_predictor.pth'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"训练完成。模型已保存到: {output_path}")

if __name__ == '__main__':
    main()