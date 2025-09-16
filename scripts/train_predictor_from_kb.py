import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path
import sys

# Add project path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.knowledge_base.json_kb import JSONKnowledgeBase
from src.shared_models import SimplePerformancePredictor

def load_data(kb_path, num_hosts=4):
    """Loads and prepares data from the knowledge base."""
    kb = JSONKnowledgeBase.load_json(kb_path)
    features, labels = [], []
    for case in kb.cases:
        if case.task_features and case.workflow_makespan > 0:
            task_feats = list(case.task_features.values()) if isinstance(case.task_features, dict) else case.task_features
            
            try:
                node_index = int(case.chosen_node.split('_')[-1])
                action_one_hot = np.zeros(num_hosts)
                if 0 <= node_index < num_hosts:
                    action_one_hot[node_index] = 1.0
                
                combined_features = np.concatenate([np.array(task_feats), action_one_hot])
                features.append(combined_features)
                labels.append(case.workflow_makespan)
            except (ValueError, IndexError):
                continue

    # [FIX] Ensure both numpy arrays are created with float32 dtype
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)

def main():
    parser = argparse.ArgumentParser(description="从知识库训练性能预测器")
    parser.add_argument('config', help='预测器配置文件路径')
    parser.add_argument('--kb-path', required=True, help='知识库JSON文件路径')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    num_hosts = config.get('num_hosts', 4)

    features, labels = load_data(args.kb_path, num_hosts=num_hosts)
    if len(features) == 0:
        print("[错误] 未从知识库中加载任何有效数据。"); return

    input_dim = features.shape[1]
    print(f"数据加载完成。特征维度(任务+动作): {input_dim}, 样本数: {len(features)}")

    model = SimplePerformancePredictor(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [128, 64, 32])
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = torch.nn.MSELoss()
    
    # PyTorch Tensors will now correctly be float32
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)

    print("开始训练导师模型...")
    epochs = config.get('epochs', 50)
    for epoch in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    output_path = Path(config.get('output_model_path', 'models/performance_predictor.pth'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"训练完成。模型已保存到: {output_path}")

if __name__ == '__main__':
    main()