#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从知识库训练性能预测器的脚本
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import logging

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.performance_predictor import PerformancePredictor, SimplePerformancePredictor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformancePredictorTrainer:
    """性能预测器训练器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 训练配置
        self.training_cfg = self.config.get('training', {})
        self.model_cfg = self.config.get('model', {})
        
        # 设置随机种子
        seed = self.training_cfg.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = None
        self.simple_model = None
        
    def load_training_data(self, kb_path: str = None) -> List[Dict]:
        """加载训练数据"""
        if kb_path is None:
            kb_path = self.config.get('knowledge_base', {}).get('path', 'data/kb_training_dataset.json')

        logger.info(f"加载训练数据: {kb_path}")

        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"知识库文件不存在: {kb_path}")

        with open(kb_path, 'r') as f:
            kb_data = json.load(f)

        # 检查数据是字典还是列表
        if isinstance(kb_data, dict) and 'cases' in kb_data:
            data = kb_data['cases']
        else:
            data = kb_data

        # 检查并解析字符串格式的JSON对象
        if data and isinstance(data[0], str):
            logger.info("数据样本是字符串格式，将进行JSON解析...")
            try:
                data = [json.loads(s) for s in data]
            except json.JSONDecodeError as e:
                logger.error(f"解析JSON字符串时出错: {e}")
                raise

        logger.info(f"加载了 {len(data)} 个训练样本")
        return data
    
    def prepare_features(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和标签"""
        logger.info("准备特征和标签...")
        
        features = []
        labels = []
        
        for sample in data:
            # 提取特征
            task_features = sample.get('task_features', {})
            platform_features = sample.get('platform_features', {})
            
            feature_vector = [
                task_features.get('task_flops', 0),
                task_features.get('task_memory', 0),
                task_features.get('task_inputs', 0),
                task_features.get('task_outputs', 0),
                task_features.get('task_dependencies', 0),
                platform_features.get('num_nodes', 0),
                platform_features.get('avg_flops', 0),
                platform_features.get('avg_memory', 0),
            ]
            
            features.append(feature_vector)
            labels.append(sample.get('makespan', 0))
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # 过滤掉makespan为0的异常数据
        if len(labels) > 0:
            valid_indices = labels > 1e-6  # Use a small epsilon for float comparison
            features = features[valid_indices]
            labels = labels[valid_indices]

        logger.info(f"有效样本数: {len(features)}")
        logger.info(f"特征形状: {features.shape}")
        logger.info(f"标签形状: {labels.shape}")
        
        return features, labels
    
    def train_simple_predictor(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """训练简单性能预测器"""
        logger.info("训练简单性能预测器...")

        if len(features) == 0:
            logger.warning("没有有效的训练数据，跳过训练。")
            return {
                'r2': 0.0,
                'rmse': 0.0,
                'scaler': None
            }
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # 初始化模型
        input_dim = X_train.shape[1]
        self.simple_model = SimplePerformancePredictor(input_dim=input_dim).to(self.device)
        
        # 损失函数和优化器
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.simple_model.parameters(), lr=0.001)
        
        # 训练参数
        epochs = self.training_cfg.get('epochs', 100)
        batch_size = self.training_cfg.get('batch_size', 32)
        
        # 训练循环
        self.simple_model.train()
        for epoch in range(epochs):
            # 批次训练
            permutation = torch.randperm(X_train_tensor.size(0))
            total_loss = 0
            num_batches = 0
            
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.simple_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # 验证
        self.simple_model.eval()
        with torch.no_grad():
            y_pred = self.simple_model(X_test_tensor).cpu().numpy().flatten()
            y_true = y_test_tensor.cpu().numpy().flatten()
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            logger.info(f"验证集 R²: {r2:.4f}")
            logger.info(f"验证集 RMSE: {rmse:.4f}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'scaler': scaler
        }
    
    def save_model(self, output_path: str = "models/wass_models.pth"):
        """保存训练好的模型"""
        logger.info(f"保存模型到: {output_path}")
        
        # 创建目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据
        checkpoint = {
            'performance_predictor': self.simple_model.state_dict() if self.simple_model else None,
            'metadata': {
                'performance_predictor': {
                    'validation_results': {
                        'r2': 0.0,  # 这将在训练后更新
                        'rmse': 0.0
                    }
                }
            }
        }
        
        torch.save(checkpoint, output_path)
        logger.info("模型保存完成")
    
    def train(self, kb_path: str = None) -> Dict:
        """完整的训练流程"""
        logger.info("开始训练性能预测器...")
        
        # 加载数据
        data = self.load_training_data(kb_path)
        
        # 准备特征
        features, labels = self.prepare_features(data)
        
        # 训练简单预测器
        validation_results = self.train_simple_predictor(features, labels)
        
        # 保存模型
        model_output_path = self.config.get('model', {}).get('output_path', 'models/wass_models.pth')
        self.save_model(model_output_path)
        
        # 更新验证结果
        checkpoint = torch.load(model_output_path, map_location='cpu', weights_only=False)
        checkpoint['metadata']['performance_predictor']['validation_results'] = validation_results
        torch.save(checkpoint, model_output_path)
        
        logger.info("性能预测器训练完成!")
        return validation_results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练性能预测器')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--kb-path', help='知识库路径')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = PerformancePredictorTrainer(args.config)
    
    # 训练
    try:
        results = trainer.train(args.kb_path)
        print(f"训练完成! 验证R²: {results['r2']:.4f}")
        return 0
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())