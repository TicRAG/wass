#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
from typing import Dict, List, Any

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.shared_models import SimplePerformancePredictor
from src.knowledge_base.json_kb import JSONKnowledgeBase
from src.drl_agent import DQNAgent

class WASSOfflineTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        self.drl_cfg, self.ckpt_cfg, self.log_cfg, rag_cfg, pred_cfg = (self.config.get(k, {}) for k in ['drl', 'checkpoint', 'logging', 'rag', 'predictor'])
        
        predictor_model_path = pred_cfg.get('model_path', 'models/performance_predictor.pth')
        try:
            state_dict = torch.load(predictor_model_path, map_location="cpu", weights_only=False)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            first_layer_key = next(iter(state_dict))
            self.predictor_input_dim = state_dict[first_layer_key].shape[1]
            print(f"✅ 导师输入维度动态识别: {self.predictor_input_dim}")
            self.tutor = SimplePerformancePredictor(input_dim=self.predictor_input_dim)
            self.tutor.load_state_dict(state_dict)
            self.tutor.eval()
            print(f"🧠 导师 (性能预测器) 加载成功: {predictor_model_path}")
        except Exception as e:
            print(f"[致命错误] 无法加载导师模型: {e}"); raise e
            
        kb_path = rag_cfg.get('knowledge_base', {}).get('path', 'data/real_heuristic_kb.json')
        self.knowledge_base = JSONKnowledgeBase.load_json(kb_path)
        if not self.knowledge_base.cases: raise ValueError("知识库为空，无法训练。")
        print(f"📚 用于训练的真实知识库加载成功: {kb_path} ({len(self.knowledge_base.cases)} cases)")
        self.num_hosts = self.drl_cfg.get('num_hosts', 4)

    def _extract_features_from_case(self, case):
        features = case.task_features
        if isinstance(features, dict): features = list(features.values())
        return np.array(features, dtype=np.float32)

    def train(self):
        episodes = self.drl_cfg.get('episodes', 10) # With real data, fewer epochs are needed
        episodes = 20
        print(f"🚀 开始从知识库进行离线DRL训练，共 {episodes} 轮...")

        sample_state = self._extract_features_from_case(self.knowledge_base.cases[0])
        state_dim = len(sample_state)
        action_dim = self.num_hosts
        
        agent_cfg = self.drl_cfg.get('agent_settings', {})
        learner = DQNAgent(state_dim, action_dim, agent_cfg.get('hidden_dims'))
        optimizer = optim.Adam(learner.parameters(), lr=agent_cfg.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        learner.train()

        training_cases = self.knowledge_base.cases
        for ep in range(episodes):
            total_loss = 0.0
            random.shuffle(training_cases)
            
            for case in training_cases:
                state = self._extract_features_from_case(case)
                if state.shape[0] != state_dim: continue
                
                with torch.no_grad():
                    tutor_scores = []
                    for action in range(self.num_hosts):
                        action_one_hot = np.zeros(self.num_hosts); action_one_hot[action] = 1.0
                        
                        # [FIX] Construct the input for the tutor EXACTLY as it was trained
                        tutor_input = np.concatenate([state, action_one_hot]).astype(np.float32)
                        if tutor_input.shape[0] != self.predictor_input_dim:
                             print(f"维度不匹配！导师期望 {self.predictor_input_dim}, 得到 {tutor_input.shape[0]}")
                             continue

                        predicted_makespan = self.tutor.predict(tutor_input)
                        tutor_scores.append(predicted_makespan)

                if not tutor_scores: continue
                best_action = np.argmin(tutor_scores)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                target_action_tensor = torch.LongTensor([best_action])

                optimizer.zero_grad()
                action_logits = learner(state_tensor)
                loss = criterion(action_logits, target_action_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_cases)
            if (ep + 1) % 2 == 0 or ep == episodes - 1:
                print(f"轮次 {ep+1}/{episodes}, 平均损失: {avg_loss:.4f}")

        self.save_model(learner)

    def save_model(self, model: nn.Module):
        path = self.ckpt_cfg.get('final_model_path', 'models/improved_wass_drl.pth')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'q_network_state_dict': model.state_dict()}, path)
        print(f"📁 学习者模型已保存: {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='WASS-RAG DRL Offline Trainer')
    parser.add_argument('--config', default='configs/drl.yaml')
    args = parser.parse_args()
    trainer = WASSOfflineTrainer(args.config)
    trainer.train()
    print("🎉 离线训练完成!")

if __name__ == "__main__":
    main()