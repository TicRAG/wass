#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG DRL Trainer - Final Corrected Version
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import yaml

# Add project path to import local modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.knowledge_base.json_kb import JSONKnowledgeBase
# [FIX] Import the unified model definition from the shared module
from src.shared_models import SimplePerformancePredictor
from src.drl.reward import compute_final_reward, EpisodeStats

# --- Dataclass Definitions ---
@dataclass
class TaskState: id: str; computation_size: float; parents: List[str]; children: List[str]; is_critical_path: bool; data_locality_score: float; data_size: float = 0.0
@dataclass
class NodeState: id: str; speed: float; current_load: float; available_time: float; data_availability: Dict[str, float]
@dataclass
class EnvironmentState: current_time: float; pending_tasks: List[TaskState]; node_states: List[NodeState]; workflow_progress: float; critical_path_length: float

# --- DRL Agent Definition ---
class DQNAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [256, 128, 64]
        layers, c_dim = [], state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(c_dim, h_dim), nn.ReLU(), nn.Dropout(0.1)])
            c_dim = h_dim
        layers.append(nn.Linear(c_dim, action_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

class DRLAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma, self.batch_size = config.get('gamma', 0.99), config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.epsilon, self.epsilon_end, self.epsilon_decay = config.get('epsilon_start', 1.0), config.get('epsilon_end', 0.1), config.get('epsilon_decay', 0.995)
        h_dims = config.get('hidden_dims', [256, 128, 64])
        self.q_network = DQNAgent(state_dim, action_dim, h_dims).to(self.device)
        self.target_network = DQNAgent(state_dim, action_dim, h_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('learning_rate', 0.001))
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        self.training_step = 0; self.update_target()
    def update_target(self): self.target_network.load_state_dict(self.q_network.state_dict())
    def remember(self, s, a, r, s_next, d): self.memory.append(self.experience(s,a,r,s_next,d))
    def act(self, state, training=True):
        if not training or random.random() > self.epsilon:
            with torch.no_grad():
                q_vals = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                return q_vals.argmax().item()
        return random.randint(0, self.action_dim - 1)
    def replay(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = (torch.FloatTensor(np.array(t)).to(self.device) for t in zip(*batch))
        actions, rewards, dones = actions.long(), rewards, dones.bool()
        curr_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        loss = nn.SmoothL1Loss()(curr_q.squeeze(), target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > self.epsilon_end: self.epsilon *= self.epsilon_decay
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0: self.update_target()

# --- Main Trainer ---
class WRENCHDRLTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        self.drl_cfg, self.ckpt_cfg, self.log_cfg, rag_cfg, pred_cfg = (self.config.get(k, {}) for k in ['drl', 'checkpoint', 'logging', 'rag', 'predictor'])
        
        kb_path = rag_cfg.get('knowledge_base', {}).get('path', 'data/real_heuristic_kb.json')
        self.knowledge_base = JSONKnowledgeBase.load_json(kb_path)
        print(f"📚 Knowledge Base loaded: {kb_path} ({len(self.knowledge_base.cases)} cases)")
        
        predictor_model_path = pred_cfg.get('model_path', 'models/performance_predictor.pth')
        
        # --- Final Fix 1: Dynamically infer input dim and load the unified predictor ---
        try:
            state_dict = torch.load(predictor_model_path, map_location="cpu", weights_only=False)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            
            # Infer input dimension from the first layer's weights
            first_layer_key = next(iter(state_dict)) 
            predictor_input_dim = state_dict[first_layer_key].shape[1]
            print(f"✅ Predictor input dimension dynamically inferred: {predictor_input_dim}")

            self.predictor = SimplePerformancePredictor(input_dim=predictor_input_dim)
            self.predictor.load_state_dict(state_dict)
            self.predictor.eval()
            print(f"🧠 Performance Predictor loaded successfully: {predictor_model_path}")

        except Exception as e:
            print(f"[FATAL] Could not load or parse the performance predictor model: {e}")
            raise e

        # --- Final Fix 2: Provide default curriculum to prevent IndexError ---
        self.curriculum_stages = self.drl_cfg.get('curriculum_learning', {}).get('stages')
        if not self.curriculum_stages or not isinstance(self.curriculum_stages, list):
            print("[WARNING] Curriculum learning config not found or invalid. Using default.")
            self.curriculum_stages = [
                {"name": "Stage1", "tasks": 10, "nodes": 4, "complexity": 0.3, "episodes": 200},
                {"name": "Stage2", "tasks": 20, "nodes": 4, "complexity": 0.6, "episodes": 300},
            ]
        self.current_stage = 0
        self.agent, self.best_makespan = None, float('inf')

    # Placeholder methods for brevity, use your actual implementation
    def _create_mock_env(self, stage_cfg):
        num_tasks, num_nodes = stage_cfg["tasks"], stage_cfg["nodes"]
        nodes = [NodeState(id=f"H{i}", speed=2+i, current_load=0, available_time=0, data_availability={}) for i in range(num_nodes)]
        tasks = [TaskState(id=f"T{i}", computation_size=1e9, parents=[], children=[], is_critical_path=False, data_locality_score=1.0) for i in range(num_tasks)]
        return EnvironmentState(current_time=0.0, pending_tasks=tasks, node_states=nodes, workflow_progress=0.0, critical_path_length=100.0), tasks, nodes
    def _extract_state(self, task, nodes, env): return np.random.rand(48) # Placeholder
    def _get_rag_reward(self, state, action): return random.random() # Placeholder
    def _simulate_step(self, task, action, env): return env, len(env.pending_tasks) == 1 # Placeholder

    def train_episode(self):
        stage_cfg = self.curriculum_stages[self.current_stage]
        env, tasks, _ = self._create_mock_env(stage_cfg)
        current_tasks = tasks[:]
        for task in current_tasks:
            state = self._extract_state(task, env.node_states, env)
            action = self.agent.act(state)
            reward = self._get_rag_reward(state, action)
            env, done = self._simulate_step(task, action, env)
            next_state = self._extract_state(task, env.node_states, env) if not done else np.zeros_like(state)
            self.agent.remember(state, action, reward, next_state, done)
            self.agent.replay()
            if done: break
        return {'makespan': env.current_time, 'epsilon': self.agent.epsilon}

    def train(self, episodes: int = 1000):
        episodes = self.drl_cfg.get('episodes', episodes)
        print(f"🚀 Starting DRL training for {episodes} episodes...")
        
        stage_cfg = self.curriculum_stages[self.current_stage]
        env, tasks, nodes = self._create_mock_env(stage_cfg)
        state_dim = len(self._extract_state(tasks[0], nodes, env))
        action_dim = stage_cfg["nodes"]
        
        agent_cfg = self.drl_cfg.get('agent_settings', {})
        self.agent = DRLAgent(state_dim, action_dim, agent_cfg)
        
        for ep in range(episodes):
            results = self.train_episode()
            if (ep + 1) % self.log_cfg.get('log_interval', 50) == 0:
                 print(f"Episode {ep+1}: Makespan={results['makespan']:.2f}, Epsilon={results['epsilon']:.3f}")
            # Curriculum stage transition logic
            if self.current_stage < len(self.curriculum_stages) - 1:
                if (ep + 1) >= self.curriculum_stages[self.current_stage].get('episodes', 200):
                    self.current_stage += 1
                    print(f"🎓 Advancing to next stage: {self.curriculum_stages[self.current_stage]['name']}")
        
        self.save_model(self.ckpt_cfg.get('final_model_path', 'models/improved_wass_drl.pth'))

    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'q_network_state_dict': self.agent.q_network.state_dict()}, path)
        print(f"📁 Model saved to: {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='WASS-RAG DRL Trainer')
    parser.add_argument('--config', default='configs/drl.yaml', help='Path to the DRL config file')
    args = parser.parse_args()
    trainer = WRENCHDRLTrainer(args.config)
    trainer.train()
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()