#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG DRL Trainer - Offline Training (Final Version)
This version uses the Tutor (Performance Predictor) to train the Learner (DRL Agent)
in an offline manner, without requiring an interactive WRENCH environment or
external workflow files during the training phase.
"""
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque, namedtuple
import random
from typing import Dict, List, Any

# Add project path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.shared_models import SimplePerformancePredictor

# --- DRL Agent (Learner) Definition ---
class LearnerAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers, current_dim = [], state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, h_dim), nn.ReLU()])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Main Offline Trainer ---
class WASSOfflineTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.drl_cfg = self.config.get('drl', {})
        self.ckpt_cfg = self.config.get('checkpoint', {})
        self.log_cfg = self.config.get('logging', {})
        pred_cfg = self.config.get('predictor', {})

        # --- Load the Tutor (Performance Predictor) ---
        predictor_model_path = pred_cfg.get('model_path', 'models/performance_predictor.pth')
        try:
            state_dict = torch.load(predictor_model_path, map_location="cpu", weights_only=False)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            first_layer_key = next(iter(state_dict))
            self.predictor_input_dim = state_dict[first_layer_key].shape[1]
            print(f"✅ Tutor's input dimension dynamically inferred: {self.predictor_input_dim}")

            self.tutor = SimplePerformancePredictor(input_dim=self.predictor_input_dim)
            self.tutor.load_state_dict(state_dict)
            self.tutor.eval()
            print(f"🧠 Tutor (Performance Predictor) loaded successfully: {predictor_model_path}")
        except Exception as e:
            print(f"[FATAL] Could not load the Tutor model: {e}")
            raise e
            
        # --- Training Parameters ---
        self.num_hosts = self.drl_cfg.get('num_hosts', 4)
        self.state_dim = self.drl_cfg.get('state_dim', 5) # Dimension of state features

    def _generate_random_state(self) -> np.ndarray:
        """Generates a random state vector to represent a synthetic task."""
        # This simulates the features of a task that needs scheduling.
        # The dimensions and value ranges should ideally match the data
        # the Tutor was trained on.
        return np.random.rand(self.state_dim).astype(np.float32)

    def train(self):
        """Main offline training loop."""
        episodes = self.drl_cfg.get('episodes', 1000)
        steps_per_episode = self.drl_cfg.get('steps_per_episode', 50)
        print(f"🚀 Starting Offline DRL Training for {episodes} episodes...")

        # --- Initialize the Learner (DRL Agent) ---
        action_dim = self.num_hosts
        agent_cfg = self.drl_cfg.get('agent_settings', {})
        learner = LearnerAgent(self.state_dim, action_dim, agent_cfg.get('hidden_dims'))
        optimizer = optim.Adam(learner.parameters(), lr=agent_cfg.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        learner.train()

        # --- Training Loop ---
        for ep in range(episodes):
            total_loss = 0.0
            
            # In each episode, we generate a number of synthetic tasks to learn from
            for step in range(steps_per_episode):
                # 1. Create a synthetic task state
                state = self._generate_random_state()
                
                # 2. Ask the Tutor to evaluate every possible action for this state
                with torch.no_grad():
                    tutor_scores = []
                    for action in range(self.num_hosts):
                        action_one_hot = np.zeros(self.num_hosts); action_one_hot[action] = 1.0
                        
                        # Construct the input vector for the tutor
                        tutor_input = np.concatenate([state, action_one_hot]).astype(np.float32)
                        
                        # Pad or truncate the vector to match the tutor's expected input dimension
                        if len(tutor_input) < self.predictor_input_dim:
                            tutor_input = np.pad(tutor_input, (0, self.predictor_input_dim - len(tutor_input)))
                        else:
                            tutor_input = tutor_input[:self.predictor_input_dim]

                        predicted_makespan = self.tutor.predict(tutor_input)
                        tutor_scores.append(predicted_makespan)

                # 3. Identify the best action according to the Tutor
                best_action = np.argmin(tutor_scores)
                
                # 4. Train the Learner to imitate the Tutor's best action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                target_action_tensor = torch.LongTensor([best_action])

                optimizer.zero_grad()
                action_logits = learner(state_tensor)
                loss = criterion(action_logits, target_action_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / steps_per_episode
            if (ep + 1) % self.log_cfg.get('log_interval', 100) == 0:
                print(f"Episode {ep+1}/{episodes}, Average Loss: {avg_loss:.4f}")

        self.save_model(learner)

    def save_model(self, model: nn.Module):
        path = self.ckpt_cfg.get('final_model_path', 'models/improved_wass_drl.pth')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # The key must match what the evaluation script expects.
        # Based on logs, it expects 'q_network_state_dict'.
        torch.save({'q_network_state_dict': model.state_dict()}, path)
        print(f"📁 Learner model saved to: {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='WASS-RAG DRL Offline Trainer')
    parser.add_argument('--config', default='configs/drl.yaml', help='Path to the DRL config file')
    args = parser.parse_args()
    trainer = WASSOfflineTrainer(args.config)
    trainer.train()
    print("🎉 Offline Training complete!")

if __name__ == "__main__":
    main()