#!/usr/bin/env python3
"""
WASS-RAG 阶段三：DRL 代理训练脚本 (语法修正版)

该脚本实现了论文中描述的 DRL 训练循环。它包含：
1. 一个自定义的 Gym 环境 (WassEnv)，将我们的调度问题包装起来。
2. 一个奖励函数，利用阶段二训练好的 PerformancePredictor 作为“教师”来提供奖励。
3. 使用 Stable-Baselines3 库中的 PPO 算法来训练 PolicyNetwork。
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 项目路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# --- 导入我们自己的模块 ---
from experiments.real_experiment_framework import WassExperimentRunner, ExperimentConfig
from src.ai_schedulers import WASSRAGScheduler, SchedulingState, PolicyNetwork

class WassEnv(gym.Env):
    """
    一个将 WASS 调度问题包装为与 Stable-Baselines3 兼容的自定义 Gym 环境。
    """
    metadata = {"render_modes": []}

    def __init__(self, config_dict):
        super().__init__()
        
        # 1. 初始化仿真器和“教师”模型
        sim_config = ExperimentConfig(
            name="DRL_Training_Env",
            workflow_sizes=config_dict["workflow_sizes"],
            scheduling_methods=[], # 在 DRL 训练中不需要
            cluster_sizes=config_dict["cluster_sizes"],
            repetitions=1, # 在 DRL 训练中不需要
            output_dir="temp_drl_env_results",
            ai_model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        self.sim_runner = WassExperimentRunner(sim_config)
        self.teacher_model = WASSRAGScheduler(model_path="models/wass_models.pth")
        
        # 2. 定义动作空间和观察空间
        self.max_nodes = max(config_dict["cluster_sizes"])
        self.action_space = spaces.Discrete(self.max_nodes)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
        
        self.current_simulation_config = None

    def reset(self, seed=None, options=None):
        """开始一轮新的仿真（一个 episode）"""
        super().reset(seed=seed)

        task_count = np.random.choice(self.sim_runner.config.workflow_sizes)
        cluster_size = np.random.choice(self.sim_runner.config.cluster_sizes)
        
        self.workflow, self.cluster = self.sim_runner._generate_scenario(task_count, cluster_size, self.np_random.integers(0, 1e9))
        self.nodes = list(self.cluster.keys())
        
        self.pending_tasks = {t['id'] for t in self.workflow['tasks']}
        self.task_finish_times = {}
        self.node_available_times = {node: 0.0 for node in self.nodes}
        self.task_placements = {}
        
        observation, info = self._get_next_observation()
        return observation, info

    def step(self, action):
        """执行一个动作并推进环境"""
        if action >= len(self.nodes):
            return self._get_next_observation()[0], -100.0, True, False, {"error": "Invalid action, node index out of bounds"}
        
        chosen_node = self.nodes[action]

        action_embedding = self.teacher_model._encode_action(chosen_node, self.current_state)
        predicted_finish_time = self.teacher_model._predict_performance(self.current_state_embedding, action_embedding, {})
        
        reward = 10.0 / (predicted_finish_time + 1e-6) # 放大奖励信号

        task_to_schedule = self.current_task_obj
        est = self.current_state.cluster_state['earliest_start_times'][chosen_node]
        exec_time = task_to_schedule['flops'] / (self.cluster[chosen_node]['cpu_capacity'] * 1e9)
        finish_time = est + exec_time
        
        self.task_finish_times[task_to_schedule['id']] = finish_time
        self.node_available_times[chosen_node] = finish_time
        self.task_placements[task_to_schedule['id']] = chosen_node
        self.pending_tasks.remove(task_to_schedule['id'])

        observation, info = self._get_next_observation()
        
        terminated = len(self.pending_tasks) == 0
        truncated = False
        
        return observation, reward, terminated, truncated, info

    def _get_next_observation(self):
        """找到下一个就绪的任务并为其构建观察向量"""
        if not self.pending_tasks:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"is_success": True}

        # --- 语法修正处 ---
        # 将之前的列表推导式改为标准的 for 循环以避免语法错误
        ready_tasks = []
        for task_id in sorted(list(self.pending_tasks)):
            task = next(t for t in self.workflow['tasks'] if t['id'] == task_id)
            if all(dep in self.task_finish_times for dep in task.get('dependencies', [])):
                ready_tasks.append(task)
        # --- 修正结束 ---
        
        if not ready_tasks:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"is_success": True, "reason": "No more ready tasks, workflow complete."}

        self.current_task_obj = ready_tasks[0]
        
        current_time = min(self.node_available_times.values())
        earliest_start_times = {}
        for node in self.nodes:
            deps = self.current_task_obj.get('dependencies', [])
            data_ready = max([self.task_finish_times.get(d, 0) for d in deps], default=0)
            earliest_start_times[node] = max(self.node_available_times[node], data_ready)

        self.current_state = SchedulingState(
            workflow_graph=self.workflow,
            cluster_state={"nodes": self.cluster, "earliest_start_times": earliest_start_times},
            pending_tasks=list(self.pending_tasks),
            current_task=self.current_task_obj['id'],
            available_nodes=self.nodes,
            timestamp=current_time
        )
        
        self.current_state_embedding = self.teacher_model._extract_simple_features_fallback(self.current_state)
        
        obs = np.concatenate([
            self.current_state_embedding.cpu().numpy(),
            np.zeros(32)
        ])
        
        return obs.astype(np.float32), {}

def main():
    print("🚀 WASS-RAG Stage 3: DRL Agent Training 🚀")
    
    env_config = {
        "workflow_sizes": [10, 50, 100],
        "cluster_sizes": [4, 8, 16],
    }
    env = DummyVecEnv([lambda: WassEnv(env_config)])
    
    print("\n🕵️  (Skipping env checker for DummyVecEnv)")
    
    policy_kwargs = {"net_arch": [128, 128]}
    
    agent = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./drl_tensorboard_logs/"
    )

    total_timesteps = 100000
    print(f"\n🧠 Starting PPO training for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    print("\n✅ DRL training complete. Saving the policy network...")
    model_path = Path("models/wass_models.pth")
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print("   Found existing model file. Updating Policy Network weights.")
    except (FileNotFoundError, EOFError):
        checkpoint = {}
        print("   No existing model file found. Creating a new checkpoint.")
    
    trained_policy_state_dict = agent.policy.state_dict()
    
    policy_net = PolicyNetwork(state_dim=64, hidden_dim=128)
    new_state_dict = policy_net.state_dict()
    
    # 手动将 SB3 的权重映射到我们的自定义网络结构
    # 这个过程比较脆弱，依赖于层名
    key_mapping = {
        'mlp_extractor.policy_net.0.weight': 'network.0.weight',
        'mlp_extractor.policy_net.0.bias': 'network.0.bias',
        'mlp_extractor.policy_net.2.weight': 'network.2.weight',
        'mlp_extractor.policy_net.2.bias': 'network.2.bias',
        'action_net.weight': 'network.4.weight',
        'action_net.bias': 'network.4.bias',
    }
    
    for sb3_key, custom_key in key_mapping.items():
        if sb3_key in trained_policy_state_dict and custom_key in new_state_dict:
            new_state_dict[custom_key] = trained_policy_state_dict[sb3_key]
        else:
            print(f"Warning: Could not map key {sb3_key}")
    
    checkpoint["policy_network"] = new_state_dict
    
    torch.save(checkpoint, model_path)
    print(f"✅ Policy Network weights updated and saved to {model_path}")
    print("\n🎉 All three stages are complete! Your AI schedulers are now fully trained.")

if __name__ == "__main__":
    main()