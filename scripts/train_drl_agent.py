#!/usr/bin/env python3
"""
WASS-RAG 阶段三：DRL 代理训练脚本

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

# --- 项目路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# --- 导入我们自己的模块 ---
from experiments.real_experiment_framework import WassExperimentRunner
from src.ai_schedulers import WASSRAGScheduler, SchedulingState, PolicyNetwork

class WassEnv(gym.Env):
    """
    一个将 WASS 调度问题包装为与 Stable-Baselines3 兼容的自定义 Gym 环境。
    """
    metadata = {"render_modes": []}

    def __init__(self, config_dict):
        super().__init__()
        
        # 1. 初始化仿真器和“教师”模型
        self.sim_runner = WassExperimentRunner(config_dict)
        self.teacher_model = WASSRAGScheduler(model_path="models/wass_models.pth")
        
        # 2. 定义动作空间和观察空间
        # 动作：选择一个节点的索引。假设最多有32个节点。
        self.action_space = spaces.Discrete(32) 
        # 观察：状态特征 + 动作特征。假设 state=32, action=32 -> 64维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
        
        self.current_simulation_config = None

    def reset(self, seed=None, options=None):
        """开始一轮新的仿真（一个 episode）"""
        super().reset(seed=seed)

        # 随机选择一个场景配置
        task_count = np.random.choice(self.sim_runner.config.workflow_sizes)
        cluster_size = np.random.choice(self.sim_runner.config.cluster_sizes)
        
        # 创建工作流和集群
        self.workflow, self.cluster = self.sim_runner._generate_scenario(task_count, cluster_size, seed if seed is not None else int.from_bytes(os.urandom(4), 'little'))
        self.nodes = list(self.cluster.keys())
        
        # 重置仿真状态
        self.pending_tasks = {t['id'] for t in self.workflow['tasks']}
        self.task_finish_times = {}
        self.node_available_times = {node: 0.0 for node in self.nodes}
        self.task_placements = {}
        
        # 找到第一个要调度的任务
        observation, info = self._get_next_observation()
        return observation, info

    def step(self, action):
        """执行一个动作并推进环境"""
        # 1. 解析动作
        # 检查动作是否有效（选择的节点是否存在）
        if action >= len(self.nodes):
            # 无效动作，给予惩罚并结束
            return self._get_next_observation()[0], -100.0, True, False, {"error": "Invalid action"}
        
        chosen_node = self.nodes[action]

        # 2. 计算奖励（核心）
        # 使用“教师”模型来预测这个决策的好坏
        action_embedding = self.teacher_model._encode_action(chosen_node, self.current_state)
        predicted_finish_time = self.teacher_model._predict_performance(self.current_state_embedding, action_embedding, {})
        
        # 奖励函数：完成时间越短，奖励越高。我们使用 1/time 的形式。
        reward = 1.0 / (predicted_finish_time + 1e-6)

        # 3. 更新仿真状态
        task_to_schedule = self.current_task_obj
        est = self.current_state.cluster_state['earliest_start_times'][chosen_node]
        exec_time = task_to_schedule['flops'] / (self.cluster[chosen_node]['cpu_capacity'] * 1e9)
        finish_time = est + exec_time
        
        self.task_finish_times[task_to_schedule['id']] = finish_time
        self.node_available_times[chosen_node] = finish_time
        self.task_placements[task_to_schedule['id']] = chosen_node
        self.pending_tasks.remove(task_to_schedule['id'])

        # 4. 获取下一个观察
        observation, info = self._get_next_observation()
        
        # 5. 检查是否结束
        terminated = len(self.pending_tasks) == 0
        truncated = False # 我们不设置时间步截断
        
        return observation, reward, terminated, truncated, info

    def _get_next_observation(self):
        """找到下一个就绪的任务并为其构建观察向量"""
        if not self.pending_tasks:
            return np.zeros(self.observation_space.shape), {"is_success": True}

        # 找到下一个就绪的任务
        ready_tasks = [
            task for task_id in sorted(list(self.pending_tasks))
            if all(dep in self.task_finish_times for dep in (task := next(t for t in self.workflow['tasks'] if t['id'] == task_id))['dependencies'])
        ]
        
        if not ready_tasks:
            # 如果没有就绪任务，说明工作流结束或卡死
            return np.zeros(self.observation_space.shape), {"error": "No ready tasks"}

        self.current_task_obj = ready_tasks[0]
        
        # 构建当前状态
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
        
        # 生成状态特征
        self.current_state_embedding = self.teacher_model._extract_simple_features_fallback(self.current_state)
        
        # 对于PPO，观察通常是状态本身
        # 这里的观察空间被简化了，实际应用中可以更复杂
        # 为了与 PolicyNetwork 的输入匹配，我们用 state_embedding 和一个零向量拼接
        obs = np.concatenate([
            self.current_state_embedding.cpu().numpy(),
            np.zeros(32) # 占位符
        ])
        
        return obs.astype(np.float32), {}

def main():
    print("🚀 WASS-RAG Stage 3: DRL Agent Training 🚀")
    
    # 1. 创建环境
    env_config = {
        "workflow_sizes": [10, 50, 100],
        "cluster_sizes": [4, 8, 16],
        "repetitions": 1 # 在 DRL 训练中，这个参数意义不大
    }
    # Stable Baselines3 需要一个函数来创建环境
    env_fn = lambda: WassEnv(env_config)
    env = env_fn()
    
    # 检查自定义环境是否符合 Gym API
    print("\n🕵️  Checking custom environment...")
    check_env(env)
    print("✅ Environment check passed!")
    
    # 2. 定义 DRL 代理
    # 我们将使用一个现有的 PolicyNetwork 结构，但让 SB3 来训练它
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 128], # Actor network
            "vf": [128, 128]  # Critic network
        }
    }
    
    # PPO 是一个非常强大且稳定的算法
    agent = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1, # 打印训练过程
        tensorboard_log="./drl_tensorboard_logs/"
    )

    # 3. 开始训练
    # total_timesteps 可以根据需要调整，100000 是一个不错的起点
    total_timesteps = 100000
    print(f"\n🧠 Starting PPO training for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # 4. 保存训练好的策略网络
    print("\n✅ DRL training complete. Saving the policy network...")
    model_path = Path("models/wass_models.pth")
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print("   Found existing model file. Updating Policy Network weights.")
    except (FileNotFoundError, EOFError):
        checkpoint = {}
        print("   No existing model file found. Creating a new checkpoint.")
    
    # 从 SB3 代理中提取策略网络的状态字典
    trained_policy_state_dict = agent.policy.state_dict()
    
    # 注意：这里的结构需要与 ai_schedulers.py 中的 PolicyNetwork 匹配
    # SB3 的 MlpPolicy 结构更复杂，直接保存可能不兼容
    # 简化：我们只保存 actor 网络的部分权重
    # 在实际项目中，需要确保网络结构完全一致
    # 这里我们做一个映射
    policy_net = PolicyNetwork(state_dim=64, hidden_dim=128)
    new_state_dict = policy_net.state_dict()
    
    # 简单的权重映射（可能需要根据实际层名调整）
    new_state_dict['network.0.weight'] = trained_policy_state_dict['mlp_extractor.policy_net.0.weight']
    new_state_dict['network.0.bias'] = trained_policy_state_dict['mlp_extractor.policy_net.0.bias']
    new_state_dict['network.2.weight'] = trained_policy_state_dict['mlp_extractor.policy_net.2.weight']
    new_state_dict['network.2.bias'] = trained_policy_state_dict['mlp_extractor.policy_net.2.bias']
    # 最后一层
    new_state_dict['network.4.weight'] = trained_policy_state_dict['action_net.weight']
    new_state_dict['network.4.bias'] = trained_policy_state_dict['action_net.bias']
    
    checkpoint["policy_network"] = new_state_dict
    
    torch.save(checkpoint, model_path)
    print(f"✅ Policy Network weights updated and saved to {model_path}")
    print("\n🎉 All three stages are complete! Your AI schedulers are now fully trained.")


if __name__ == "__main__":
    main()