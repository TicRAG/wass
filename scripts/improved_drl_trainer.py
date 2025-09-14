#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 改进的DRL训练器
实现密集奖励函数以提高训练效果
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

# 先添加项目路径再导入本地包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.drl.reward import compute_step_reward, compute_final_reward, StepContext, EpisodeStats  # noqa: E402

# 确保结果目录存在（冒烟运行可能尚未创建）
Path('results').mkdir(exist_ok=True)

@dataclass
class TaskState:
    """任务状态"""
    id: str
    computation_size: float
    parents: List[str]
    children: List[str]
    is_critical_path: bool
    data_locality_score: float

@dataclass
class NodeState:
    """节点状态"""
    id: str
    speed: float
    current_load: float
    available_time: float
    data_availability: Dict[str, float]  # 数据可用性评分

@dataclass
class EnvironmentState:
    """环境状态"""
    current_time: float
    pending_tasks: List[TaskState]
    node_states: List[NodeState]
    workflow_progress: float
    critical_path_length: float

class DenseRewardCalculator:
    """(Deprecated soon) 保留旧接口以防兼容问题，但内部委托到新 shaping。"""
    def calculate_step_reward(self, task: TaskState, chosen_node: NodeState, all_nodes: List[NodeState], environment: EnvironmentState) -> float:
        # 旧接口保持但现在直接返回0（避免误用），真实奖励在训练循环外部构造 StepContext
        return 0.0
    def calculate_final_reward(self, final_makespan: float, baseline_makespan: float) -> float:
        stats = EpisodeStats(makespan=final_makespan)
        return compute_final_reward(stats)

class ImprovedDQN(nn.Module):
    """改进的DQN网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        current_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 使用Xavier初始化
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class ImprovedDQNAgent:
    """改进的DQN智能体"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.99,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 网络
        self.q_network = ImprovedDQN(state_dim, action_dim).to(self.device)
        self.target_network = ImprovedDQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
        
        # 探索策略
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 训练统计
        self.training_step = 0
        self.update_target()
    
    def update_target(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def act(self, state, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 采样经验
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # 计算Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # 定期更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target()
        
        return loss.item()

class WRENCHDRLTrainer:
    """WRENCH DRL训练器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.reward_calculator = DenseRewardCalculator()
        self.agent = None
        self.training_history = []
        # 新增：读取扩展训练/日志配置
        self.drl_cfg = self.config.get('drl', {})
        self.checkpoint_cfg = self.config.get('checkpoint', {})
        self.logging_cfg = self.config.get('logging', {})
        Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/')).mkdir(parents=True, exist_ok=True)
        Path(self.logging_cfg.get('metrics_file', 'results/training_metrics.jsonl')).parent.mkdir(parents=True, exist_ok=True)
        self.best_makespan = float('inf')
    
    def create_mock_environment(self) -> Tuple[EnvironmentState, List[TaskState], List[NodeState]]:
        """创建模拟训练环境"""
        # 创建节点状态
        node_states = []
        for i in range(4):
            node_states.append(NodeState(
                id=f"ComputeHost{i+1}",
                speed=2.0 + i * 0.5,  # 不同的处理速度
                current_load=random.uniform(0, 0.5),
                available_time=random.uniform(0, 10),
                data_availability={f"task_{j}": random.random() for j in range(20)}
            ))
        
        # 创建任务状态
        task_states = []
        for i in range(20):
            task_states.append(TaskState(
                id=f"task_{i}",
                computation_size=random.uniform(1e9, 1e10),
                parents=[f"task_{j}" for j in range(max(0, i-2), i)],
                children=[f"task_{j}" for j in range(i+1, min(20, i+3))],
                is_critical_path=random.random() > 0.7,
                data_locality_score=random.random()
            ))
        
        # 创建环境状态
        environment = EnvironmentState(
            current_time=0.0,
            pending_tasks=task_states,
            node_states=node_states,
            workflow_progress=0.0,
            critical_path_length=100.0
        )
        
        return environment, task_states, node_states
    
    def extract_state_features(self, 
                             current_task: TaskState, 
                             node_states: List[NodeState],
                             environment: EnvironmentState) -> np.ndarray:
        """提取状态特征"""
        features = []
        
        # 任务特征
        features.extend([
            current_task.computation_size / 1e10,  # 归一化
            len(current_task.parents),
            len(current_task.children),
            float(current_task.is_critical_path),
            current_task.data_locality_score
        ])
        
        # 节点特征
        for node in node_states:
            features.extend([
                node.speed / 5.0,  # 归一化
                node.current_load,
                node.available_time / 100.0,  # 归一化
                node.data_availability.get(current_task.id, 0.0)
            ])
        
        # 环境特征
        features.extend([
            environment.workflow_progress,
            environment.current_time / 1000.0,  # 归一化
            len(environment.pending_tasks) / 20.0  # 归一化
        ])
        
        return np.array(features, dtype=np.float32)
    
    def simulate_step(self, 
                     task: TaskState, 
                     action: int, 
                     node_states: List[NodeState],
                     environment: EnvironmentState) -> Tuple[float, EnvironmentState, bool]:
        """模拟一步执行"""
        chosen_node = node_states[action]
        
        # 计算执行时间
        execution_time = task.computation_size / chosen_node.speed
        # 奖励不在此计算（外部根据更新后的全局信息构造 StepContext）
        # 更新环境
        new_environment = EnvironmentState(
            current_time=environment.current_time + execution_time,
            pending_tasks=[t for t in environment.pending_tasks if t.id != task.id],
            node_states=node_states,
            workflow_progress=environment.workflow_progress + 1.0/20.0,
            critical_path_length=environment.critical_path_length
        )
        
        # 更新节点状态
        chosen_node.current_load += 0.1
        chosen_node.available_time += execution_time
        
        # 检查是否结束
        done = len(new_environment.pending_tasks) == 0
        return 0.0, new_environment, done
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个episode"""
        environment, task_states, node_states = self.create_mock_environment()
        
        total_reward = 0.0
        total_makespan = 0.0
        step_count = 0
        
        current_tasks = task_states.copy()

        reward_debug_path = self.logging_cfg.get('reward_debug', 'results/reward_debug.log')
        debug_file = None
        try:
            debug_file = open(reward_debug_path, 'a')
        except Exception:
            debug_file = None

        # 准备增强型 StepContext 统计
        total_cp_tasks = sum(1 for t in task_states if t.is_critical_path) or 1
        baseline_avg_wait = np.mean([n.available_time for n in node_states]) or 1.0
        completed_ids = set()
        task_map = {t.id: t for t in task_states}

        while current_tasks and step_count < 50:  # 限制最大步数
            # 选择当前任务（简化：按顺序）
            current_task = current_tasks[0]
            
            # 提取状态特征
            state = self.extract_state_features(current_task, node_states, environment)
            
            # 智能体选择动作
            action = self.agent.act(state, training=True)
            
            # 执行动作
            _, new_environment, done = self.simulate_step(
                current_task, action, node_states, environment
            )
            # 更新完成集
            completed_ids.add(current_task.id)

            # 构造增强 StepContext
            try:
                completed_cp = sum(1 for cid in completed_ids if task_map[cid].is_critical_path)
                ctx = StepContext(
                    completed_critical_path_tasks=completed_cp,
                    total_critical_path_tasks=total_cp_tasks,
                    node_busy_times={n.id: n.current_load for n in node_states},
                    ready_task_count=len(current_tasks)-1,  # 去掉当前即将调度的
                    total_nodes=len(node_states),
                    avg_queue_wait=np.mean([n.available_time for n in node_states]),
                    queue_wait_baseline=baseline_avg_wait
                )
                step_reward, _metrics = compute_step_reward(ctx, debug_writer=debug_file)
            except Exception:
                step_reward = 0.0
            
            # 提取下一状态特征
            if len(current_tasks) > 1:
                next_task = current_tasks[1]
                next_state = self.extract_state_features(next_task, node_states, new_environment)
            else:
                next_state = np.zeros_like(state)
            
            # 存储经验
            self.agent.remember(state, action, step_reward, next_state, done)
            
            # 更新状态
            environment = new_environment
            current_tasks = current_tasks[1:]
            total_reward += step_reward
            total_makespan = environment.current_time
            step_count += 1
            
            # 训练智能体
            if step_count % 4 == 0:  # 每4步训练一次
                loss = self.agent.replay()
            
            if done:
                break

        # 计算最终奖励 (新 makespan 稀疏奖励)
        final_reward = compute_final_reward(EpisodeStats(makespan=total_makespan))
        total_reward += final_reward
        # 可选：将最终奖励写入最后一个 transition（可在未来扩展）

        if debug_file:
            try:
                debug_file.write(f"FINAL\tmakespan={total_makespan:.4f}\treward={final_reward:.4f}\n")
                debug_file.close()
            except Exception:
                pass

        return {
            'total_reward': total_reward,
            'makespan': total_makespan,
            'step_count': step_count,
            'epsilon': self.agent.epsilon
        }
    
    def train(self, episodes: int = 1000) -> Dict[str, Any]:
        """训练DRL智能体"""
        print(f"🚀 开始改进的DRL训练: {episodes} episodes (配置 episodes={self.drl_cfg.get('episodes', episodes)})")
        
        # 初始化智能体
        state_dim = 5 + 4 * 4 + 3  # 任务特征 + 节点特征 + 环境特征
        action_dim = 4  # 4个节点
        
        self.agent = ImprovedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **self.config.get('drl', {})
        )
        
        # 训练循环
        best_makespan = float('inf')
        recent_rewards = deque(maxlen=100)
        
        log_interval = self.drl_cfg.get('log_interval', 50)
        eval_interval = self.drl_cfg.get('eval_interval', 100)
        checkpoint_interval = self.drl_cfg.get('checkpoint_interval', 100)
        rolling_window = self.drl_cfg.get('rolling_window', 100)
        metrics_path = self.logging_cfg.get('metrics_file', 'results/training_metrics.jsonl')
        ckpt_dir = Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/'))
        keep_last = self.checkpoint_cfg.get('keep_last', 5)
        save_best = self.checkpoint_cfg.get('save_best', True)
        kept_ckpts = []

        for episode in range(episodes):
            episode_results = self.train_episode()
            self.training_history.append(episode_results)
            recent_rewards.append(episode_results['total_reward'])
            
            # 更新最佳性能
            if episode_results['makespan'] < best_makespan:
                best_makespan = episode_results['makespan']
            
            # 打印进度
            if episode % log_interval == 0:
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Episode {episode}: "
                      f"平均奖励={avg_reward:.3f}, "
                      f"Makespan={episode_results['makespan']:.2f}, "
                      f"ε={episode_results['epsilon']:.3f}")
            # 写入流式指标日志
            try:
                with open(metrics_path, 'a') as mf:
                    mf.write(json.dumps({
                        'episode': episode,
                        'reward': episode_results['total_reward'],
                        'makespan': episode_results['makespan'],
                        'epsilon': episode_results['epsilon'],
                        'timestamp': time.time()
                    }) + '\n')
            except Exception as e:
                print(f"⚠️ 写入指标日志失败: {e}")

            # 检查点保存
            if checkpoint_interval and episode % checkpoint_interval == 0 and episode > 0:
                ckpt_path = ckpt_dir / f"episode_{episode}.pth"
                self.save_model(str(ckpt_path))
                kept_ckpts.append(ckpt_path)
                # 控制数量
                if len(kept_ckpts) > keep_last:
                    old = kept_ckpts.pop(0)
                    try:
                        old.unlink()
                    except Exception:
                        pass
            # 保存最佳
            if save_best and episode_results['makespan'] < self.best_makespan:
                self.best_makespan = episode_results['makespan']
                best_path = ckpt_dir / 'best_model.pth'
                self.save_model(str(best_path))
            # 评估钩子占位
            if eval_interval and episode % eval_interval == 0 and episode > 0:
                pass  # 可在此接入验证环境
        
        # 训练完成统计
        final_avg_reward = np.mean([h['total_reward'] for h in self.training_history[-100:]])
        final_avg_makespan = np.mean([h['makespan'] for h in self.training_history[-100:]])
        
        training_summary = {
            'total_episodes': episodes,
            'final_avg_reward': final_avg_reward,
            'final_avg_makespan': final_avg_makespan,
            'best_makespan': best_makespan,
            'training_history': self.training_history
        }
        print(f"✅ DRL训练完成! (新奖励框架集成)")
        print(f"   最终平均奖励: {final_avg_reward:.3f}")
        print(f"   最终平均Makespan: {final_avg_makespan:.2f}s")
        print(f"   最佳Makespan: {best_makespan:.2f}s")

        return training_summary
    
    def save_model(self, model_path: str):
        """保存训练好的模型"""
        if self.agent is None:
            raise ValueError("智能体尚未初始化或训练")
        
        checkpoint = {
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'training_step': self.agent.training_step,
            'epsilon': self.agent.epsilon,
            'config': self.config,
            'training_history': self.training_history,
            'metadata': {
                'state_dim': self.agent.state_dim,
                'action_dim': self.agent.action_dim,
                'training_completed': True,
                'final_performance': {
                    'avg_makespan': np.mean([h['makespan'] for h in self.training_history[-100:]]),
                    'best_makespan': min(h['makespan'] for h in self.training_history)
                }
            }
        }
        
        torch.save(checkpoint, model_path)
        print(f"📁 模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = model_path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 训练历史已保存: {history_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='WASS-RAG 改进DRL训练器')
    parser.add_argument('--config', default='configs/experiment.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练episode数')
    parser.add_argument('--output', default='models/improved_wass_drl.pth', help='输出模型路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化训练器
    trainer = WRENCHDRLTrainer(args.config)
    
    # 训练
    results = trainer.train(args.episodes)
    
    # 保存模型
    trainer.save_model(args.output)
    
    # 保存训练结果摘要
    summary_path = args.output.replace('.pth', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"🎉 训练完成! 模型和结果已保存到 {Path(args.output).parent}")

if __name__ == "__main__":
    main()
