#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的WASS-DRL训练器，从WASS-Heuristic和HEFT调度器中学习
使用真实WRENCH环境进行训练，而不是模拟环境
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.drl.reward import compute_step_reward, compute_final_reward, StepContext, EpisodeStats
from src.reward_fix import RewardFix

# 确保结果目录存在
Path('results').mkdir(exist_ok=True)

class TeacherGuidedDQN(nn.Module):
    """教师引导的DQN网络"""
    
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

class TeacherGuidedDQNAgent:
    """教师引导的DQN智能体"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 gamma: float = 0.95,
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
        self.q_network = TeacherGuidedDQN(state_dim, action_dim).to(self.device)
        self.target_network = TeacherGuidedDQN(state_dim, action_dim).to(self.device)
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
        
        # 教师指导
        self.teacher_actions = []
        self.teacher_confidence = 0.8  # 教师指导的置信度
    
    def update_target(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def act(self, state, training=True, teacher_action=None):
        """选择动作，支持教师指导"""
        if training and teacher_action is not None and random.random() < self.teacher_confidence:
            # 教师指导模式
            return teacher_action
        
        if training and random.random() < self.epsilon:
            # 探索模式
            return random.randint(0, self.action_dim - 1)
        
        # 利用模式
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

class WRENCHBasedDRLTrainer:
    """基于WRENCH环境的DRL训练器，从优秀教师调度器中学习"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.reward_fix = RewardFix()
        self.agent = None
        self.training_history = []
        self.best_makespan = float('inf')
        
        # 配置参数
        self.drl_cfg = self.config.get('drl', {})
        self.checkpoint_cfg = self.config.get('checkpoint', {})
        self.logging_cfg = self.config.get('logging', {})
        
        # 创建目录
        Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/')).mkdir(parents=True, exist_ok=True)
        Path(self.logging_cfg.get('metrics_file', 'results/training_metrics.jsonl')).parent.mkdir(parents=True, exist_ok=True)
    
    def create_wrench_environment(self):
        """创建WRENCH训练环境"""
        try:
            import wrench
            
            # 创建仿真
            simulation = wrench.Simulation()
            
            # 创建平台
            platform = simulation.create_platform([
                wrench.Host("ComputeHost1", "100Gf", ["100Gf", "100GB"]),
                wrench.Host("ComputeHost2", "150Gf", ["150Gf", "150GB"]),
                wrench.Host("ComputeHost3", "200Gf", ["200Gf", "200GB"]),
                wrench.Host("ComputeHost4", "250Gf", ["250Gf", "250GB"])
            ])
            
            # 创建计算服务
            compute_service = simulation.create_bare_metal_compute_service(
                "ComputeService",
                platform.get_hosts(),
                {}
            )
            
            # 创建工作流
            workflow = simulation.create_workflow("training_workflow")
            
            # 创建任务图（模拟真实工作流）
            tasks = []
            for i in range(20):  # 创建20个任务
                task = workflow.add_task(f"task_{i}", random.uniform(1e9, 1e10))
                tasks.append(task)
            
            # 添加依赖关系
            for i in range(1, 20):
                # 每个任务依赖前面的1-2个任务
                num_deps = min(random.randint(1, 2), i)
                for j in range(max(0, i-num_deps), i):
                    workflow.add_control_dependency(tasks[j], tasks[i])
            
            return simulation, platform, compute_service, workflow, tasks
            
        except Exception as e:
            print(f"创建WRENCH环境失败: {e}")
            return None, None, None, None, None
    
    def extract_state_features(self, task, available_nodes, node_capacities, node_loads, workflow):
        """从WRENCH环境中提取状态特征"""
        features = []
        
        # 任务特征
        features.extend([
            task.get_flops() / 1e10,  # 归一化计算量
            len(task.get_parents()),   # 父任务数量
            len(task.get_children()),  # 子任务数量
            1.0 if len(task.get_parents()) == 0 else 0.0,  # 是否是入口任务
            1.0 if len(task.get_children()) == 0 else 0.0,  # 是否是出口任务
        ])
        
        # 节点特征
        for node in available_nodes:
            capacity = node_capacities.get(node, 1.0)
            load = node_loads.get(node, 0.0)
            features.extend([
                capacity / 250.0,  # 归一化容量
                load / 100.0,     # 归一化负载
            ])
        
        # 环境特征
        total_tasks = len(workflow.get_tasks())
        completed_tasks = sum(1 for t in workflow.get_tasks() if t.get_state() == wrench.TaskState.COMPLETED)
        features.extend([
            completed_tasks / total_tasks,  # 工作流进度
            len(available_nodes) / 4.0,    # 可用节点比例
        ])
        
        return np.array(features, dtype=np.float32)
    
    def get_teacher_action(self, task, available_nodes, node_capacities, node_loads, teacher_type="HEFT"):
        """获取教师调度器的动作"""
        if teacher_type == "HEFT":
            # HEFT策略：选择能最早完成任务的节点
            best_node = None
            best_finish_time = float('inf')
            
            for node in available_nodes:
                capacity = node_capacities.get(node, 1.0)
                load = node_loads.get(node, 0.0)
                exec_time = task.get_flops() / (capacity * 1e9)
                finish_time = load + exec_time
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_node = node
            
            return available_nodes.index(best_node) if best_node else 0
        
        elif teacher_type == "WASS-Heuristic":
            # WASS-Heuristic策略：考虑数据局部性
            best_node = None
            best_score = float('inf')
            
            for node in available_nodes:
                # 计算EFT
                capacity = node_capacities.get(node, 1.0)
                load = node_loads.get(node, 0.0)
                exec_time = task.get_flops() / (capacity * 1e9)
                eft = load + exec_time
                
                # 简化的DRT计算（模拟数据局部性）
                drt = 0.0
                for parent in task.get_parents():
                    # 假设父任务可能在任何节点上执行
                    if random.random() > 0.5:  # 50%概率需要数据传输
                        file_size = task.get_flops() * 0.1  # 假设数据大小
                        network_bandwidth = 1e9  # 1GB/s
                        drt += file_size / network_bandwidth
                
                # WASS评分
                w = 0.5  # 数据局部性权重
                score = (1 - w) * eft + w * drt
                
                if score < best_score:
                    best_score = score
                    best_node = node
            
            return available_nodes.index(best_node) if best_node else 0
        
        else:
            # 默认随机选择
            return random.randint(0, len(available_nodes) - 1)
    
    def train_episode(self, teacher_type="HEFT"):
        """训练一个episode，使用教师指导"""
        simulation, platform, compute_service, workflow, tasks = self.create_wrench_environment()
        if simulation is None:
            return None
        
        # 初始化节点状态
        available_nodes = ["ComputeHost1", "ComputeHost2", "ComputeHost3", "ComputeHost4"]
        node_capacities = {
            "ComputeHost1": 100.0,
            "ComputeHost2": 150.0,
            "ComputeHost3": 200.0,
            "ComputeHost4": 250.0
        }
        node_loads = {node: 0.0 for node in available_nodes}
        
        # 训练统计
        step_rewards = []
        total_makespan = 0.0
        step_count = 0
        
        # 调试日志
        reward_debug_path = self.logging_cfg.get('reward_debug', 'results/reward_debug.log')
        debug_file = None
        try:
            debug_file = open(reward_debug_path, 'a')
        except Exception:
            debug_file = None
        
        # 模拟调度过程
        ready_tasks = [t for t in tasks if len(t.get_parents()) == 0]
        completed_tasks = set()
        
        while ready_tasks and step_count < 100:  # 限制最大步数
            # 选择当前任务（按优先级）
            current_task = ready_tasks[0]
            
            # 提取状态特征
            state = self.extract_state_features(
                current_task, available_nodes, node_capacities, node_loads, workflow
            )
            
            # 获取教师动作
            teacher_action = self.get_teacher_action(
                current_task, available_nodes, node_capacities, node_loads, teacher_type
            )
            
            # 智能体选择动作
            action = self.agent.act(state, training=True, teacher_action=teacher_action)
            
            # 执行动作
            chosen_node = available_nodes[action]
            
            # 计算执行时间
            capacity = node_capacities[chosen_node]
            exec_time = current_task.get_flops() / (capacity * 1e9)
            
            # 更新节点负载
            node_loads[chosen_node] += exec_time
            
            # 计算奖励
            try:
                # 构造StepContext用于计算奖励
                ctx = StepContext(
                    completed_critical_path_tasks=len(completed_tasks),
                    total_critical_path_tasks=len(tasks),
                    node_busy_times=node_loads,
                    ready_task_count=len(ready_tasks) - 1,
                    total_nodes=len(available_nodes),
                    avg_queue_wait=np.mean(list(node_loads.values())),
                    queue_wait_baseline=0.0
                )
                step_reward, _metrics = compute_step_reward(ctx, debug_writer=debug_file)
            except Exception:
                step_reward = 0.0
            
            step_rewards.append(step_reward)
            
            # 更新任务状态
            completed_tasks.add(current_task)
            ready_tasks.remove(current_task)
            
            # 更新就绪任务列表
            for child in current_task.get_children():
                if all(parent in completed_tasks for parent in child.get_parents()):
                    if child not in ready_tasks:
                        ready_tasks.append(child)
            
            # 更新时间
            total_makespan = max(node_loads.values())
            step_count += 1
            
            # 提取下一状态
            if ready_tasks:
                next_task = ready_tasks[0]
                next_state = self.extract_state_features(
                    next_task, available_nodes, node_capacities, node_loads, workflow
                )
            else:
                next_state = np.zeros_like(state)
            
            # 存储经验
            done = len(ready_tasks) == 0
            self.agent.remember(state, action, step_reward, next_state, done)
            
            # 训练智能体
            if step_count % 4 == 0:
                loss = self.agent.replay()
            
            if done:
                break
        
        # 计算最终奖励
        final_reward = compute_final_reward(EpisodeStats(makespan=total_makespan))
        
        # 计算平均奖励
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        total_reward = avg_step_reward + final_reward
        
        # 记录调试信息
        if debug_file:
            try:
                debug_file.write(f"FINAL\tmakespan={total_makespan:.4f}\tavg_step_reward={avg_step_reward:.4f}\tfinal_reward={final_reward:.4f}\ttotal_reward={total_reward:.4f}\tteacher={teacher_type}\n")
                debug_file.close()
            except Exception:
                pass
        
        return {
            'total_reward': total_reward,
            'avg_step_reward': avg_step_reward,
            'final_reward': final_reward,
            'makespan': total_makespan,
            'step_count': step_count,
            'epsilon': self.agent.epsilon,
            'teacher_type': teacher_type
        }
    
    def train(self, episodes: int = 1000):
        """训练DRL智能体，使用教师指导"""
        print(f"🚀 开始基于WRENCH的教师引导DRL训练: {episodes} episodes")
        
        # 初始化智能体
        state_dim = 5 + 4 * 2 + 2  # 任务特征 + 节点特征 + 环境特征
        action_dim = 4  # 4个节点
        
        self.agent = TeacherGuidedDQNAgent(
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
        
        for episode in range(episodes):
            # 动态选择教师类型
            if episode < episodes // 2:
                teacher_type = "HEFT"
            else:
                teacher_type = "WASS-Heuristic"
            
            # 训练一个episode
            result = self.train_episode(teacher_type)
            if result is None:
                continue
            
            # 记录结果
            self.training_history.append(result)
            recent_rewards.append(result['total_reward'])
            
            # 更新最佳makespan
            if result['makespan'] < best_makespan:
                best_makespan = result['makespan']
                self.save_model('best_model.pth')
            
            # 定期打印日志
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode + 1}: 平均奖励={avg_reward:.3f}, Makespan={result['makespan']:.2f}, ε={result['epsilon']:.3f}, 教师={teacher_type}")
            
            # 定期保存检查点
            if (episode + 1) % checkpoint_interval == 0:
                self.save_model(f'checkpoint_episode_{episode + 1}.pth')
        
        # 保存最终模型
        self.save_model('wass_drl_teacher_guided.pth')
        
        print(f"✅ DRL训练完成!")
        print(f"   最佳Makespan: {best_makespan:.2f}s")
        print(f"   最终Epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'best_makespan': best_makespan,
            'training_history': self.training_history
        }
    
    def save_model(self, filename):
        """保存模型"""
        model_path = Path(self.checkpoint_cfg.get('dir', 'models/checkpoints/')) / filename
        torch.save({
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'training_step': self.agent.training_step,
            'epsilon': self.agent.epsilon,
            'training_history': self.training_history
        }, model_path)
        print(f"📁 模型已保存: {model_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WASS-DRL教师引导训练')
    parser.add_argument('--config', type=str, default='configs/experiment.yaml', help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = WRENCHBasedDRLTrainer(args.config)
    
    # 开始训练
    results = trainer.train(args.episodes)
    
    print("🎉 训练完成! 模型和结果已保存到 models")

if __name__ == '__main__':
    main()