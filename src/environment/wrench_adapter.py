import wrench
import gym
from gym import spaces
import numpy as np
from typing import List, Dict, Any

class WrenchEnv(gym.Env):
    """
    一个将WRENCH模拟器封装成OpenAI Gym环境的适配器。
    这是DRL智能体进行交互式学习的基础。
    """
    def __init__(self, platform_file: str, workflow_file: str,
                 task_feature_extractor: callable,
                 node_feature_extractor: callable):
        super(WrenchEnv, self).__init__()

        self.platform_file = platform_file
        self.workflow_file = workflow_file
        self.task_feature_extractor = task_feature_extractor
        self.node_feature_extractor = node_feature_extractor

        # 初始化WRENCH模拟器
        self.simulation = self._create_simulation()
        
        # 定义动作空间和观察空间
        self.num_hosts = len(self.simulation.get_platform().get_compute_hosts())
        self.action_space = spaces.Discrete(self.num_hosts)

        # 观察空间（状态）的维度需要根据您的特征提取逻辑来确定
        # 我们在这里使用一个占位符，稍后在训练器中动态确定
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _create_simulation(self):
        """创建一个新的WRENCH模拟会话。"""
        sim = wrench.Simulation()
        sim.add_platform(self.platform_file)
        # 我们将使用一个自定义的调度器来与DRL智能体交互
        # 因此在这里不添加工作流文件或调度器
        return sim

    def reset(self):
        """
        重置环境到一个新的回合（episode）。
        这会重新加载工作流并准备模拟。
        """
        # 为每个回合创建一个新的模拟实例以保证隔离性
        self.simulation = self._create_simulation()
        
        # 加载工作流并创建一个特殊的 "drl_scheduler"
        # 这个调度器会在需要决策时暂停模拟，等待我们的 'action'
        self.drl_scheduler = wrench.DRLScheduler()
        self.simulation.add_workflow(self.workflow_file, self.drl_scheduler)
        
        # 启动模拟，它会在第一个调度事件时暂停
        self.simulation.launch()
        
        return self._get_state()

    def step(self, action: int):
        """
        在环境中执行一步。
        'action' 是DRL智能体选择的主机（host）ID。
        """
        # 检查模拟是否已结束
        if self.drl_scheduler.is_simulation_over():
            return self._get_state(), 0, True, {'makespan': self.simulation.get_makespan()}

        # 获取可用的主机
        available_hosts = self.simulation.get_platform().get_compute_hosts()
        if action >= len(available_hosts):
            # 如果动作无效，给予一个负奖励并结束
            return self._get_state(), -1, True, {"error": "Invalid action"}

        # 将DRL智能体的决策（action）传递给WRENCH调度器
        decision = wrench.SchedulingDecision(
            job_id=self.drl_scheduler.get_waiting_job().get_id(),
            resource_id=available_hosts[action].get_id()
        )
        self.drl_scheduler.make_decision(decision)
        
        # 继续模拟直到下一个决策点或模拟结束
        self.simulation.resume()
        
        # 获取新状态
        state = self._get_state()
        
        # 在这个适配器中，我们不计算奖励，奖励将在训练器中通过RAG导师计算
        # done 标志也由训练器在外部判断
        reward = 0.0 
        done = self.drl_scheduler.is_simulation_over()
        
        info = {}
        if done:
            info['makespan'] = self.simulation.get_makespan()

        return state, reward, done, info

    def _get_state(self):
        """
        从WRENCH模拟器中提取当前状态，并将其转换为特征向量。
        """
        # 如果模拟结束，返回一个零向量
        if self.drl_scheduler.is_simulation_over():
            return np.zeros(self.observation_space.shape)

        # 获取当前需要调度的任务
        current_job = self.drl_scheduler.get_waiting_job()
        
        # 提取任务和节点的特征
        task_features = self.task_feature_extractor(current_job)
        node_features = self.node_feature_extractor(self.simulation.get_platform())

        # 将所有特征拼接成一个扁平的状态向量
        # 注意：这里的拼接逻辑需要与您的`extract_state_features`方法保持一致
        state_vector = np.concatenate([task_features, node_features]).astype(np.float32)
        
        return state_vector

    def render(self, mode='human'):
        pass

    def close(self):
        pass