import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class TrainingFix:
    """修复后的训练策略"""
    
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.01, total_episodes=2000):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.performance_history = []
        self.epsilon_history = []
        
    def adaptive_epsilon(self, episode, recent_performance=None):
        """
        自适应epsilon探索策略
        
        Args:
            episode: 当前回合数
            recent_performance: 最近性能指标（可选）
        
        Returns:
            当前epsilon值
        """
        self.current_episode = episode
        
        # 1. 基础衰减（指数衰减）
        progress = min(1.0, episode / self.total_episodes)
        base_epsilon = self.initial_epsilon * (self.min_epsilon / self.initial_epsilon) ** progress
        
        # 2. 性能自适应调整
        if recent_performance is not None:
            # 如果性能下降，增加探索
            if len(self.performance_history) > 0:
                last_performance = self.performance_history[-1]
                if recent_performance < last_performance * 0.9:  # 性能下降超过10%
                    base_epsilon = min(1.0, base_epsilon * 1.5)  # 增加探索
                elif recent_performance > last_performance * 1.1:  # 性能提升超过10%
                    base_epsilon = max(self.min_epsilon, base_epsilon * 0.8)  # 减少探索
            
            self.performance_history.append(recent_performance)
        
        # 3. 周期性探索
        # 每100个回合，进行一次高探索
        if episode % 100 == 0:
            base_epsilon = max(0.3, base_epsilon * 2.0)
        
        # 4. 确保epsilon在合理范围内
        epsilon = max(self.min_epsilon, min(1.0, base_epsilon))
        
        # 5. 记录历史
        self.epsilon_history.append(epsilon)
        
        return epsilon
    
    def curriculum_learning(self, episode, task_complexity):
        """
        课程学习策略，逐步增加任务复杂度
        
        Args:
            episode: 当前回合数
            task_complexity: 原始任务复杂度
        
        Returns:
            调整后的任务复杂度
        """
        # 分阶段课程学习
        if episode < 500:  # 阶段1：简单任务
            complexity_factor = 0.5
        elif episode < 1000:  # 阶段2：中等任务
            complexity_factor = 0.75
        elif episode < 1500:  # 阶段3：复杂任务
            complexity_factor = 0.9
        else:  # 阶段4：原始复杂度
            complexity_factor = 1.0
        
        # 应用复杂度调整
        adjusted_complexity = task_complexity * complexity_factor
        
        return adjusted_complexity
    
    def adaptive_learning_rate(self, episode, base_lr=0.001):
        """
        自适应学习率调整
        
        Args:
            episode: 当前回合数
            base_lr: 基础学习率
        
        Returns:
            调整后的学习率
        """
        # 1. 基础衰减
        progress = min(1.0, episode / self.total_episodes)
        lr = base_lr * (0.1 / 1.0) ** progress
        
        # 2. 性能自适应
        if len(self.performance_history) > 10:
            # 计算最近10个回合的性能变化
            recent_perf = self.performance_history[-10:]
            perf_trend = np.polyfit(range(10), recent_perf, 1)[0]
            
            # 如果性能提升缓慢，增加学习率
            if perf_trend < 0.01:
                lr = min(base_lr, lr * 1.5)
            # 如果性能提升很快，减少学习率
            elif perf_trend > 0.1:
                lr = max(base_lr * 0.1, lr * 0.8)
        
        return lr
    
    def dynamic_target_update(self, episode, base_frequency=10):
        """
        动态目标网络更新频率
        
        Args:
            episode: 当前回合数
            base_frequency: 基础更新频率
        
        Returns:
            是否应该更新目标网络
        """
        # 训练初期更频繁更新
        if episode < 200:
            return episode % 5 == 0
        # 中期适度更新
        elif episode < 1000:
            return episode % base_frequency == 0
        # 后期较少更新
        else:
            return episode % (base_frequency * 2) == 0
    
    def debug_training_info(self, episode, epsilon, lr, should_update_target):
        """记录训练调试信息"""
        logger.info(f"Training Debug - Episode={episode}, "
                   f"Epsilon={epsilon:.4f}, "
                   f"LearningRate={lr:.6f}, "
                   f"UpdateTarget={should_update_target}")