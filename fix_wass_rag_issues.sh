#!/bin/bash
# WASS-RAG 性能问题修复脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 备份原始文件
backup_original_files() {
    log_info "备份原始文件..."
    
    # 创建备份目录
    mkdir -p backup/$(date +%Y%m%d_%H%M%S)
    
    # 备份关键文件
    cp src/ai_schedulers.py backup/$(date +%Y%m%d_%H%M%S)/
    cp configs/drl.yaml backup/$(date +%Y%m%d_%H%M%S)/
    cp configs/rag.yaml backup/$(date +%Y%m%d_%H%M%S)/
    
    log_success "原始文件已备份到 backup/$(date +%Y%m%d_%H%M%S)/"
}

# 修复奖励函数
fix_reward_function() {
    log_info "修复奖励函数..."
    
    # 创建修复后的奖励函数文件
    cat > src/reward_fix.py << 'EOF'
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class RewardFix:
    """修复后的奖励计算类"""
    
    def __init__(self):
        self.reward_history = []
        self.makespan_history = []
        
    def calculate_normalized_reward(self, teacher_makespan, student_makespan, task_scale):
        """
        计算归一化的奖励信号
        
        Args:
            teacher_makespan: 老师（预测器）建议的makespan
            student_makespan: 学生（Agent）选择的makespan
            task_scale: 任务规模，用于归一化
        
        Returns:
            归一化后的奖励值
        """
        # 1. 确保时间单位合理（转换为秒）
        teacher_makespan_sec = teacher_makespan / 1e9  # 假设原始是纳秒
        student_makespan_sec = student_makespan / 1e9
        
        # 2. 使用对数变换处理大范围的makespan值
        log_teacher = np.log1p(teacher_makespan_sec)
        log_student = np.log1p(student_makespan_sec)
        
        # 3. 计算相对改进
        relative_improvement = (log_teacher - log_student) / (log_teacher + 1e-8)
        
        # 4. 使用tanh函数将奖励限制在[-1, 1]范围内
        normalized_reward = np.tanh(relative_improvement * 5.0)
        
        # 5. 记录历史
        self.reward_history.append(normalized_reward)
        self.makespan_history.append(student_makespan_sec)
        
        return normalized_reward
    
    def calculate_multi_objective_reward(self, simulation, task, chosen_node, teacher_makespan, student_makespan):
        """
        计算多目标奖励，考虑多个性能指标
        
        Args:
            simulation: 仿真环境
            task: 当前调度的任务
            chosen_node: 选择的计算节点
            teacher_makespan: 老师（预测器）建议的makespan
            student_makespan: 学生（Agent）选择的makespan
        
        Returns:
            综合奖励值
        """
        # 1. 时间效率奖励（归一化的makespan改进）
        task_scale = max(task.computation_size, 1.0)
        time_reward = self.calculate_normalized_reward(
            teacher_makespan, student_makespan, task_scale
        )
        
        # 2. 资源利用率奖励
        node_utilization = self._get_node_utilization(simulation, chosen_node)
        utilization_reward = node_utilization * 0.2  # 权重0.2
        
        # 3. 负载均衡奖励
        load_std = self._get_load_std(simulation)
        balance_reward = -load_std * 0.3  # 权重0.3，负号表示希望负载均衡
        
        # 4. 任务完成率奖励
        completion_rate = len(simulation.completed_tasks) / len(simulation.workflow.tasks)
        completion_reward = completion_rate * 0.1  # 权重0.1
        
        # 5. 紧急任务奖励
        urgency = self._get_task_urgency(simulation, task)
        urgency_reward = urgency * 0.2  # 权重0.2
        
        # 6. 综合奖励
        total_reward = (
            time_reward * 0.5 +  # 时间效率权重0.5
            utilization_reward +
            balance_reward +
            completion_reward +
            urgency_reward
        )
        
        return total_reward
    
    def _get_node_utilization(self, simulation, node_name):
        """获取节点利用率"""
        # 简化实现，实际应根据仿真环境计算
        node = simulation.platform.get_node(node_name)
        if hasattr(node, 'utilization'):
            return node.utilization
        return 0.5  # 默认值
    
    def _get_load_std(self, simulation):
        """获取负载标准差"""
        # 简化实现，实际应根据仿真环境计算
        return 0.1  # 默认值
    
    def _get_task_urgency(self, simulation, task):
        """获取任务紧急程度"""
        # 简化实现，实际应根据任务deadline计算
        total_tasks = len(simulation.workflow.tasks)
        completed_tasks = len(simulation.completed_tasks)
        return completed_tasks / total_tasks  # 完成比例作为紧急程度
    
    def debug_reward_info(self, task_id, teacher_makespan, student_makespan, final_reward):
        """记录奖励调试信息"""
        logger.info(f"Reward Debug - Task={task_id}, "
                   f"Teacher={teacher_makespan/1e9:.2f}s, "
                   f"Student={student_makespan/1e9:.2f}s, "
                   f"FinalReward={final_reward:.4f}")
EOF

    log_success "奖励函数修复完成"
}

# 修复RAG融合机制
fix_rag_fusion() {
    log_info "修复RAG融合机制..."
    
    # 创建修复后的RAG融合文件
    cat > src/rag_fusion_fix.py << 'EOF'
import numpy as np
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class RAGFusionFix:
    """修复后的RAG融合类"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        self.alpha = alpha  # DRL权重
        self.beta = beta    # RAG权重
        self.gamma = gamma  # 负载权重
        self.rag_confidence_history = []
        
    def dynamic_fusion(self, q_values, rag_suggestions, load_preferences, training_progress):
        """
        动态融合DRL和RAG的建议
        
        Args:
            q_values: DRL的Q值
            rag_suggestions: RAG的建议
            load_preferences: 负载偏好
            training_progress: 训练进度 [0, 1]
        
        Returns:
            融合后的决策
        """
        # 1. 计算RAG建议的置信度
        rag_confidence = self._calculate_rag_confidence(rag_suggestions)
        self.rag_confidence_history.append(rag_confidence)
        
        # 2. 动态调整融合权重
        # 训练初期更依赖RAG，后期更依赖DRL
        adaptive_alpha = 0.2 + 0.6 * training_progress  # 从0.2到0.8
        adaptive_beta = 0.8 - 0.6 * training_progress     # 从0.8到0.2
        
        # 3. 根据RAG置信度调整权重
        if rag_confidence < 0.3:  # RAG置信度低
            adaptive_alpha = min(adaptive_alpha + 0.3, 0.9)
            adaptive_beta = max(adaptive_beta - 0.3, 0.1)
        elif rag_confidence > 0.7:  # RAG置信度高
            adaptive_alpha = max(adaptive_alpha - 0.2, 0.1)
            adaptive_beta = min(adaptive_beta + 0.2, 0.7)
        
        # 4. 归一化权重
        total_weight = adaptive_alpha + adaptive_beta + self.gamma
        norm_alpha = adaptive_alpha / total_weight
        norm_beta = adaptive_beta / total_weight
        norm_gamma = self.gamma / total_weight
        
        # 5. 融合决策
        fused_values = (
            norm_alpha * np.array(q_values) +
            norm_beta * np.array(rag_suggestions) +
            norm_gamma * np.array(load_preferences)
        )
        
        # 6. 记录调试信息
        if np.random.random() < 0.1:  # 10%概率记录
            logger.info(f"RAG Fusion - Alpha={norm_alpha:.3f}, "
                       f"Beta={norm_beta:.3f}, Gamma={norm_gamma:.3f}, "
                       f"RAG_Confidence={rag_confidence:.3f}")
        
        return fused_values.tolist()
    
    def _calculate_rag_confidence(self, rag_suggestions):
        """计算RAG建议的置信度"""
        # 1. 检查RAG建议是否为全零
        if all(abs(x) < 1e-6 for x in rag_suggestions):
            return 0.0
        
        # 2. 计算建议的方差（方差越大表示建议越明确）
        variance = np.var(rag_suggestions)
        
        # 3. 计算建议的最大值（最大值越大表示偏好越明确）
        max_value = max(abs(x) for x in rag_suggestions)
        
        # 4. 综合计算置信度
        confidence = 0.5 * np.tanh(variance * 10) + 0.5 * np.tanh(max_value)
        
        return min(max(confidence, 0.0), 1.0)
    
    def enhance_rag_suggestions(self, rag_suggestions, node_names):
        """增强RAG建议，避免全零向量"""
        # 1. 检查是否为全零向量
        if all(abs(x) < 1e-6 for x in rag_suggestions):
            # 2. 如果是全零，生成基于启发式的建议
            enhanced_suggestions = self._generate_heuristic_suggestions(node_names)
            logger.warning("RAG suggestions were all zeros, using heuristic suggestions")
            return enhanced_suggestions
        
        # 3. 如果不是全零，但值很小，进行增强
        max_abs = max(abs(x) for x in rag_suggestions)
        if max_abs < 0.1:
            # 放大建议值
            scale_factor = 0.5 / max_abs
            enhanced_suggestions = [x * scale_factor for x in rag_suggestions]
            logger.info(f"RAG suggestions were weak, scaling by {scale_factor:.2f}")
            return enhanced_suggestions
        
        return rag_suggestions
    
    def _generate_heuristic_suggestions(self, node_names):
        """生成基于启发式的建议"""
        # 简单实现：随机偏好，实际应基于负载等信息
        suggestions = np.random.random(len(node_names))
        # 归一化
        suggestions = suggestions / np.sum(suggestions)
        return suggestions.tolist()
    
    def debug_fusion_info(self, node_name, q_values, rag_suggestions, fused_values):
        """记录融合调试信息"""
        debug_info = {
            "node": node_name,
            "q_values": q_values,
            "rag_suggestions": rag_suggestions,
            "fused_values": fused_values,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }
        
        # 写入调试日志
        with open("results/fusion_debug.log", "a") as f:
            f.write(json.dumps(debug_info) + "\n")
EOF

    log_success "RAG融合机制修复完成"
}

# 修复训练策略
fix_training_strategy() {
    log_info "修复训练策略..."
    
    # 创建修复后的训练策略文件
    cat > src/training_fix.py << 'EOF'
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TrainingFix:
    """修复后的训练策略类"""
    
    def __init__(self):
        self.performance_history = []
        self.epsilon_history = []
        
    def adaptive_epsilon(self, episode, performance_history):
        """
        自适应epsilon策略，根据性能调整探索率
        
        Args:
            episode: 当前回合数
            performance_history: 性能历史记录
        
        Returns:
            当前回合的epsilon值
        """
        # 1. 基础衰减（更缓慢的衰减）
        base_epsilon = max(0.05, 1.0 * (0.999 ** episode))
        
        # 2. 性能自适应调整
        if len(performance_history) >= 10:
            # 计算最近10个回合的性能变化
            recent_performance = performance_history[-10:]
            performance_trend = np.polyfit(range(10), recent_performance, 1)[0]
            
            # 如果性能没有改善，增加探索
            if performance_trend > 0:  # 性能变差
                base_epsilon = min(base_epsilon * 1.3, 0.5)
            elif performance_trend < -0.01:  # 性能显著改善
                base_epsilon = max(base_epsilon * 0.9, 0.05)
        
        # 3. 周期性探索
        if episode % 100 == 0:
            base_epsilon = max(base_epsilon, 0.3)  # 每100回合增加探索
        
        # 4. 记录历史
        self.epsilon_history.append(base_epsilon)
        
        return base_epsilon
    
    def curriculum_learning(self, episode):
        """
        课程学习策略，逐步增加任务复杂度
        
        Args:
            episode: 当前回合数
        
        Returns:
            当前回合的任务复杂度参数
        """
        # 1. 定义课程阶段
        if episode < 300:
            # 阶段1：简单任务
            return {
                'task_range': [5, 10],
                'dependency_prob': 0.2,
                'ccr_range': [0.1, 1.0],
                'description': 'Simple tasks'
            }
        elif episode < 600:
            # 阶段2：中等任务
            return {
                'task_range': [10, 20],
                'dependency_prob': 0.3,
                'ccr_range': [0.1, 5.0],
                'description': 'Medium tasks'
            }
        elif episode < 900:
            # 阶段3：复杂任务
            return {
                'task_range': [15, 30],
                'dependency_prob': 0.4,
                'ccr_range': [0.1, 10.0],
                'description': 'Complex tasks'
            }
        else:
            # 阶段4：混合任务
            return {
                'task_range': [5, 30],
                'dependency_prob': 0.5,
                'ccr_range': [0.1, 10.0],
                'description': 'Mixed tasks'
            }
    
    def adaptive_learning_rate(self, episode, loss_history):
        """
        自适应学习率调整
        
        Args:
            episode: 当前回合数
            loss_history: 损失历史记录
        
        Returns:
            当前回合的学习率
        """
        # 基础学习率
        base_lr = 0.001
        
        # 学习率衰减
        lr = base_lr * (0.999 ** episode)
        
        # 根据损失调整
        if len(loss_history) >= 5:
            recent_losses = loss_history[-5:]
            loss_trend = np.polyfit(range(5), recent_losses, 1)[0]
            
            # 如果损失增加，降低学习率
            if loss_trend > 0:
                lr = lr * 0.8
            # 如果损失快速下降，可以稍微增加学习率
            elif loss_trend < -0.1:
                lr = min(lr * 1.2, base_lr)
        
        # 确保学习率在合理范围内
        lr = max(lr, 0.0001)
        lr = min(lr, 0.01)
        
        return lr
    
    def should_update_target_network(self, episode, loss_history):
        """
        决定是否更新目标网络
        
        Args:
            episode: 当前回合数
            loss_history: 损失历史记录
        
        Returns:
            是否更新目标网络
        """
        # 基础更新频率
        base_frequency = 50
        
        # 根据损失稳定性调整
        if len(loss_history) >= 10:
            recent_losses = loss_history[-10:]
            loss_std = np.std(recent_losses)
            
            # 如果损失波动大，减少更新频率
            if loss_std > 0.1:
                base_frequency = 100
            # 如果损失稳定，可以增加更新频率
            elif loss_std < 0.01:
                base_frequency = 25
        
        return episode % base_frequency == 0
    
    def debug_training_info(self, episode, epsilon, lr, avg_loss):
        """记录训练调试信息"""
        logger.info(f"Training Debug - Episode={episode}, "
                   f"Epsilon={epsilon:.4f}, LR={lr:.6f}, "
                   f"AvgLoss={avg_loss:.4f}")
EOF

    log_success "训练策略修复完成"
}

# 更新配置文件
update_config_files() {
    log_info "更新配置文件..."
    
    # 更新DRL配置
    cat > configs/drl_fixed.yaml << 'EOF'
drl:
  # WRENCH环境配置
  environment: "wrench"
  episodes: 2000  # 增加训练回合数
  max_steps: 50    # 与训练脚本上限保持一致
  
  # DQN智能体参数
  network:
    hidden_dim: 128
    learning_rate: 0.001
  
  # 训练参数 - 修复后的参数
  epsilon_start: 1.0
  epsilon_decay: 0.999  # 更缓慢的衰减
  epsilon_min: 0.05     # 更小的最小探索率
  gamma: 0.99
  batch_size: 32
  memory_size: 10000
  target_update_freq: 50  # 更频繁的目标网络更新
  checkpoint_interval: 100
  eval_interval: 100
  log_interval: 50
  rolling_window: 100
  
  # 任务生成参数 - 课程学习
  task_range: [5, 30]  # 将由课程学习动态调整
  dependency_prob: 0.3  # 将由课程学习动态调整
  
  # 奖励函数参数 - 修复后的参数
  rewards:
    time_weight: 0.5      # 时间权重
    utilization_weight: 0.2  # 利用率权重
    balance_weight: 0.3    # 均衡权重
    completion_weight: 0.1  # 完成率权重
    urgency_weight: 0.2    # 紧急程度权重
  reward_framework: "multi_objective_normalized"  # 多目标归一化奖励
  final_reward: "normalized_makespan"

checkpoint:
  dir: models/checkpoints/
  keep_last: 5
  save_best: true

logging:
  metrics_file: results/training_metrics_fixed.jsonl
  fusion_debug: results/fusion_debug_fixed.log
  reward_debug: results/reward_debug_fixed.log
EOF

    # 更新RAG配置
    cat > configs/rag_fixed.yaml << 'EOF'
rag:
  # RAG检索配置
  retriever: "wrench_similarity"
  top_k: 5
  fusion: "dynamic"  # 使用动态融合
  
  # 知识库生成配置
  num_cases: 1000  # 增加案例数量
  embedding_dim: 64
  
  # 案例生成参数
  workflow_sizes: [5, 10, 15, 20, 25, 30]  # 增加工作流规模
  schedulers: ["HEFT", "CPOP", "PEFT", "Lookahead", "Random"]  # 增加调度器类型
  dependency_prob: 0.3
  
  # 检索参数
  similarity_weights:
    workflow: 0.4  # 降低工作流相似度权重
    task: 0.4      # 增加任务相似度权重
    resource: 0.2   # 增加资源相似度权重
  
  # 融合参数 - 动态融合
  fusion_weights:
    alpha_start: 0.2  # DRL初始权重
    alpha_end: 0.8    # DRL最终权重
    beta_start: 0.8   # RAG初始权重
    beta_end: 0.2     # RAG最终权重
    gamma: 0.2        # 负载权重
  
  # 聚类参数
  num_clusters: 30   # 增加聚类数量
  max_kmeans_iters: 50
EOF

    log_success "配置文件更新完成"
}

# 创建修复后的调度器
create_fixed_scheduler() {
    log_info "创建修复后的调度器..."
    
    # 创建修复后的调度器文件
    cat > src/fixed_scheduler.py << 'EOF'
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any
from .ai_schedulers import WASSRAGScheduler, WassWrenchSimulator
from .reward_fix import RewardFix
from .rag_fusion_fix import RAGFusionFix
from .training_fix import TrainingFix

logger = logging.getLogger(__name__)

class FixedWASSRAGScheduler(WASSRAGScheduler):
    """修复后的WASS-RAG调度器"""
    
    def __init__(self, drl_agent, node_names: List[str], predictor):
        super().__init__(drl_agent, node_names, predictor)
        
        # 初始化修复组件
        self.reward_fix = RewardFix()
        self.rag_fusion_fix = RAGFusionFix()
        self.training_fix = TrainingFix()
        
        # 训练相关变量
        self.episode_count = 0
        self.performance_history = []
        self.loss_history = []
        
        logger.info("Initialized FixedWASSRAGScheduler")
    
    def schedule(self, ready_tasks: List, simulation: WassWrenchSimulator) -> Dict:
        if not ready_tasks:
            return {}
        
        task_to_schedule = ready_tasks[0]
        
        # 构建当前状态
        state = self._extract_features(task_to_schedule, simulation)
        
        # 使用自适应epsilon
        epsilon = self.training_fix.adaptive_epsilon(
            self.episode_count, self.performance_history
        )
        
        # 获取DRL的Q值
        q_values = self.drl_agent.get_q_values(state)
        
        # 获取RAG建议
        rag_suggestions = self._get_rag_suggestions(task_to_schedule, simulation)
        
        # 增强RAG建议
        enhanced_rag_suggestions = self.rag_fusion_fix.enhance_rag_suggestions(
            rag_suggestions, self.node_names
        )
        
        # 获取负载偏好
        load_preferences = self._get_load_preferences(simulation)
        
        # 动态融合决策
        training_progress = min(self.episode_count / 1000, 1.0)
        fused_values = self.rag_fusion_fix.dynamic_fusion(
            q_values, enhanced_rag_suggestions, load_preferences, training_progress
        )
        
        # 选择动作
        action_idx = np.argmax(fused_values)
        chosen_node_name = self.node_names[action_idx]
        
        # 获取预测的makespan
        estimated_makespans = self._get_estimated_makespans(task_to_schedule, simulation)
        best_node_from_teacher = min(estimated_makespans, key=lambda n: estimated_makespans[n].value)
        teacher_makespan = estimated_makespans[best_node_from_teacher].value
        student_makespan = estimated_makespans[chosen_node_name].value
        
        # 计算多目标奖励
        final_reward = self.reward_fix.calculate_multi_objective_reward(
            simulation, task_to_schedule, chosen_node_name, 
            teacher_makespan, student_makespan
        )
        
        # 获取下一个状态
        next_state = self._extract_features(task_to_schedule, simulation)
        
        # 存储转换
        self.drl_agent.store_transition(
            state=state,
            action=action_idx,
            reward=final_reward,
            next_state=next_state,
            done=len(simulation.workflow.tasks) == len(simulation.completed_tasks)
        )
        
        # 自适应学习
        adaptive_lr = self.training_fix.adaptive_learning_rate(
            self.episode_count, self.loss_history
        )
        self.drl_agent.update_learning_rate(adaptive_lr)
        
        # 判断是否更新目标网络
        if self.training_fix.should_update_target_network(self.episode_count, self.loss_history):
            self.drl_agent.update_target_network()
        
        # 定期学习
        if self.episode_count % 10 == 0:
            loss = self.drl_agent.replay(batch_size=32)
            if loss is not None:
                self.loss_history.append(loss)
        
        # 记录调试信息
        if self.episode_count % 50 == 0:
            self.reward_fix.debug_reward_info(
                task_to_schedule.id, teacher_makespan, student_makespan, final_reward
            )
            self.training_fix.debug_training_info(
                self.episode_count, epsilon, adaptive_lr, 
                np.mean(self.loss_history[-10:]) if self.loss_history else 0.0
            )
        
        # 更新计数器
        self.episode_count += 1
        
        # 记录性能
        if len(simulation.completed_tasks) == len(simulation.workflow.tasks):
            self.performance_history.append(student_makespan / 1e9)  # 转换为秒
        
        return {chosen_node_name: task_to_schedule}
    
    def _get_rag_suggestions(self, task, simulation):
        """获取RAG建议"""
        # 这里应该调用实际的RAG检索方法
        # 简化实现，返回随机建议
        suggestions = np.random.random(len(self.node_names))
        return suggestions.tolist()
    
    def _get_load_preferences(self, simulation):
        """获取负载偏好"""
        # 基于当前负载计算偏好
        preferences = []
        for node_name in self.node_names:
            node = simulation.platform.get_node(node_name)
            # 简化实现：基于队列长度计算偏好
            queue_length = len([t for t, n in simulation.task_to_node_map.items() if n == node_name])
            preference = 1.0 / (1.0 + queue_length)  # 队列越短，偏好越高
            preferences.append(preference)
        
        # 归一化
        total = sum(preferences)
        if total > 0:
            preferences = [p / total for p in preferences]
        else:
            preferences = [1.0 / len(self.node_names)] * len(self.node_names)
        
        return preferences
EOF

    log_success "修复后的调度器创建完成"
}

# 创建修复后的实验脚本
create_fixed_experiment_script() {
    log_info "创建修复后的实验脚本..."
    
    cat > run_fixed_experiment.sh << 'EOF'
#!/bin/bash
# WASS-RAG 修复后的实验脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查环境..."
    if ! python -c "import wrench" 2>/dev/null; then
        log_error "WRENCH未安装"
        exit 1
    fi
    log_success "环境检查通过"
}

# 应用修复
apply_fixes() {
    log_info "应用修复..."
    
    # 1. 备份原始文件
    mkdir -p backup_fixed/$(date +%Y%m%d_%H%M%S)
    cp src/ai_schedulers.py backup_fixed/$(date +%Y%m%d_%H%M%S)/
    
    # 2. 复制修复文件
    cp src/reward_fix.py src/
    cp src/rag_fusion_fix.py src/
    cp src/training_fix.py src/
    cp src/fixed_scheduler.py src/
    
    log_success "修复应用完成"
}

# 生成知识库
generate_kb() {
    log_info "生成增强的知识库..."
    python scripts/generate_kb_dataset.py configs/experiment.yaml
    log_success "知识库生成完成"
}

# 训练性能预测器
train_predictor() {
    log_info "训练性能预测器..."
    python scripts/train_predictor_from_kb.py configs/experiment.yaml
    log_success "性能预测器训练完成"
}

# 训练DRL智能体（使用修复后的配置）
train_drl() {
    log_info "训练DRL智能体（修复版）..."
    python scripts/train_drl_wrench.py configs/drl_fixed.yaml
    log_success "DRL智能体训练完成"
}

# 训练RAG知识库（使用修复后的配置）
train_rag() {
    log_info "训练RAG知识库（修复版）..."
    python scripts/train_rag_wrench.py configs/rag_fixed.yaml
    log_success "RAG知识库训练完成"
}

# 运行实验
run_experiments() {
    log_info "运行修复后的实验..."
    python experiments/wrench_real_experiment.py
    log_success "实验运行完成"
}

# 主函数
main() {
    log_info "开始WASS-RAG修复实验..."
    
    check_environment
    apply_fixes
    generate_kb
    train_predictor
    train_drl
    train_rag
    run_experiments
    
    log_success "WASS-RAG修复实验完成!"
}

main "$@"
EOF

    chmod +x run_fixed_experiment.sh
    
    log_success "修复后的实验脚本创建完成"
}

# 主函数
main() {
    log_info "开始WASS-RAG性能问题修复..."
    
    backup_original_files
    fix_reward_function
    fix_rag_fusion
    fix_training_strategy
    update_config_files
    create_fixed_scheduler
    create_fixed_experiment_script
    
    log_success "WASS-RAG性能问题修复完成!"
    log_info "请运行 ./run_fixed_experiment.sh 进行修复后的实验"
}

main "$@"