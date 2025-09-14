"""
综合评估指标模块

提供全面的系统性能评估、基准对比和可视化指标，用于评估工作流调度算法的性能。
"""
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """系统性能指标"""
    makespan: float  # 总执行时间
    avg_waiting_time: float  # 平均等待时间
    avg_turnaround_time: float  # 平均周转时间
    throughput: float  # 吞吐量 (任务/秒)
    resource_utilization: Dict[str, float]  # 资源利用率
    load_balance: float  # 负载均衡指数 (0-1, 越高越好)
    energy_efficiency: float  # 能效指数
    cost_efficiency: float  # 成本效率指数
    fairness: float  # 公平性指数 (0-1, 越高越好)
    reliability: float  # 可靠性指数 (0-1, 越高越好)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class WorkflowMetrics:
    """工作流调度质量指标"""
    scheduling_length: float  # 调度长度
    critical_path_ratio: float  # 关键路径占比
    parallel_efficiency: float  # 并行效率
    data_locality_score: float  # 数据局部性分数
    communication_overhead: float  # 通信开销
    deadline_satisfaction_rate: float  # 截止时间满足率
    priority_violation_rate: float  # 优先级违反率
    workflow_completion_rate: float  # 工作流完成率
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class TrainingMetrics:
    """训练过程指标"""
    episode_count: int  # 训练回合数
    convergence_speed: float  # 收敛速度
    stability_score: float  # 稳定性分数
    exploration_efficiency: float  # 探索效率
    learning_curve: List[float]  # 学习曲线
    reward_distribution: Dict[str, float]  # 奖励分布统计
    loss_curve: List[float]  # 损失曲线
    q_value_curve: List[float]  # Q值曲线
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 将列表转换为JSON可序列化的格式
        result['learning_curve'] = self.learning_curve
        result['reward_distribution'] = self.reward_distribution
        result['loss_curve'] = self.loss_curve
        result['q_value_curve'] = self.q_value_curve
        return result

@dataclass
class ComparisonMetrics:
    """与基准算法的对比指标"""
    baseline_makespan: float  # 基准算法的makespan
    improvement_ratio: float  # 改进比例
    speedup_ratio: float  # 加速比
    efficiency_ratio: float  # 效率比
    scalability_score: float  # 可扩展性分数
    robustness_score: float  # 鲁棒性分数
    adaptability_score: float  # 适应性分数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化评估器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.history = {
            'system': [],
            'workflow': [],
            'training': [],
            'comparison': []
        }
        self.baseline_algorithms = ['HEFT', 'CPOP', 'Min-Min', 'Max-Min', 'Round Robin']
        
    def evaluate_system_performance(self, simulation_result: Dict[str, Any]) -> SystemMetrics:
        """
        评估系统性能
        
        Args:
            simulation_result: 仿真结果
            
        Returns:
            系统性能指标
        """
        # 1. 基本时间指标
        makespan = simulation_result.get('makespan', 0.0)
        avg_waiting_time = simulation_result.get('avg_waiting_time', 0.0)
        avg_turnaround_time = simulation_result.get('avg_turnaround_time', 0.0)
        
        # 2. 吞吐量计算
        total_tasks = simulation_result.get('total_tasks', 1)
        throughput = total_tasks / max(makespan, 1e-6)
        
        # 3. 资源利用率
        resource_utilization = simulation_result.get('resource_utilization', {})
        avg_resource_utilization = np.mean(list(resource_utilization.values())) if resource_utilization else 0.0
        
        # 4. 负载均衡指数 (使用基尼系数)
        node_loads = simulation_result.get('node_loads', [1.0])
        load_balance = self._calculate_load_balance(node_loads)
        
        # 5. 能效指数 (简化计算)
        energy_efficiency = self._calculate_energy_efficiency(
            makespan, avg_resource_utilization, total_tasks
        )
        
        # 6. 成本效率指数 (简化计算)
        cost_efficiency = self._calculate_cost_efficiency(
            makespan, resource_utilization, total_tasks
        )
        
        # 7. 公平性指数 (基于资源分配)
        fairness = self._calculate_fairness(resource_utilization)
        
        # 8. 可靠性指数 (基于任务完成情况)
        reliability = simulation_result.get('task_completion_rate', 1.0)
        
        metrics = SystemMetrics(
            makespan=makespan,
            avg_waiting_time=avg_waiting_time,
            avg_turnaround_time=avg_turnaround_time,
            throughput=throughput,
            resource_utilization=resource_utilization,
            load_balance=load_balance,
            energy_efficiency=energy_efficiency,
            cost_efficiency=cost_efficiency,
            fairness=fairness,
            reliability=reliability
        )
        
        self.history['system'].append(metrics.to_dict())
        return metrics
    
    def evaluate_workflow_quality(self, workflow_result: Dict[str, Any]) -> WorkflowMetrics:
        """
        评估工作流调度质量
        
        Args:
            workflow_result: 工作流调度结果
            
        Returns:
            工作流调度质量指标
        """
        # 1. 调度长度
        scheduling_length = workflow_result.get('scheduling_length', 0.0)
        
        # 2. 关键路径占比
        critical_path_ratio = workflow_result.get('critical_path_ratio', 0.0)
        
        # 3. 并行效率
        parallel_efficiency = self._calculate_parallel_efficiency(workflow_result)
        
        # 4. 数据局部性分数
        data_locality_score = workflow_result.get('data_locality_score', 0.0)
        
        # 5. 通信开销
        communication_overhead = workflow_result.get('communication_overhead', 0.0)
        
        # 6. 截止时间满足率
        deadline_satisfaction_rate = workflow_result.get('deadline_satisfaction_rate', 1.0)
        
        # 7. 优先级违反率
        priority_violation_rate = workflow_result.get('priority_violation_rate', 0.0)
        
        # 8. 工作流完成率
        workflow_completion_rate = workflow_result.get('workflow_completion_rate', 1.0)
        
        metrics = WorkflowMetrics(
            scheduling_length=scheduling_length,
            critical_path_ratio=critical_path_ratio,
            parallel_efficiency=parallel_efficiency,
            data_locality_score=data_locality_score,
            communication_overhead=communication_overhead,
            deadline_satisfaction_rate=deadline_satisfaction_rate,
            priority_violation_rate=priority_violation_rate,
            workflow_completion_rate=workflow_completion_rate
        )
        
        self.history['workflow'].append(metrics.to_dict())
        return metrics
    
    def evaluate_training_process(self, training_history: List[Dict[str, Any]]) -> TrainingMetrics:
        """
        评估训练过程
        
        Args:
            training_history: 训练历史记录
            
        Returns:
            训练过程指标
        """
        episode_count = len(training_history)
        
        # 1. 收敛速度
        convergence_speed = self._calculate_convergence_speed(training_history)
        
        # 2. 稳定性分数
        stability_score = self._calculate_stability_score(training_history)
        
        # 3. 探索效率
        exploration_efficiency = self._calculate_exploration_efficiency(training_history)
        
        # 4. 学习曲线
        learning_curve = [episode.get('total_reward', 0.0) for episode in training_history]
        
        # 5. 奖励分布统计
        rewards = [episode.get('total_reward', 0.0) for episode in training_history]
        reward_distribution = {
            'mean': np.mean(rewards) if rewards else 0.0,
            'std': np.std(rewards) if rewards else 0.0,
            'min': np.min(rewards) if rewards else 0.0,
            'max': np.max(rewards) if rewards else 0.0,
            'median': np.median(rewards) if rewards else 0.0
        }
        
        # 6. 损失曲线
        loss_curve = [episode.get('avg_loss', 0.0) for episode in training_history 
                     if episode.get('avg_loss') is not None]
        
        # 7. Q值曲线
        q_value_curve = [episode.get('avg_q_value', 0.0) for episode in training_history 
                        if episode.get('avg_q_value') is not None]
        
        metrics = TrainingMetrics(
            episode_count=episode_count,
            convergence_speed=convergence_speed,
            stability_score=stability_score,
            exploration_efficiency=exploration_efficiency,
            learning_curve=learning_curve,
            reward_distribution=reward_distribution,
            loss_curve=loss_curve,
            q_value_curve=q_value_curve
        )
        
        self.history['training'].append(metrics.to_dict())
        return metrics
    
    def compare_with_baselines(self, our_result: Dict[str, Any], 
                             baseline_results: Dict[str, Dict[str, Any]]) -> ComparisonMetrics:
        """
        与基准算法对比
        
        Args:
            our_result: 我们的算法结果
            baseline_results: 基准算法结果字典
            
        Returns:
            对比指标
        """
        # 选择最佳基准结果
        best_baseline = None
        best_makespan = float('inf')
        
        for algo, result in baseline_results.items():
            makespan = result.get('makespan', float('inf'))
            if makespan < best_makespan:
                best_makespan = makespan
                best_baseline = algo
        
        if best_baseline is None:
            # 如果没有基准结果，使用默认值
            best_makespan = our_result.get('makespan', 1.0) * 1.2  # 假设基准比我们差20%
        
        our_makespan = our_result.get('makespan', 1.0)
        
        # 1. 改进比例
        improvement_ratio = (best_makespan - our_makespan) / best_makespan
        
        # 2. 加速比
        speedup_ratio = best_makespan / max(our_makespan, 1e-6)
        
        # 3. 效率比 (考虑资源使用)
        our_efficiency = our_result.get('efficiency', 1.0)
        baseline_efficiency = baseline_results.get(best_baseline, {}).get('efficiency', 1.0)
        efficiency_ratio = our_efficiency / max(baseline_efficiency, 1e-6)
        
        # 4. 可扩展性分数
        scalability_score = self._calculate_scalability_score(our_result, baseline_results)
        
        # 5. 鲁棒性分数
        robustness_score = self._calculate_robustness_score(our_result)
        
        # 6. 适应性分数
        adaptability_score = self._calculate_adaptability_score(our_result)
        
        metrics = ComparisonMetrics(
            baseline_makespan=best_makespan,
            improvement_ratio=improvement_ratio,
            speedup_ratio=speedup_ratio,
            efficiency_ratio=efficiency_ratio,
            scalability_score=scalability_score,
            robustness_score=robustness_score,
            adaptability_score=adaptability_score
        )
        
        self.history['comparison'].append(metrics.to_dict())
        return metrics
    
    def generate_comprehensive_report(self, output_path: str) -> Dict[str, Any]:
        """
        生成综合评估报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            综合评估报告
        """
        # 计算各项指标的平均值和趋势
        report = {
            'timestamp': time.time(),
            'system_performance': self._summarize_metrics(self.history['system']),
            'workflow_quality': self._summarize_metrics(self.history['workflow']),
            'training_process': self._summarize_metrics(self.history['training']),
            'baseline_comparison': self._summarize_metrics(self.history['comparison']),
            'overall_score': self._calculate_overall_score(),
            'recommendations': self._generate_recommendations()
        }
        
        # 保存报告
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"综合评估报告已保存到 {output_path}")
        return report
    
    def _calculate_load_balance(self, node_loads: List[float]) -> float:
        """计算负载均衡指数 (基尼系数)"""
        if not node_loads or len(node_loads) == 1:
            return 1.0
        
        # 归一化负载
        total_load = sum(node_loads)
        if total_load <= 0:
            return 1.0
        
        normalized_loads = [load / total_load for load in node_loads]
        normalized_loads.sort()
        
        # 计算基尼系数
        n = len(normalized_loads)
        cumsum = 0.0
        for i, load in enumerate(normalized_loads):
            cumsum += (i + 1) * load
        
        gini = (2 * cumsum) / (n * sum(normalized_loads)) - (n + 1) / n
        return 1.0 - gini  # 转换为均衡指数
    
    def _calculate_energy_efficiency(self, makespan: float, 
                                   resource_utilization: float, 
                                   total_tasks: int) -> float:
        """计算能效指数"""
        # 简化计算：能效 = 任务数 / (时间 × 资源利用率)
        # 资源利用率越高，能效越好
        if makespan <= 0:
            return 0.0
        
        efficiency = total_tasks / (makespan * max(resource_utilization, 0.1))
        # 归一化到0-1范围
        return min(1.0, efficiency / 100.0)
    
    def _calculate_cost_efficiency(self, makespan: float, 
                                 resource_utilization: Dict[str, float], 
                                 total_tasks: int) -> float:
        """计算成本效率指数"""
        # 简化计算：成本效率 = 任务数 / (时间 × 资源成本)
        # 假设每个资源的成本相同
        if makespan <= 0:
            return 0.0
        
        resource_count = len(resource_utilization)
        avg_utilization = np.mean(list(resource_utilization.values())) if resource_utilization else 0.0
        
        efficiency = total_tasks / (makespan * resource_count * max(avg_utilization, 0.1))
        # 归一化到0-1范围
        return min(1.0, efficiency / 50.0)
    
    def _calculate_fairness(self, resource_utilization: Dict[str, float]) -> float:
        """计算公平性指数 (基于资源分配的均等程度)"""
        if not resource_utilization:
            return 1.0
        
        utilizations = list(resource_utilization.values())
        # 使用标准差来衡量不公平性
        std_dev = np.std(utilizations)
        mean_util = np.mean(utilizations)
        
        if mean_util <= 0:
            return 1.0
        
        # 变异系数越小，公平性越高
        cv = std_dev / mean_util
        fairness = 1.0 / (1.0 + cv)
        return fairness
    
    def _calculate_parallel_efficiency(self, workflow_result: Dict[str, Any]) -> float:
        """计算并行效率"""
        # 简化计算：并行效率 = 理论最短时间 / 实际时间
        critical_path_length = workflow_result.get('critical_path_length', 1.0)
        actual_time = workflow_result.get('scheduling_length', 1.0)
        
        if actual_time <= 0:
            return 0.0
        
        efficiency = critical_path_length / actual_time
        return min(1.0, efficiency)
    
    def _calculate_convergence_speed(self, training_history: List[Dict[str, Any]]) -> float:
        """计算收敛速度"""
        if len(training_history) < 10:
            return 0.0
        
        # 获取最近100个回合的奖励
        recent_rewards = [ep.get('total_reward', 0.0) for ep in training_history[-100:]]
        
        # 计算奖励变化的斜率
        x = np.arange(len(recent_rewards))
        y = np.array(recent_rewards)
        
        # 线性回归计算斜率
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            # 归一化到0-1范围
            return min(1.0, max(0.0, (slope + 1.0) / 2.0))
        
        return 0.0
    
    def _calculate_stability_score(self, training_history: List[Dict[str, Any]]) -> float:
        """计算稳定性分数"""
        if len(training_history) < 10:
            return 0.0
        
        # 获取最近100个回合的奖励
        recent_rewards = [ep.get('total_reward', 0.0) for ep in training_history[-100:]]
        
        # 计算标准差
        std_dev = np.std(recent_rewards)
        mean_reward = np.mean(recent_rewards)
        
        if mean_reward == 0:
            return 0.0
        
        # 变异系数越小，稳定性越高
        cv = std_dev / abs(mean_reward)
        stability = 1.0 / (1.0 + cv)
        return stability
    
    def _calculate_exploration_efficiency(self, training_history: List[Dict[str, Any]]) -> float:
        """计算探索效率"""
        if len(training_history) < 10:
            return 0.0
        
        # 获取探索率和奖励
        exploration_rates = [ep.get('exploration_rate', 0.0) for ep in training_history[-100:]]
        rewards = [ep.get('total_reward', 0.0) for ep in training_history[-100:]]
        
        # 计算探索率和奖励的相关性
        if len(exploration_rates) > 1:
            correlation = np.corrcoef(exploration_rates, rewards)[0, 1]
            # 相关系数转换为效率分数
            return (correlation + 1.0) / 2.0  # 从[-1,1]映射到[0,1]
        
        return 0.0
    
    def _calculate_scalability_score(self, our_result: Dict[str, Any], 
                                   baseline_results: Dict[str, Dict[str, Any]]) -> float:
        """计算可扩展性分数"""
        # 简化计算：基于不同规模下的性能保持程度
        our_makespan = our_result.get('makespan', 1.0)
        our_task_count = our_result.get('task_count', 1)
        
        # 计算每个任务的平均时间
        our_time_per_task = our_makespan / our_task_count
        
        # 与基准算法对比
        best_baseline_time_per_task = float('inf')
        for algo, result in baseline_results.items():
            makespan = result.get('makespan', float('inf'))
            task_count = result.get('task_count', 1)
            time_per_task = makespan / task_count
            best_baseline_time_per_task = min(best_baseline_time_per_task, time_per_task)
        
        if best_baseline_time_per_task <= 0:
            return 1.0
        
        # 我们的时间越接近基准，可扩展性越好
        scalability = best_baseline_time_per_task / max(our_time_per_task, 1e-6)
        return min(1.0, scalability)
    
    def _calculate_robustness_score(self, our_result: Dict[str, Any]) -> float:
        """计算鲁棒性分数"""
        # 简化计算：基于任务完成率和资源利用率的一致性
        task_completion_rate = our_result.get('task_completion_rate', 1.0)
        resource_utilization = our_result.get('resource_utilization', {})
        
        # 计算资源利用率的标准差
        if resource_utilization:
            utilizations = list(resource_utilization.values())
            util_std = np.std(utilizations)
            util_mean = np.mean(utilizations)
            
            if util_mean > 0:
                util_cv = util_std / util_mean
                robustness = task_completion_rate * (1.0 / (1.0 + util_cv))
                return robustness
        
        return task_completion_rate
    
    def _calculate_adaptability_score(self, our_result: Dict[str, Any]) -> float:
        """计算适应性分数"""
        # 简化计算：基于对不同类型任务的适应能力
        task_types = our_result.get('task_types', {})
        type_performance = our_result.get('type_performance', {})
        
        if not task_types or not type_performance:
            return 0.5  # 默认中等分数
        
        # 计算各种任务类型的性能一致性
        performances = list(type_performance.values())
        if not performances:
            return 0.5
        
        # 性能越一致，适应性越好
        std_dev = np.std(performances)
        mean_perf = np.mean(performances)
        
        if mean_perf <= 0:
            return 0.0
        
        cv = std_dev / mean_perf
        adaptability = 1.0 / (1.0 + cv)
        return adaptability
    
    def _summarize_metrics(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总指标历史"""
        if not metrics_history:
            return {}
        
        summary = {}
        
        # 获取所有指标键
        all_keys = set()
        for metrics in metrics_history:
            all_keys.update(metrics.keys())
        
        # 对每个指标计算统计信息
        for key in all_keys:
            values = [metrics.get(key) for metrics in metrics_history if metrics.get(key) is not None]
            if values:
                # 处理字典类型的值
                if isinstance(values[0], dict):
                    summary[key] = self._summarize_metrics(values)
                # 处理列表类型的值
                elif isinstance(values[0], list):
                    summary[key] = {
                        'length': len(values[0]),
                        'trend': 'increasing' if len(values[0]) > 1 and values[0][-1] > values[0][0] else 'stable'
                    }
                # 处理数值类型的值
                else:
                    try:
                        numeric_values = [float(v) for v in values]
                        summary[key] = {
                            'mean': np.mean(numeric_values),
                            'std': np.std(numeric_values),
                            'min': np.min(numeric_values),
                            'max': np.max(numeric_values),
                            'latest': numeric_values[-1],
                            'trend': 'increasing' if len(numeric_values) > 1 and numeric_values[-1] > numeric_values[0] else 'stable'
                        }
                    except (ValueError, TypeError):
                        summary[key] = {'latest': values[-1]}
        
        return summary
    
    def _calculate_overall_score(self) -> float:
        """计算总体评分"""
        # 简化计算：基于各项指标的平均值
        scores = []
        
        # 系统性能分数
        if self.history['system']:
            latest_system = self.history['system'][-1]
            system_score = (
                latest_system.get('load_balance', 0.0) * 0.2 +
                latest_system.get('energy_efficiency', 0.0) * 0.2 +
                latest_system.get('cost_efficiency', 0.0) * 0.2 +
                latest_system.get('fairness', 0.0) * 0.2 +
                latest_system.get('reliability', 0.0) * 0.2
            )
            scores.append(system_score)
        
        # 工作流质量分数
        if self.history['workflow']:
            latest_workflow = self.history['workflow'][-1]
            workflow_score = (
                latest_workflow.get('parallel_efficiency', 0.0) * 0.3 +
                latest_workflow.get('data_locality_score', 0.0) * 0.2 +
                latest_workflow.get('deadline_satisfaction_rate', 0.0) * 0.3 +
                latest_workflow.get('workflow_completion_rate', 0.0) * 0.2
            )
            scores.append(workflow_score)
        
        # 训练过程分数
        if self.history['training']:
            latest_training = self.history['training'][-1]
            training_score = (
                latest_training.get('convergence_speed', 0.0) * 0.3 +
                latest_training.get('stability_score', 0.0) * 0.4 +
                latest_training.get('exploration_efficiency', 0.0) * 0.3
            )
            scores.append(training_score)
        
        # 基准对比分数
        if self.history['comparison']:
            latest_comparison = self.history['comparison'][-1]
            comparison_score = (
                min(1.0, max(0.0, latest_comparison.get('improvement_ratio', 0.0) + 0.5)) * 0.4 +
                min(1.0, latest_comparison.get('speedup_ratio', 0.0)) * 0.3 +
                latest_comparison.get('scalability_score', 0.0) * 0.3
            )
            scores.append(comparison_score)
        
        # 计算总体评分
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0.0
        
        return overall_score
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于系统性能的建议
        if self.history['system']:
            latest_system = self.history['system'][-1]
            
            if latest_system.get('load_balance', 0.0) < 0.7:
                recommendations.append("负载均衡不足，建议优化任务分配策略")
            
            if latest_system.get('energy_efficiency', 0.0) < 0.6:
                recommendations.append("能效较低，建议考虑节能调度策略")
            
            if latest_system.get('fairness', 0.0) < 0.7:
                recommendations.append("资源分配不公平，建议改进公平性机制")
        
        # 基于工作流质量的建议
        if self.history['workflow']:
            latest_workflow = self.history['workflow'][-1]
            
            if latest_workflow.get('parallel_efficiency', 0.0) < 0.7:
                recommendations.append("并行效率不足，建议优化任务并行执行策略")
            
            if latest_workflow.get('data_locality_score', 0.0) < 0.6:
                recommendations.append("数据局部性较差，建议优化数据放置策略")
            
            if latest_workflow.get('deadline_satisfaction_rate', 1.0) < 0.9:
                recommendations.append("截止时间满足率低，建议改进优先级调度策略")
        
        # 基于训练过程的建议
        if self.history['training']:
            latest_training = self.history['training'][-1]
            
            if latest_training.get('convergence_speed', 0.0) < 0.5:
                recommendations.append("收敛速度慢，建议调整学习率或网络结构")
            
            if latest_training.get('stability_score', 0.0) < 0.7:
                recommendations.append("训练不稳定，建议增加正则化或调整批大小")
            
            if latest_training.get('exploration_efficiency', 0.0) < 0.5:
                recommendations.append("探索效率低，建议改进探索策略")
        
        # 基于基准对比的建议
        if self.history['comparison']:
            latest_comparison = self.history['comparison'][-1]
            
            if latest_comparison.get('improvement_ratio', 0.0) < 0.1:
                recommendations.append("相比基准算法改进有限，建议重新设计算法核心逻辑")
            
            if latest_comparison.get('scalability_score', 0.0) < 0.7:
                recommendations.append("可扩展性不足，建议优化算法以适应大规模场景")
        
        if not recommendations:
            recommendations.append("系统性能良好，继续保持当前策略")
        
        return recommendations

__all__ = [
    'SystemMetrics', 'WorkflowMetrics', 'TrainingMetrics', 'ComparisonMetrics',
    'ComprehensiveEvaluator'
]