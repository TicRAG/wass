"""
评估报告生成器

整合评估指标和可视化结果，生成综合评估报告。
"""
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .evaluation_metrics import (
    SystemMetrics, WorkflowMetrics, TrainingMetrics, ComparisonMetrics,
    ComprehensiveEvaluator
)
from .evaluation_visualizer import EvaluationVisualizer

logger = logging.getLogger(__name__)

class EvaluationReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化评估器和可视化器
        self.evaluator = ComprehensiveEvaluator()
        self.visualizer = EvaluationVisualizer(str(self.output_dir))
        
        # 报告数据
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_performance': {},
            'workflow_quality': {},
            'training_process': {},
            'baseline_comparison': {},
            'recommendations': [],
            'overall_score': 0.0
        }
    
    def generate_comprehensive_report(self, 
                                    system_metrics_history: List[Dict[str, Any]],
                                    workflow_metrics_history: List[Dict[str, Any]],
                                    training_metrics_history: List[Dict[str, Any]],
                                    comparison_metrics_history: List[Dict[str, Any]],
                                    baseline_data: Optional[Dict[str, Any]] = None,
                                    save_path: Optional[str] = None) -> str:
        """
        生成综合评估报告
        
        Args:
            system_metrics_history: 系统性能指标历史
            workflow_metrics_history: 工作流调度质量指标历史
            training_metrics_history: 训练过程指标历史
            comparison_metrics_history: 基准对比指标历史
            baseline_data: 基准数据
            save_path: 保存路径
            
        Returns:
            生成的报告路径
        """
        logger.info("开始生成综合评估报告...")
        
        # 1. 分析系统性能指标
        self._analyze_system_metrics(system_metrics_history)
        
        # 2. 分析工作流调度质量指标
        self._analyze_workflow_metrics(workflow_metrics_history)
        
        # 3. 分析训练过程指标
        self._analyze_training_metrics(training_metrics_history)
        
        # 4. 分析基准对比指标
        self._analyze_comparison_metrics(comparison_metrics_history, baseline_data)
        
        # 5. 计算总体评分
        self._calculate_overall_score()
        
        # 6. 生成改进建议
        self._generate_recommendations()
        
        # 7. 生成可视化图表
        self._generate_visualizations()
        
        # 8. 保存报告数据
        report_data_path = self.output_dir / "report_data.json"
        with open(report_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, ensure_ascii=False, indent=2)
        
        # 9. 生成HTML报告
        html_report_path = self.visualizer.generate_summary_report(
            self.report_data, 
            str(self.output_dir / "evaluation_report.html")
        )
        
        # 10. 生成交互式仪表板
        dashboard_path = self.visualizer.create_interactive_dashboard(
            self.report_data,
            str(self.output_dir / "interactive_dashboard.html")
        )
        
        logger.info(f"综合评估报告已生成，保存到 {html_report_path}")
        return html_report_path
    
    def _analyze_system_metrics(self, metrics_history: List[Dict[str, Any]]):
        """分析系统性能指标"""
        if not metrics_history:
            logger.warning("没有系统性能指标数据可供分析")
            return
        
        # 计算最新指标
        latest_metrics = metrics_history[-1]
        
        # 计算历史统计
        df = pd.DataFrame(metrics_history)
        
        # 更新报告数据
        self.report_data['system_performance'] = {
            'makespan': {
                'latest': latest_metrics.get('makespan', 0),
                'mean': df['makespan'].mean() if 'makespan' in df.columns else 0,
                'std': df['makespan'].std() if 'makespan' in df.columns else 0,
                'min': df['makespan'].min() if 'makespan' in df.columns else 0,
                'max': df['makespan'].max() if 'makespan' in df.columns else 0
            },
            'load_balance': {
                'latest': latest_metrics.get('load_balance', 0),
                'mean': df['load_balance'].mean() if 'load_balance' in df.columns else 0,
                'std': df['load_balance'].std() if 'load_balance' in df.columns else 0
            },
            'resource_utilization': {
                'latest': latest_metrics.get('resource_utilization', {}),
                'mean': self._calculate_dict_mean(df, 'resource_utilization') if 'resource_utilization' in df.columns else {}
            },
            'throughput': {
                'latest': latest_metrics.get('throughput', 0),
                'mean': df['throughput'].mean() if 'throughput' in df.columns else 0,
                'std': df['throughput'].std() if 'throughput' in df.columns else 0
            },
            'energy_efficiency': {
                'latest': latest_metrics.get('energy_efficiency', 0),
                'mean': df['energy_efficiency'].mean() if 'energy_efficiency' in df.columns else 0
            },
            'cost_efficiency': {
                'latest': latest_metrics.get('cost_efficiency', 0),
                'mean': df['cost_efficiency'].mean() if 'cost_efficiency' in df.columns else 0
            },
            'fairness': {
                'latest': latest_metrics.get('fairness', 0),
                'mean': df['fairness'].mean() if 'fairness' in df.columns else 0
            },
            'reliability': {
                'latest': latest_metrics.get('reliability', 0),
                'mean': df['reliability'].mean() if 'reliability' in df.columns else 0
            }
        }
        
        # 生成系统性能图表
        self.visualizer.plot_system_metrics(metrics_history)
    
    def _analyze_workflow_metrics(self, metrics_history: List[Dict[str, Any]]):
        """分析工作流调度质量指标"""
        if not metrics_history:
            logger.warning("没有工作流调度质量指标数据可供分析")
            return
        
        # 计算最新指标
        latest_metrics = metrics_history[-1]
        
        # 计算历史统计
        df = pd.DataFrame(metrics_history)
        
        # 更新报告数据
        self.report_data['workflow_quality'] = {
            'scheduling_length': {
                'latest': latest_metrics.get('scheduling_length', 0),
                'mean': df['scheduling_length'].mean() if 'scheduling_length' in df.columns else 0,
                'std': df['scheduling_length'].std() if 'scheduling_length' in df.columns else 0
            },
            'critical_path_ratio': {
                'latest': latest_metrics.get('critical_path_ratio', 0),
                'mean': df['critical_path_ratio'].mean() if 'critical_path_ratio' in df.columns else 0
            },
            'parallel_efficiency': {
                'latest': latest_metrics.get('parallel_efficiency', 0),
                'mean': df['parallel_efficiency'].mean() if 'parallel_efficiency' in df.columns else 0,
                'std': df['parallel_efficiency'].std() if 'parallel_efficiency' in df.columns else 0
            },
            'data_locality_score': {
                'latest': latest_metrics.get('data_locality_score', 0),
                'mean': df['data_locality_score'].mean() if 'data_locality_score' in df.columns else 0
            },
            'communication_overhead': {
                'latest': latest_metrics.get('communication_overhead', 0),
                'mean': df['communication_overhead'].mean() if 'communication_overhead' in df.columns else 0
            },
            'deadline_satisfaction_rate': {
                'latest': latest_metrics.get('deadline_satisfaction_rate', 0),
                'mean': df['deadline_satisfaction_rate'].mean() if 'deadline_satisfaction_rate' in df.columns else 0
            },
            'priority_violation_rate': {
                'latest': latest_metrics.get('priority_violation_rate', 0),
                'mean': df['priority_violation_rate'].mean() if 'priority_violation_rate' in df.columns else 0
            },
            'workflow_completion_rate': {
                'latest': latest_metrics.get('workflow_completion_rate', 0),
                'mean': df['workflow_completion_rate'].mean() if 'workflow_completion_rate' in df.columns else 0
            }
        }
        
        # 生成工作流调度质量图表
        self.visualizer.plot_workflow_metrics(metrics_history)
    
    def _analyze_training_metrics(self, metrics_history: List[Dict[str, Any]]):
        """分析训练过程指标"""
        if not metrics_history:
            logger.warning("没有训练过程指标数据可供分析")
            return
        
        # 计算最新指标
        latest_metrics = metrics_history[-1]
        
        # 计算历史统计
        df = pd.DataFrame(metrics_history)
        
        # 更新报告数据
        self.report_data['training_process'] = {
            'convergence_speed': {
                'latest': latest_metrics.get('convergence_speed', 0),
                'mean': df['convergence_speed'].mean() if 'convergence_speed' in df.columns else 0
            },
            'stability_score': {
                'latest': latest_metrics.get('stability_score', 0),
                'mean': df['stability_score'].mean() if 'stability_score' in df.columns else 0
            },
            'exploration_efficiency': {
                'latest': latest_metrics.get('exploration_efficiency', 0),
                'mean': df['exploration_efficiency'].mean() if 'exploration_efficiency' in df.columns else 0
            },
            'learning_curve': {
                'latest': latest_metrics.get('learning_curve', []),
                'mean': self._calculate_list_mean(df, 'learning_curve') if 'learning_curve' in df.columns else []
            },
            'loss_curve': {
                'latest': latest_metrics.get('loss_curve', []),
                'mean': self._calculate_list_mean(df, 'loss_curve') if 'loss_curve' in df.columns else []
            },
            'q_value_curve': {
                'latest': latest_metrics.get('q_value_curve', []),
                'mean': self._calculate_list_mean(df, 'q_value_curve') if 'q_value_curve' in df.columns else []
            }
        }
        
        # 生成训练过程图表
        self.visualizer.plot_training_metrics(metrics_history)
    
    def _analyze_comparison_metrics(self, metrics_history: List[Dict[str, Any]], 
                                   baseline_data: Optional[Dict[str, Any]] = None):
        """分析基准对比指标"""
        if not metrics_history:
            logger.warning("没有基准对比指标数据可供分析")
            return
        
        # 计算最新指标
        latest_metrics = metrics_history[-1]
        
        # 计算历史统计
        df = pd.DataFrame(metrics_history)
        
        # 更新报告数据
        self.report_data['baseline_comparison'] = {
            'baseline_makespan': {
                'latest': latest_metrics.get('baseline_makespan', 0),
                'mean': df['baseline_makespan'].mean() if 'baseline_makespan' in df.columns else 0
            },
            'improvement_ratio': {
                'latest': latest_metrics.get('improvement_ratio', 0),
                'mean': df['improvement_ratio'].mean() if 'improvement_ratio' in df.columns else 0,
                'std': df['improvement_ratio'].std() if 'improvement_ratio' in df.columns else 0
            },
            'speedup_ratio': {
                'latest': latest_metrics.get('speedup_ratio', 0),
                'mean': df['speedup_ratio'].mean() if 'speedup_ratio' in df.columns else 0
            },
            'efficiency_ratio': {
                'latest': latest_metrics.get('efficiency_ratio', 0),
                'mean': df['efficiency_ratio'].mean() if 'efficiency_ratio' in df.columns else 0
            },
            'scalability_score': {
                'latest': latest_metrics.get('scalability_score', 0),
                'mean': df['scalability_score'].mean() if 'scalability_score' in df.columns else 0
            },
            'robustness_score': {
                'latest': latest_metrics.get('robustness_score', 0),
                'mean': df['robustness_score'].mean() if 'robustness_score' in df.columns else 0
            },
            'adaptability_score': {
                'latest': latest_metrics.get('adaptability_score', 0),
                'mean': df['adaptability_score'].mean() if 'adaptability_score' in df.columns else 0
            }
        }
        
        # 添加基准算法数据
        if baseline_data:
            self.report_data['baseline_comparison']['baseline_algorithms'] = baseline_data
        
        # 生成基准对比图表
        self.visualizer.plot_comparison_metrics(metrics_history)
    
    def _calculate_overall_score(self):
        """计算总体评分"""
        # 获取各部分的最新评分
        system_score = self._calculate_system_score()
        workflow_score = self._calculate_workflow_score()
        training_score = self._calculate_training_score()
        comparison_score = self._calculate_comparison_score()
        
        # 计算加权总分
        weights = {
            'system': 0.4,      # 系统性能权重
            'workflow': 0.3,    # 工作流调度质量权重
            'training': 0.2,    # 训练过程权重
            'comparison': 0.1   # 基准对比权重
        }
        
        overall_score = (
            system_score * weights['system'] +
            workflow_score * weights['workflow'] +
            training_score * weights['training'] +
            comparison_score * weights['comparison']
        )
        
        self.report_data['overall_score'] = overall_score
        self.report_data['component_scores'] = {
            'system_performance': system_score,
            'workflow_quality': workflow_score,
            'training_process': training_score,
            'baseline_comparison': comparison_score
        }
    
    def _calculate_system_score(self) -> float:
        """计算系统性能评分"""
        system_metrics = self.report_data['system_performance']
        
        # 关键指标
        key_metrics = [
            ('load_balance', 0.2),
            ('resource_utilization', 0.2),
            ('throughput', 0.15),
            ('energy_efficiency', 0.15),
            ('cost_efficiency', 0.15),
            ('fairness', 0.075),
            ('reliability', 0.075)
        ]
        
        score = 0.0
        for metric, weight in key_metrics:
            if metric in system_metrics:
                value = system_metrics[metric].get('latest', 0)
                score += value * weight
        
        return min(score, 1.0)  # 确保不超过1.0
    
    def _calculate_workflow_score(self) -> float:
        """计算工作流调度质量评分"""
        workflow_metrics = self.report_data['workflow_quality']
        
        # 关键指标
        key_metrics = [
            ('parallel_efficiency', 0.25),
            ('data_locality_score', 0.25),
            ('deadline_satisfaction_rate', 0.2),
            ('workflow_completion_rate', 0.2),
            ('communication_overhead', 0.1)  # 这个是开销，越小越好
        ]
        
        score = 0.0
        for metric, weight in key_metrics:
            if metric in workflow_metrics:
                value = workflow_metrics[metric].get('latest', 0)
                if metric == 'communication_overhead':
                    # 通信开销越小越好，这里假设最大开销为100秒
                    value = max(0, 1 - value / 100)
                score += value * weight
        
        return min(score, 1.0)  # 确保不超过1.0
    
    def _calculate_training_score(self) -> float:
        """计算训练过程评分"""
        training_metrics = self.report_data['training_process']
        
        # 关键指标
        key_metrics = [
            ('convergence_speed', 0.4),
            ('stability_score', 0.4),
            ('exploration_efficiency', 0.2)
        ]
        
        score = 0.0
        for metric, weight in key_metrics:
            if metric in training_metrics:
                value = training_metrics[metric].get('latest', 0)
                score += value * weight
        
        return min(score, 1.0)  # 确保不超过1.0
    
    def _calculate_comparison_score(self) -> float:
        """计算基准对比评分"""
        comparison_metrics = self.report_data['baseline_comparison']
        
        # 关键指标
        key_metrics = [
            ('improvement_ratio', 0.4),
            ('speedup_ratio', 0.2),
            ('efficiency_ratio', 0.2),
            ('scalability_score', 0.1),
            ('robustness_score', 0.05),
            ('adaptability_score', 0.05)
        ]
        
        score = 0.0
        for metric, weight in key_metrics:
            if metric in comparison_metrics:
                value = comparison_metrics[metric].get('latest', 0)
                score += value * weight
        
        return min(score, 1.0)  # 确保不超过1.0
    
    def _generate_recommendations(self):
        """生成改进建议"""
        recommendations = []
        
        # 系统性能建议
        system_metrics = self.report_data['system_performance']
        if system_metrics.get('load_balance', {}).get('latest', 0) < 0.7:
            recommendations.append("系统负载均衡指数较低，建议优化任务分配策略，提高资源利用率")
        
        if system_metrics.get('resource_utilization', {}).get('latest', {}):
            util_dict = system_metrics['resource_utilization']['latest']
            if isinstance(util_dict, dict):
                avg_util = np.mean(list(util_dict.values()))
                if avg_util < 0.6:
                    recommendations.append("资源利用率较低，建议增加任务并行度或优化资源分配")
        
        if system_metrics.get('energy_efficiency', {}).get('latest', 0) < 0.7:
            recommendations.append("系统能效指数较低，建议优化能耗管理策略，减少空闲资源能耗")
        
        # 工作流调度质量建议
        workflow_metrics = self.report_data['workflow_quality']
        if workflow_metrics.get('parallel_efficiency', {}).get('latest', 0) < 0.7:
            recommendations.append("工作流并行效率较低，建议优化任务依赖关系和并行执行策略")
        
        if workflow_metrics.get('data_locality_score', {}).get('latest', 0) < 0.7:
            recommendations.append("数据局部性分数较低，建议优化数据放置策略，减少数据传输开销")
        
        if workflow_metrics.get('deadline_satisfaction_rate', {}).get('latest', 0) < 0.8:
            recommendations.append("截止时间满足率较低，建议优化任务优先级调度策略")
        
        # 训练过程建议
        training_metrics = self.report_data['training_process']
        if training_metrics.get('convergence_speed', {}).get('latest', 0) < 0.6:
            recommendations.append("训练收敛速度较慢，建议调整学习率或优化网络结构")
        
        if training_metrics.get('stability_score', {}).get('latest', 0) < 0.7:
            recommendations.append("训练稳定性较差，建议增加正则化或调整探索策略")
        
        if training_metrics.get('exploration_efficiency', {}).get('latest', 0) < 0.6:
            recommendations.append("探索效率较低，建议优化探索策略，平衡探索与利用")
        
        # 基准对比建议
        comparison_metrics = self.report_data['baseline_comparison']
        if comparison_metrics.get('improvement_ratio', {}).get('latest', 0) < 0.1:
            recommendations.append("相比基准算法改进较小，建议重新评估算法设计或参数配置")
        
        if comparison_metrics.get('scalability_score', {}).get('latest', 0) < 0.7:
            recommendations.append("算法可扩展性较差，建议优化算法在大规模场景下的性能")
        
        # 总体建议
        overall_score = self.report_data['overall_score']
        if overall_score < 0.6:
            recommendations.append("系统整体性能有待提升，建议全面评估和优化系统架构")
        elif overall_score < 0.8:
            recommendations.append("系统性能良好，但仍有优化空间，建议针对性改进薄弱环节")
        else:
            recommendations.append("系统性能优秀，建议保持当前配置并持续监控性能变化")
        
        self.report_data['recommendations'] = recommendations
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        # 图表已在前面各分析步骤中生成
        pass
    
    def _calculate_dict_mean(self, df: pd.DataFrame, column: str) -> Dict[str, float]:
        """计算字典列的平均值"""
        if column not in df.columns:
            return {}
        
        # 收集所有字典
        all_dicts = []
        for d in df[column]:
            if isinstance(d, dict):
                all_dicts.append(d)
        
        if not all_dicts:
            return {}
        
        # 计算每个键的平均值
        mean_dict = {}
        keys = set()
        for d in all_dicts:
            keys.update(d.keys())
        
        for key in keys:
            values = [d.get(key, 0) for d in all_dicts if key in d]
            if values:
                mean_dict[key] = np.mean(values)
        
        return mean_dict
    
    def _calculate_list_mean(self, df: pd.DataFrame, column: str) -> List[float]:
        """计算列表列的平均值"""
        if column not in df.columns:
            return []
        
        # 收集所有列表
        all_lists = []
        for lst in df[column]:
            if isinstance(lst, list):
                all_lists.append(lst)
        
        if not all_lists:
            return []
        
        # 找到最大长度
        max_len = max(len(lst) for lst in all_lists)
        
        # 计算每个位置的平均值
        mean_list = []
        for i in range(max_len):
            values = [lst[i] for lst in all_lists if i < len(lst)]
            if values:
                mean_list.append(np.mean(values))
        
        return mean_list

__all__ = ['EvaluationReportGenerator']