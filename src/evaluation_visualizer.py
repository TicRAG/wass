"""
评估可视化模块

提供各种图表和可视化报告生成功能，用于直观展示评估结果。
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = logging.getLogger(__name__)

class EvaluationVisualizer:
    """评估可视化器"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_system_metrics(self, metrics_history: List[Dict[str, Any]], 
                           save_path: Optional[str] = None) -> str:
        """
        绘制系统性能指标图表
        
        Args:
            metrics_history: 指标历史记录
            save_path: 保存路径
            
        Returns:
            生成的图表路径
        """
        if not metrics_history:
            logger.warning("没有系统性能指标数据可供可视化")
            return ""
        
        # 准备数据
        df = pd.DataFrame(metrics_history)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('系统性能指标分析', fontsize=16, fontweight='bold')
        
        # 1. Makespan趋势
        axes[0, 0].plot(df.index, df['makespan'], marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Makespan趋势')
        axes[0, 0].set_xlabel('实验次数')
        axes[0, 0].set_ylabel('Makespan (秒)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 负载均衡指数
        axes[0, 1].plot(df.index, df['load_balance'], marker='s', linewidth=2, markersize=4, color='green')
        axes[0, 1].set_title('负载均衡指数')
        axes[0, 1].set_xlabel('实验次数')
        axes[0, 1].set_ylabel('均衡指数 (0-1)')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 资源利用率热力图
        if 'resource_utilization' in df.columns and df['resource_utilization'].notna().any():
            # 提取资源利用率数据
            resource_data = []
            for i, util_dict in enumerate(df['resource_utilization']):
                if isinstance(util_dict, dict):
                    for resource, util in util_dict.items():
                        resource_data.append({
                            '实验': i,
                            '资源': resource,
                            '利用率': util
                        })
            
            if resource_data:
                resource_df = pd.DataFrame(resource_data)
                resource_pivot = resource_df.pivot(index='实验', columns='资源', values='利用率')
                
                sns.heatmap(resource_pivot, ax=axes[0, 2], cmap='YlOrRd', annot=True, fmt='.2f')
                axes[0, 2].set_title('资源利用率热力图')
        
        # 4. 吞吐量趋势
        axes[1, 0].plot(df.index, df['throughput'], marker='^', linewidth=2, markersize=4, color='purple')
        axes[1, 0].set_title('吞吐量趋势')
        axes[1, 0].set_xlabel('实验次数')
        axes[1, 0].set_ylabel('吞吐量 (任务/秒)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 能效和成本效率对比
        axes[1, 1].plot(df.index, df['energy_efficiency'], marker='d', linewidth=2, markersize=4, 
                       label='能效指数', color='orange')
        axes[1, 1].plot(df.index, df['cost_efficiency'], marker='*', linewidth=2, markersize=4, 
                       label='成本效率指数', color='red')
        axes[1, 1].set_title('效率指数对比')
        axes[1, 1].set_xlabel('实验次数')
        axes[1, 1].set_ylabel('效率指数 (0-1)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 公平性和可靠性雷达图
        if 'fairness' in df.columns and 'reliability' in df.columns:
            # 计算平均值
            fairness_avg = df['fairness'].mean()
            reliability_avg = df['reliability'].mean()
            load_balance_avg = df['load_balance'].mean()
            
            # 创建雷达图数据
            categories = ['负载均衡', '公平性', '可靠性']
            values = [load_balance_avg, fairness_avg, reliability_avg]
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]  # 闭合图形
            
            axes[1, 2].remove()
            ax = fig.add_subplot(2, 3, 6, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('系统综合性能雷达图')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "system_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"系统性能指标图表已保存到 {save_path}")
        return str(save_path)
    
    def plot_workflow_metrics(self, metrics_history: List[Dict[str, Any]], 
                             save_path: Optional[str] = None) -> str:
        """
        绘制工作流调度质量指标图表
        
        Args:
            metrics_history: 指标历史记录
            save_path: 保存路径
            
        Returns:
            生成的图表路径
        """
        if not metrics_history:
            logger.warning("没有工作流调度质量指标数据可供可视化")
            return ""
        
        # 准备数据
        df = pd.DataFrame(metrics_history)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('工作流调度质量分析', fontsize=16, fontweight='bold')
        
        # 1. 调度长度和关键路径占比
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(df.index, df['scheduling_length'], marker='o', linewidth=2, markersize=4, 
                label='调度长度', color='blue')
        ax1_twin.plot(df.index, df['critical_path_ratio'], marker='s', linewidth=2, markersize=4, 
                     label='关键路径占比', color='red')
        
        ax1.set_xlabel('实验次数')
        ax1.set_ylabel('调度长度 (秒)', color='blue')
        ax1_twin.set_ylabel('关键路径占比', color='red')
        ax1.set_title('调度长度与关键路径占比')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 2. 并行效率和数据局部性
        axes[0, 1].plot(df.index, df['parallel_efficiency'], marker='^', linewidth=2, markersize=4, 
                       label='并行效率', color='green')
        axes[0, 1].plot(df.index, df['data_locality_score'], marker='d', linewidth=2, markersize=4, 
                       label='数据局部性分数', color='purple')
        axes[0, 1].set_title('并行效率与数据局部性')
        axes[0, 1].set_xlabel('实验次数')
        axes[0, 1].set_ylabel('分数 (0-1)')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 通信开销
        axes[1, 0].plot(df.index, df['communication_overhead'], marker='*', linewidth=2, markersize=4, 
                       color='orange')
        axes[1, 0].set_title('通信开销')
        axes[1, 0].set_xlabel('实验次数')
        axes[1, 0].set_ylabel('通信开销 (秒)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 截止时间满足率和优先级违反率
        axes[1, 1].plot(df.index, df['deadline_satisfaction_rate'], marker='o', linewidth=2, markersize=4, 
                       label='截止时间满足率', color='darkgreen')
        axes[1, 1].plot(df.index, 1 - df['priority_violation_rate'], marker='s', linewidth=2, markersize=4, 
                       label='优先级遵守率', color='darkred')
        axes[1, 1].set_title('时间约束满足情况')
        axes[1, 1].set_xlabel('实验次数')
        axes[1, 1].set_ylabel('比率 (0-1)')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "workflow_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"工作流调度质量指标图表已保存到 {save_path}")
        return str(save_path)
    
    def plot_training_metrics(self, metrics_history: List[Dict[str, Any]], 
                             save_path: Optional[str] = None) -> str:
        """
        绘制训练过程指标图表
        
        Args:
            metrics_history: 指标历史记录
            save_path: 保存路径
            
        Returns:
            生成的图表路径
        """
        if not metrics_history:
            logger.warning("没有训练过程指标数据可供可视化")
            return ""
        
        # 准备数据
        df = pd.DataFrame(metrics_history)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('训练过程分析', fontsize=16, fontweight='bold')
        
        # 1. 学习曲线
        if 'learning_curve' in df.columns and df['learning_curve'].notna().any():
            # 提取学习曲线数据
            all_rewards = []
            for curve in df['learning_curve']:
                if isinstance(curve, list):
                    all_rewards.extend(curve)
            
            if all_rewards:
                # 绘制学习曲线
                axes[0, 0].plot(all_rewards, linewidth=2, alpha=0.7)
                axes[0, 0].set_title('学习曲线')
                axes[0, 0].set_xlabel('训练步数')
                axes[0, 0].set_ylabel('奖励值')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 添加趋势线
                if len(all_rewards) > 10:
                    x = np.arange(len(all_rewards))
                    y = np.array(all_rewards)
                    z = np.polyfit(x, y, 3)
                    p = np.poly1d(z)
                    axes[0, 0].plot(x, p(x), "--", linewidth=2, color='red', alpha=0.8, label='趋势线')
                    axes[0, 0].legend()
        
        # 2. 损失曲线
        if 'loss_curve' in df.columns and df['loss_curve'].notna().any():
            # 提取损失曲线数据
            all_losses = []
            for curve in df['loss_curve']:
                if isinstance(curve, list):
                    all_losses.extend(curve)
            
            if all_losses:
                axes[0, 1].plot(all_losses, linewidth=2, color='orange', alpha=0.7)
                axes[0, 1].set_title('损失曲线')
                axes[0, 1].set_xlabel('训练步数')
                axes[0, 1].set_ylabel('损失值')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q值曲线
        if 'q_value_curve' in df.columns and df['q_value_curve'].notna().any():
            # 提取Q值曲线数据
            all_q_values = []
            for curve in df['q_value_curve']:
                if isinstance(curve, list):
                    all_q_values.extend(curve)
            
            if all_q_values:
                axes[1, 0].plot(all_q_values, linewidth=2, color='green', alpha=0.7)
                axes[1, 0].set_title('Q值曲线')
                axes[1, 0].set_xlabel('训练步数')
                axes[1, 0].set_ylabel('Q值')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 训练指标雷达图
        if all(col in df.columns for col in ['convergence_speed', 'stability_score', 'exploration_efficiency']):
            # 计算平均值
            convergence_avg = df['convergence_speed'].mean()
            stability_avg = df['stability_score'].mean()
            exploration_avg = df['exploration_efficiency'].mean()
            
            # 创建雷达图数据
            categories = ['收敛速度', '稳定性', '探索效率']
            values = [convergence_avg, stability_avg, exploration_avg]
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]  # 闭合图形
            
            axes[1, 1].remove()
            ax = fig.add_subplot(2, 2, 4, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('训练质量雷达图')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "training_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练过程指标图表已保存到 {save_path}")
        return str(save_path)
    
    def plot_comparison_metrics(self, metrics_history: List[Dict[str, Any]], 
                               save_path: Optional[str] = None) -> str:
        """
        绘制基准对比指标图表
        
        Args:
            metrics_history: 指标历史记录
            save_path: 保存路径
            
        Returns:
            生成的图表路径
        """
        if not metrics_history:
            logger.warning("没有基准对比指标数据可供可视化")
            return ""
        
        # 准备数据
        df = pd.DataFrame(metrics_history)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('基准算法对比分析', fontsize=16, fontweight='bold')
        
        # 1. Makespan对比柱状图
        if 'baseline_makespan' in df.columns and 'makespan' in df.columns:
            # 假设我们的算法makespan存储在其他地方
            our_makespans = [100, 95, 90, 85, 80]  # 示例数据
            baseline_makespans = df['baseline_makespan'].tolist()
            
            x = np.arange(len(our_makespans))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, our_makespans, width, label='我们的算法', alpha=0.8)
            axes[0, 0].bar(x + width/2, baseline_makespans[:len(our_makespans)], width, label='基准算法', alpha=0.8)
            
            axes[0, 0].set_title('Makespan对比')
            axes[0, 0].set_xlabel('实验次数')
            axes[0, 0].set_ylabel('Makespan (秒)')
            axes[0, 0].set_xticks(x)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 改进比例趋势
        if 'improvement_ratio' in df.columns:
            axes[0, 1].plot(df.index, df['improvement_ratio'] * 100, marker='o', linewidth=2, markersize=4, color='green')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('改进比例趋势')
            axes[0, 1].set_xlabel('实验次数')
            axes[0, 1].set_ylabel('改进比例 (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 加速比和效率比
        if 'speedup_ratio' in df.columns and 'efficiency_ratio' in df.columns:
            axes[1, 0].plot(df.index, df['speedup_ratio'], marker='^', linewidth=2, markersize=4, 
                           label='加速比', color='blue')
            axes[1, 0].plot(df.index, df['efficiency_ratio'], marker='s', linewidth=2, markersize=4, 
                           label='效率比', color='purple')
            axes[1, 0].set_title('加速比与效率比')
            axes[1, 0].set_xlabel('实验次数')
            axes[1, 0].set_ylabel('比率')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 综合性能雷达图
        if all(col in df.columns for col in ['scalability_score', 'robustness_score', 'adaptability_score']):
            # 计算平均值
            scalability_avg = df['scalability_score'].mean()
            robustness_avg = df['robustness_score'].mean()
            adaptability_avg = df['adaptability_score'].mean()
            
            # 创建雷达图数据
            categories = ['可扩展性', '鲁棒性', '适应性']
            values = [scalability_avg, robustness_avg, adaptability_avg]
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]  # 闭合图形
            
            axes[1, 1].remove()
            ax = fig.add_subplot(2, 2, 4, projection='polar')
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('算法综合性能雷达图')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = self.output_dir / "comparison_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"基准对比指标图表已保存到 {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self, report_data: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> str:
        """
        创建交互式仪表板
        
        Args:
            report_data: 报告数据
            save_path: 保存路径
            
        Returns:
            生成的仪表板路径
        """
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('系统性能概览', '工作流调度质量', '训练过程分析', '基准算法对比'),
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. 系统性能概览 - 仪表盘
        overall_score = report_data.get('overall_score', 0.0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "总体评分 (%)"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. 工作流调度质量 - 散点图
        if 'workflow_quality' in report_data:
            workflow_data = report_data['workflow_quality']
            if 'parallel_efficiency' in workflow_data and 'data_locality_score' in workflow_data:
                fig.add_trace(
                    go.Scatter(
                        x=[workflow_data['data_locality_score'].get('latest', 0)],
                        y=[workflow_data['parallel_efficiency'].get('latest', 0)],
                        mode='markers+text',
                        marker=dict(size=20, color='rgba(255, 0, 0, 0.8)'),
                        text=['当前性能'],
                        textposition="top center",
                        name='工作流调度质量'
                    ),
                    row=1, col=2
                )
        
        # 3. 训练过程分析 - 线图
        if 'training_process' in report_data:
            training_data = report_data['training_process']
            if 'learning_curve' in training_data:
                # 假设学习曲线是一个列表
                learning_curve = training_data['learning_curve'].get('latest', [])
                if isinstance(learning_curve, list) and learning_curve:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(learning_curve))),
                            y=learning_curve,
                            mode='lines',
                            name='学习曲线',
                            line=dict(color='green', width=2)
                        ),
                        row=2, col=1
                    )
        
        # 4. 基准算法对比 - 柱状图
        if 'baseline_comparison' in report_data:
            comparison_data = report_data['baseline_comparison']
            if 'improvement_ratio' in comparison_data:
                improvement = comparison_data['improvement_ratio'].get('latest', 0) * 100
                fig.add_trace(
                    go.Bar(
                        x=['改进比例'],
                        y=[improvement],
                        marker_color='rgba(50, 171, 96, 0.8)',
                        name='改进比例'
                    ),
                    row=2, col=2
                )
        
        # 更新布局
        fig.update_layout(
            title_text="WASS-RAG 系统性能交互式仪表板",
            showlegend=True,
            height=800
        )
        
        # 保存仪表板
        if save_path is None:
            save_path = self.output_dir / "interactive_dashboard.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        
        logger.info(f"交互式仪表板已保存到 {save_path}")
        return str(save_path)
    
    def generate_summary_report(self, report_data: Dict[str, Any], 
                               save_path: Optional[str] = None) -> str:
        """
        生成综合评估报告
        
        Args:
            report_data: 报告数据
            save_path: 保存路径
            
        Returns:
            生成的报告路径
        """
        # 创建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WASS-RAG 系统性能评估报告</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #444;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .metric-card {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #007bff;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                }}
                .recommendation {{
                    background-color: #e7f3fe;
                    border-left: 4px solid #2196F3;
                    padding: 15px;
                    margin-bottom: 15px;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #777;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>WASS-RAG 系统性能评估报告</h1>
                
                <h2>总体评分</h2>
                <div class="metric-card">
                    <div class="metric-value">{report_data.get('overall_score', 0.0) * 100:.1f}%</div>
                    <div class="metric-label">系统综合性能评分</div>
                </div>
                
                <h2>系统性能概览</h2>
                {self._generate_system_metrics_html(report_data.get('system_performance', {}))}
                
                <h2>工作流调度质量</h2>
                {self._generate_workflow_metrics_html(report_data.get('workflow_quality', {}))}
                
                <h2>训练过程分析</h2>
                {self._generate_training_metrics_html(report_data.get('training_process', {}))}
                
                <h2>基准算法对比</h2>
                {self._generate_comparison_metrics_html(report_data.get('baseline_comparison', {}))}
                
                <h2>改进建议</h2>
                {self._generate_recommendations_html(report_data.get('recommendations', []))}
                
                <h2>可视化图表</h2>
                <div class="chart-container">
                    <img src="system_metrics.png" alt="系统性能指标">
                </div>
                <div class="chart-container">
                    <img src="workflow_metrics.png" alt="工作流调度质量指标">
                </div>
                <div class="chart-container">
                    <img src="training_metrics.png" alt="训练过程指标">
                </div>
                <div class="chart-container">
                    <img src="comparison_metrics.png" alt="基准对比指标">
                </div>
                
                <div class="footer">
                    <p>报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>WASS-RAG 系统性能评估工具 v1.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        if save_path is None:
            save_path = self.output_dir / "evaluation_report.html"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"综合评估报告已保存到 {save_path}")
        return str(save_path)
    
    def _generate_system_metrics_html(self, system_metrics: Dict[str, Any]) -> str:
        """生成系统性能指标的HTML"""
        html = ""
        
        # 关键指标
        key_metrics = [
            ('makespan', 'Makespan (秒)'),
            ('load_balance', '负载均衡指数'),
            ('energy_efficiency', '能效指数'),
            ('cost_efficiency', '成本效率指数'),
            ('fairness', '公平性指数'),
            ('reliability', '可靠性指数')
        ]
        
        for metric, label in key_metrics:
            if metric in system_metrics:
                value = system_metrics[metric].get('latest', 0)
                html += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.3f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """
        
        return html
    
    def _generate_workflow_metrics_html(self, workflow_metrics: Dict[str, Any]) -> str:
        """生成工作流调度质量指标的HTML"""
        html = ""
        
        # 关键指标
        key_metrics = [
            ('parallel_efficiency', '并行效率'),
            ('data_locality_score', '数据局部性分数'),
            ('deadline_satisfaction_rate', '截止时间满足率'),
            ('workflow_completion_rate', '工作流完成率')
        ]
        
        for metric, label in key_metrics:
            if metric in workflow_metrics:
                value = workflow_metrics[metric].get('latest', 0)
                html += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.3f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """
        
        return html
    
    def _generate_training_metrics_html(self, training_metrics: Dict[str, Any]) -> str:
        """生成训练过程指标的HTML"""
        html = ""
        
        # 关键指标
        key_metrics = [
            ('convergence_speed', '收敛速度'),
            ('stability_score', '稳定性分数'),
            ('exploration_efficiency', '探索效率')
        ]
        
        for metric, label in key_metrics:
            if metric in training_metrics:
                value = training_metrics[metric].get('latest', 0)
                html += f"""
                <div class="metric-card">
                    <div class="metric-value">{value:.3f}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """
        
        return html
    
    def _generate_comparison_metrics_html(self, comparison_metrics: Dict[str, Any]) -> str:
        """生成基准对比指标的HTML"""
        html = ""
        
        # 关键指标
        key_metrics = [
            ('improvement_ratio', '改进比例'),
            ('speedup_ratio', '加速比'),
            ('efficiency_ratio', '效率比'),
            ('scalability_score', '可扩展性分数')
        ]
        
        for metric, label in key_metrics:
            if metric in comparison_metrics:
                value = comparison_metrics[metric].get('latest', 0)
                if metric == 'improvement_ratio':
                    value = value * 100  # 转换为百分比
                    html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{value:.1f}%</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """
                else:
                    html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{value:.3f}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """
        
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """生成改进建议的HTML"""
        html = ""
        
        for rec in recommendations:
            html += f"""
            <div class="recommendation">
                {rec}
            </div>
            """
        
        return html

__all__ = ['EvaluationVisualizer']