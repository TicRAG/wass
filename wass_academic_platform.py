"""
WASS完整学术研究平台

集成真实WRENCH仿真的完整workflow管理系统
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import logging
import time
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WASSAcademicPlatform:
    """
    WASS学术研究平台
    
    集成真实WRENCH仿真和完整的workflow管理
    """
    
    def __init__(self, config_path: str = "configs/experiment.yaml"):
        self.config_path = config_path
        self.config = None
        self.wrench_simulator = None
        self.results = {}
        
    def load_config(self):
        """加载配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"配置加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            # 使用默认配置
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'data': {
                'adapter': 'jsonl',
                'train_file': 'train.jsonl',
                'valid_file': 'valid.jsonl',
                'test_file': 'test.jsonl'
            },
            'simulation': {
                'enabled': True,
                'platform': 'wrench',
                'compute_hosts': ['compute_host_1'],
                'storage_hosts': ['storage_host']
            },
            'workflow': {
                'stages': ['data_prep', 'labeling', 'training', 'evaluation'],
                'parallel': True
            },
            'paths': {
                'data_dir': './data',
                'results_dir': './results'
            }
        }
    
    def initialize_wrench_simulator(self):
        """初始化WRENCH仿真器"""
        try:
            from wass_wrench_simulator import create_wass_wrench_simulator
            self.wrench_simulator = create_wass_wrench_simulator()
            
            if self.wrench_simulator.initialize():
                logger.info("🎉 WRENCH仿真器初始化成功")
                info = self.wrench_simulator.get_simulation_info()
                logger.info(f"WRENCH版本: {info.get('wrench_version', 'unknown')}")
                logger.info(f"主机数量: {info.get('host_count', 0)}")
                logger.info(f"Mock数据: {info.get('mock_data', True)}")
                return True
            else:
                logger.warning("WRENCH仿真器初始化失败，使用fallback模式")
                return False
                
        except Exception as e:
            logger.error(f"WRENCH仿真器加载失败: {e}")
            return False
    
    def create_academic_workflow(self) -> Dict:
        """创建学术研究工作流"""
        workflow = {
            'name': 'wass_academic_research_workflow',
            'description': 'WASS学术研究完整工作流',
            'created_at': datetime.now().isoformat(),
            'tasks': [
                {
                    'id': 'data_preprocessing',
                    'name': '数据预处理',
                    'flops': 2e9,  # 2 GFlops
                    'memory': 1e9,  # 1 GB
                    'dependencies': [],
                    'stage': 'data_prep'
                },
                {
                    'id': 'feature_extraction',
                    'name': '特征提取',
                    'flops': 5e9,  # 5 GFlops
                    'memory': 2e9,  # 2 GB  
                    'dependencies': ['data_preprocessing'],
                    'stage': 'labeling'
                },
                {
                    'id': 'label_function_execution',
                    'name': '标注函数执行',
                    'flops': 3e9,  # 3 GFlops
                    'memory': 1.5e9,  # 1.5 GB
                    'dependencies': ['feature_extraction'],
                    'stage': 'labeling'
                },
                {
                    'id': 'graph_construction',
                    'name': '图构建',
                    'flops': 4e9,  # 4 GFlops
                    'memory': 3e9,  # 3 GB
                    'dependencies': ['label_function_execution'],
                    'stage': 'training'
                },
                {
                    'id': 'gnn_training',
                    'name': 'GNN训练',
                    'flops': 15e9,  # 15 GFlops
                    'memory': 6e9,  # 6 GB
                    'dependencies': ['graph_construction'],
                    'stage': 'training'
                },
                {
                    'id': 'drl_policy_training',
                    'name': 'DRL策略训练',
                    'flops': 10e9,  # 10 GFlops
                    'memory': 4e9,  # 4 GB
                    'dependencies': ['gnn_training'],
                    'stage': 'training'
                },
                {
                    'id': 'model_evaluation',
                    'name': '模型评估',
                    'flops': 2e9,  # 2 GFlops
                    'memory': 2e9,  # 2 GB
                    'dependencies': ['drl_policy_training'],
                    'stage': 'evaluation'
                },
                {
                    'id': 'result_analysis',
                    'name': '结果分析',
                    'flops': 1e9,  # 1 GFlops
                    'memory': 1e9,  # 1 GB
                    'dependencies': ['model_evaluation'],
                    'stage': 'evaluation'
                }
            ]
        }
        
        return workflow
    
    def run_academic_simulation(self) -> Dict:
        """运行学术仿真"""
        logger.info("🚀 开始学术研究仿真")
        
        # 创建工作流
        workflow = self.create_academic_workflow()
        
        # 运行WRENCH仿真
        if self.wrench_simulator:
            simulation_result = self.wrench_simulator.run_simulation(workflow)
        else:
            simulation_result = self._fallback_simulation(workflow)
        
        # 分析仿真结果
        analysis = self._analyze_simulation_results(simulation_result, workflow)
        
        # 生成学术报告
        report = self._generate_academic_report(simulation_result, analysis)
        
        return {
            'workflow': workflow,
            'simulation_result': simulation_result,
            'analysis': analysis,
            'academic_report': report
        }
    
    def _analyze_simulation_results(self, sim_result: Dict, workflow: Dict) -> Dict:
        """分析仿真结果"""
        tasks = workflow.get('tasks', [])
        
        # 按阶段分组分析
        stages = {}
        for task in tasks:
            stage = task.get('stage', 'unknown')
            if stage not in stages:
                stages[stage] = {
                    'tasks': [],
                    'total_flops': 0,
                    'total_memory': 0
                }
            stages[stage]['tasks'].append(task)
            stages[stage]['total_flops'] += task.get('flops', 0)
            stages[stage]['total_memory'] += task.get('memory', 0)
        
        # 计算阶段执行时间（假设顺序执行阶段）
        stage_times = {}
        for stage_name, stage_data in stages.items():
            # 假设阶段内任务可以并行
            max_stage_flops = max(task.get('flops', 0) for task in stage_data['tasks'])
            stage_times[stage_name] = max_stage_flops / 2e9  # 假设2GFlops处理速度
        
        # 性能分析
        total_execution_time = sim_result.get('execution_time', 0)
        total_flops = sim_result.get('total_flops', 0)
        throughput = sim_result.get('throughput', 0)
        
        analysis = {
            'stage_analysis': {
                stage: {
                    'task_count': len(data['tasks']),
                    'computational_load': data['total_flops'],
                    'memory_requirement': data['total_memory'],
                    'estimated_time': stage_times.get(stage, 0)
                }
                for stage, data in stages.items()
            },
            'performance_metrics': {
                'total_execution_time': total_execution_time,
                'total_computational_load': total_flops,
                'system_throughput': throughput,
                'efficiency': sim_result.get('efficiency', 0),
                'cpu_utilization': sim_result.get('cpu_utilization', 0)
            },
            'resource_utilization': {
                'host_count': sim_result.get('host_count', 0),
                'memory_usage_ratio': sim_result.get('memory_usage', 0),
                'parallel_efficiency': sim_result.get('parallel_tasks', 0) / len(tasks) if tasks else 0
            }
        }
        
        return analysis
    
    def _generate_academic_report(self, sim_result: Dict, analysis: Dict) -> Dict:
        """生成学术报告"""
        is_real_wrench = not sim_result.get('mock_data', True)
        
        report = {
            'title': 'WASS学术研究平台仿真报告',
            'timestamp': datetime.now().isoformat(),
            'simulation_platform': {
                'type': sim_result.get('platform_type', 'unknown'),
                'method': sim_result.get('simulation_method', 'unknown'),
                'wrench_version': sim_result.get('wrench_version', 'unknown'),
                'real_wrench_integration': is_real_wrench
            },
            'workflow_summary': {
                'workflow_name': sim_result.get('workflow_id', 'unknown'),
                'total_tasks': sim_result.get('task_count', 0),
                'workflow_depth': sim_result.get('workflow_depth', 0),
                'critical_path': sim_result.get('critical_path', [])
            },
            'performance_results': {
                'execution_time_seconds': sim_result.get('execution_time', 0),
                'computational_throughput_flops': sim_result.get('throughput', 0),
                'system_efficiency_percent': sim_result.get('efficiency', 0) * 100,
                'cpu_utilization_percent': sim_result.get('cpu_utilization', 0) * 100
            },
            'infrastructure_details': {
                'compute_hosts': sim_result.get('hosts', []),
                'host_count': sim_result.get('host_count', 0),
                'memory_utilization_percent': sim_result.get('memory_usage', 0) * 100
            },
            'academic_insights': {
                'scalability_assessment': self._assess_scalability(analysis),
                'bottleneck_analysis': self._identify_bottlenecks(analysis),
                'optimization_recommendations': self._generate_recommendations(analysis)
            },
            'validation': {
                'simulation_validity': is_real_wrench,
                'data_source': 'Real WRENCH 0.3-dev' if is_real_wrench else 'Simulated',
                'reproducibility': 'High' if is_real_wrench else 'Medium'
            }
        }
        
        return report
    
    def _assess_scalability(self, analysis: Dict) -> str:
        """评估可扩展性"""
        cpu_util = analysis['performance_metrics']['cpu_utilization']
        if cpu_util > 0.8:
            return "High CPU utilization suggests good scalability potential"
        elif cpu_util > 0.5:
            return "Moderate CPU utilization indicates room for scaling"
        else:
            return "Low CPU utilization suggests under-utilized resources"
    
    def _identify_bottlenecks(self, analysis: Dict) -> List[str]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 检查各阶段的计算负载
        stages = analysis['stage_analysis']
        max_load = max(stage['computational_load'] for stage in stages.values())
        
        for stage_name, stage_data in stages.items():
            if stage_data['computational_load'] > max_load * 0.8:
                bottlenecks.append(f"{stage_name}阶段计算负载较高")
        
        # 检查内存使用
        memory_usage = analysis['resource_utilization']['memory_usage_ratio']
        if memory_usage > 0.8:
            bottlenecks.append("内存使用率较高，可能成为瓶颈")
        
        return bottlenecks or ["未发现明显瓶颈"]
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        parallel_eff = analysis['resource_utilization']['parallel_efficiency']
        if parallel_eff < 0.5:
            recommendations.append("增加任务并行度以提高资源利用率")
        
        cpu_util = analysis['performance_metrics']['cpu_utilization']
        if cpu_util < 0.6:
            recommendations.append("优化任务调度以提高CPU利用率")
        
        host_count = analysis['resource_utilization']['host_count']
        if host_count < 3:
            recommendations.append("考虑增加计算节点以提升并行处理能力")
        
        return recommendations or ["当前配置较为合理"]
    
    def _fallback_simulation(self, workflow: Dict) -> Dict:
        """后备仿真"""
        tasks = workflow.get('tasks', [])
        total_flops = sum(task.get('flops', 0) for task in tasks)
        
        return {
            'success': True,
            'workflow_id': workflow.get('name', 'fallback'),
            'execution_time': total_flops / 1e9,
            'task_count': len(tasks),
            'total_flops': total_flops,
            'mock_data': True,
            'platform_type': 'Fallback Simulation'
        }
    
    def save_results(self, results: Dict):
        """保存结果"""
        results_dir = Path(self.config.get('paths', {}).get('results_dir', './results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        results_file = results_dir / 'wass_academic_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存学术报告
        report_file = results_dir / 'academic_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results['academic_report'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {results_dir}")
    
    def run_complete_academic_research(self) -> Dict:
        """运行完整的学术研究流程"""
        logger.info("🎓 启动WASS学术研究平台")
        
        # 1. 加载配置
        self.load_config()
        
        # 2. 初始化WRENCH仿真器
        wrench_success = self.initialize_wrench_simulator()
        
        # 3. 运行学术仿真
        results = self.run_academic_simulation()
        
        # 4. 添加平台状态信息
        results['platform_status'] = {
            'wrench_initialized': wrench_success,
            'config_loaded': self.config is not None,
            'simulation_method': 'real_wrench' if wrench_success else 'fallback'
        }
        
        # 5. 保存结果
        self.save_results(results)
        
        # 6. 打印摘要
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """打印结果摘要"""
        report = results['academic_report']
        sim_result = results['simulation_result']
        
        print("\n" + "="*60)
        print("🎓 WASS学术研究平台 - 执行摘要")
        print("="*60)
        
        print(f"📊 仿真平台: {report['simulation_platform']['type']}")
        print(f"🔬 WRENCH集成: {'✅ 真实' if report['simulation_platform']['real_wrench_integration'] else '❌ 模拟'}")
        print(f"📈 工作流任务: {report['workflow_summary']['total_tasks']} 个")
        print(f"⏱️  执行时间: {report['performance_results']['execution_time_seconds']:.2f} 秒")
        print(f"🚀 系统吞吐量: {report['performance_results']['computational_throughput_flops']:.2e} Flops/s")
        print(f"💻 CPU利用率: {report['performance_results']['cpu_utilization_percent']:.1f}%")
        print(f"🖥️  计算主机: {report['infrastructure_details']['host_count']} 个")
        
        print(f"\n📋 学术评估:")
        print(f"   可扩展性: {report['academic_insights']['scalability_assessment']}")
        print(f"   数据有效性: {report['validation']['simulation_validity']}")
        print(f"   可重现性: {report['validation']['reproducibility']}")
        
        print(f"\n🎯 Mock数据状态: {sim_result.get('mock_data', True)}")
        
        print("="*60)

def main():
    """主函数"""
    platform = WASSAcademicPlatform()
    results = platform.run_complete_academic_research()
    return results

if __name__ == "__main__":
    main()
