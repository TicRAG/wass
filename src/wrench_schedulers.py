"""
Custom Python schedulers and the main WrenchExperimentRunner class.
"""
from __future__ import annotations
from typing import List, Dict, Any
import wrench
import pandas as pd
from pathlib import Path
import json

# 基础调度器类
class BaseScheduler:
    """基础调度器类，用于自定义调度策略"""
    def __init__(self, simulation: 'wrench.Simulation', compute_service, hosts: Dict[str, Any] = None):
        self.sim = simulation
        self.compute_service = compute_service
        self.hosts = hosts
        self.completed_tasks = set()

    def schedule_ready_tasks(self, workflow: wrench.Workflow, storage_service):
        """调度所有准备就绪的任务"""
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            if task not in self.completed_tasks:
                # 调用子类的调度决策方法
                host_name = self.get_scheduling_decision(task)
                if host_name:
                    # 准备文件位置字典
                    file_locations = {}
                    for f in task.get_input_files():
                        file_locations[f] = storage_service
                    for f in task.get_output_files():
                        file_locations[f] = storage_service
                    
                    # 创建标准作业并提交
                    job = self.sim.create_standard_job([task], file_locations)
                    self.compute_service.submit_standard_job(job)

    def get_scheduling_decision(self, task: wrench.Task):
        """获取调度决策（由子类实现）"""
        raise NotImplementedError

    def handle_completion(self, task: wrench.Task):
        """处理任务完成事件"""
        self.completed_tasks.add(task)

class FIFOScheduler(BaseScheduler):
    """简单的先进先出调度器"""
    def get_scheduling_decision(self, task: wrench.Task):
        # 总是选择第一个可用主机
        return list(self.hosts.keys())[0]

class HEFTScheduler(BaseScheduler):
    """简化版HEFT（异构最早完成时间）调度器"""
    def get_scheduling_decision(self, task: wrench.Task):
        # 简化版HEFT：选择能最早完成任务的主机
        best_host = None
        min_eft = float('inf')

        for host_name in self.hosts.keys():
            # 获取主机对应的计算服务
            compute_service = self.compute_service
            # 获取主机核心速度 - 使用get_core_flop_rates方法
            core_speeds = compute_service.get_core_flop_rates()
            host_speed = core_speeds.get(host_name, 1e9)  # 默认1 GFLOPS
            
            # 计算任务在该主机上的执行时间
            exec_time = task.get_flops() / host_speed
            # 计算预计完成时间
            finish_time = self.sim.get_simulated_time() + exec_time
            
            if finish_time < min_eft:
                min_eft = finish_time
                best_host = host_name
        
        return best_host or list(self.hosts.keys())[0]

class WASSHeuristicScheduler(HEFTScheduler):
    """WASS启发式调度器"""
    # 实际实现中会包含更复杂的数据感知逻辑
    pass

class WrenchExperimentRunner:
    """
    处理多个WRENCH仿真以比较不同调度器的性能
    """
    def __init__(self, schedulers: Dict[str, Any], config: Dict[str, Any]):
        self.schedulers_map = schedulers
        self.config = config
        self.platform_file = config.get("platform_file")
        self.workflow_dir = Path(config.get("workflow_dir", "workflows"))
        self.workflow_sizes = config.get("workflow_sizes", [20, 50, 100])
        self.repetitions = config.get("repetitions", 3)

    def _run_single_simulation(self, scheduler_name: str, scheduler_class: Any, workflow_file: str) -> Dict[str, Any]:
        """运行单个WRENCH仿真并返回结果"""
        try:
            # 读取平台文件内容
            with open(self.platform_file, 'r') as f:
                platform_xml = f.read()
            
            # 创建模拟对象
            simulation = wrench.Simulation()
            
            # 启动仿真，指定控制器主机
            controller_host = "ControllerHost"
            simulation.start(platform_xml, controller_host)
            
            # 获取所有主机名
            all_hostnames = simulation.get_all_hostnames()
            
            # 过滤出计算主机（排除控制器主机）
            compute_hosts = [host for host in all_hostnames if host != controller_host]
            
            # 创建存储服务（在StorageHost上，挂载点为/storage）
            storage_service = simulation.create_simple_storage_service("StorageHost", ["/storage"])
            
            # 创建裸机计算服务
            compute_service = simulation.create_bare_metal_compute_service(
                controller_host, 
                {host: (-1, -1) for host in compute_hosts},  # 所有核心都可用
                "/scratch", 
                {}, 
                {}
            )
            
            # 创建主机信息字典
            hosts_dict = {name: {} for name in compute_hosts}
            
            # 实例化调度器
            if callable(scheduler_class) and not isinstance(scheduler_class, type):
                # 工厂函数形式的调度器
                scheduler_impl = scheduler_class(simulation, compute_service, hosts_dict)
            else:
                # 类形式的调度器
                scheduler_impl = scheduler_class(simulation, compute_service, hosts_dict)
            
            # 加载工作流
            workflow = simulation.create_workflow_from_json(str(workflow_file))
            
            # 创建工作流中的所有文件副本
            for file in workflow.get_input_files():
                storage_service.create_file_copy(file)
            
            # 开始调度
            scheduler_impl.schedule_ready_tasks(workflow, storage_service)
            
            # 运行仿真循环
            while not workflow.is_done():
                # 等待下一个事件
                event = simulation.wait_for_next_event()
                
                # 处理任务完成事件
                if event["event_type"] == "standard_job_completion":
                    job = event["standard_job"]
                    for task in job.get_tasks():
                        scheduler_impl.handle_completion(task)
                    # 调度新的就绪任务
                    scheduler_impl.schedule_ready_tasks(workflow, storage_service)
                elif event["event_type"] == "simulation_termination":
                    break
            
            # 获取makespan
            makespan = simulation.get_simulated_time()
            
            # 终止仿真
            simulation.terminate()
            
            return {
                "scheduler": scheduler_name,
                "workflow": workflow_file.name,
                "makespan": makespan,
                "status": "success"
            }
        except Exception as e:
            print(f"ERROR running {scheduler_name} on {workflow_file.name}: {e}")
            return {"scheduler": scheduler_name, "workflow": workflow_file.name, "makespan": float('inf'), "status": "failed"}

    def run_all(self) -> List[Dict[str, Any]]:
        """运行所有配置的实验"""
        results = []
        total_exps = len(self.schedulers_map) * len(self.workflow_sizes) * self.repetitions
        print(f"总实验数: {total_exps}")
        
        exp_count = 0
        for name, sched_class in self.schedulers_map.items():
            for size in self.workflow_sizes:
                for _ in range(self.repetitions):
                    wf_file = self.workflow_dir / f"{size}-tasks-wf.json" # 假设的文件名
                    if not wf_file.exists():
                        # 尝试其他可能的文件名格式
                        wf_file = self.workflow_dir / f"montage_{size}_tasks.json"
                        if not wf_file.exists():
                            continue
                    exp_count += 1
                    print(f"运行实验 [{exp_count}/{total_exps}]: {name} on {wf_file.name}")
                    result = self._run_single_simulation(name, sched_class, wf_file)
                    results.append(result)
        return results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析并打印实验结果摘要"""
        if not results:
            print("没有可供分析的实验结果。")
            return
        df = pd.DataFrame(results)
        summary = df.groupby('scheduler')['makespan'].agg(['mean', 'std', 'min', 'count']).reset_index()
        summary = summary.rename(columns={
            'scheduler': '调度器', 'mean': '平均Makespan', 'std': '标准差',
            'min': '最佳', 'count': '实验次数'
        })
        print("\n" + "="*60)
        print(summary.to_string(index=False))
        print("="*60 + "\n")