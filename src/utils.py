"""工具函数、日志设置和最终版的实验运行器。"""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Generator, List
from pathlib import Path
import sys
import wrench
import pandas as pd
import numpy as np

def get_logger(name, level=logging.INFO):
    """获取一个已配置的日志器。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# --- 最终版实验运行器 ---
class WrenchExperimentRunner:
    """处理多个WRENCH模拟的执行，以比较不同的调度器。"""
    def __init__(self, schedulers: Dict[str, Any], config: Dict[str, Any]):
        self.schedulers_map = schedulers
        self.config = config
        self.platform_file = config.get("platform_file")
        self.workflow_dir = Path(config.get("workflow_dir", "workflows"))
        self.workflow_sizes = config.get("workflow_sizes", [20, 50, 100])
        self.repetitions = config.get("repetitions", 3)
        self.output_dir = Path(config.get("output_dir", "results/final_experiments"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_single_simulation(self, scheduler_name: str, scheduler_impl: Any, workflow_file: str) -> Dict[str, Any]:
        """运行单个WRENCH模拟并返回结果。"""
        try:
            sim = wrench.Simulation()
            sim.add_platform(self.platform_file)
            
            all_hosts = list(sim.get_platform().get_compute_hosts().keys())
            controller = all_hosts[0]
            compute_hosts = all_hosts[1:] if len(all_hosts) > 1 else all_hosts
            
            cs = wrench.BareMetalComputeService(controller, compute_hosts)
            sim.add_compute_service(cs)

            # 实例化调度器
            scheduler_instance = scheduler_impl
            if hasattr(scheduler_impl, 'set_simulation_context'):
                 scheduler_instance.set_simulation_context(sim, cs, {h:{} for h in compute_hosts})

            sim.set_scheduler(scheduler_instance)
            workflow = sim.create_workflow_from_json(str(workflow_file))
            job = sim.create_standard_job(workflow.get_tasks())
            cs.submit_job(job)
            sim.run()
            
            return {"scheduler": scheduler_name, "workflow": workflow_file.name, "makespan": sim.get_makespan(), "status": "success"}
        except Exception as e:
            print(f"ERROR running {scheduler_name} on {workflow_file.name}: {e}")
            return {"scheduler": scheduler_name, "workflow": workflow_file.name, "makespan": float('inf'), "status": "failed"}

    def run_all(self) -> List[Dict[str, Any]]:
        """运行所有配置的实验。"""
        results = []
        total_exps = len(self.schedulers_map) * len(self.workflow_sizes) * self.repetitions
        print(f"总实验数: {total_exps}")
        
        exp_count = 0
        for name, sched_impl in self.schedulers_map.items():
            for size in self.workflow_sizes:
                # [FIX] 智能扫描工作流文件，而不是依赖固定名称
                matching_files = list(self.workflow_dir.glob(f'*_{size}_*.json')) + \
                                 list(self.workflow_dir.glob(f'*_{size}.json')) + \
                                 list(self.workflow_dir.glob(f'*{size}-tasks-wf.json'))
                
                if not matching_files:
                    print(f"[警告] 在 {self.workflow_dir} 中未找到大小为 {size} 的工作流文件，跳过...")
                    continue
                
                # 为了实验的一致性，我们只使用找到的第一个匹配文件
                workflow_file_to_run = matching_files[0]
                
                for rep in range(self.repetitions):
                    exp_count += 1
                    print(f"运行实验 [{exp_count}/{total_exps}]: {name} on {workflow_file_to_run.name}")
                    result = self._run_single_simulation(name, sched_impl, workflow_file_to_run)
                    results.append(result)
        return results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析、打印并保存实验结果摘要。"""
        if not results:
            print("没有可供分析的实验结果。"); return

        df = pd.DataFrame(results)
        
        # 保存详细的原始结果
        detailed_csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(detailed_csv_path, index=False)
        print(f"✅ 详细实验结果已保存到: {detailed_csv_path}")

        summary = df.groupby('scheduler')['makespan'].agg(['mean', 'std', 'min', 'count']).reset_index()
        summary = summary.rename(columns={'scheduler': '调度器', 'mean': '平均Makespan', 'std': '标准差', 'min': '最佳', 'count': '实验次数'})

        print("\n" + "="*60); print("📈 实验结果分析:"); print(summary.to_string(index=False)); print("="*60 + "\n")
        
        # [FIX] 保存最终的摘要结果
        summary_csv_path = self.output_dir / "summary_results.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"✅ 实验结果摘要已保存到: {summary_csv_path}")