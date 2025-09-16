"""
Custom Python schedulers and the main WrenchExperimentRunner class.
"""
from __future__ import annotations
from typing import List, Dict, Any
import wrench
import pandas as pd
from pathlib import Path
import json

# [FIX] Restore the definitions for the heuristic schedulers
class BaseScheduler:
    """Base class for custom schedulers."""
    def __init__(self, simulation: 'wrench.Simulation', compute_service, hosts: Dict[str, Any] = None):
        self.sim = simulation
        self.cs = compute_service
        self.hosts = hosts

    def get_scheduling_decision(self, job: wrench.StandardJob, available_resources: List[str]):
        raise NotImplementedError

class FIFOScheduler(BaseScheduler):
    """A simple First-In, First-Out scheduler."""
    def get_scheduling_decision(self, job: wrench.StandardJob, available_resources: List[str]):
        # Always chooses the first available resource
        return available_resources[0]

class HEFTScheduler(BaseScheduler):
    """A simple HEFT (Heterogeneous Earliest Finish Time) scheduler."""
    def get_scheduling_decision(self, job: wrench.StandardJob, available_resources: List[str]):
        # A simplified HEFT: choose the host that will finish the job the earliest
        best_host = None
        min_eft = float('inf')

        for host_name in available_resources:
            # This is a simplified calculation. A full HEFT would be more complex.
            task = job.get_tasks()[0]
            host = self.sim.get_platform().get_host_by_name(host_name)
            exec_time = task.get_flops() / host.get_speed()
            
            # A more realistic model would get the host's current workload
            finish_time = self.sim.get_simulated_time() + exec_time
            
            if finish_time < min_eft:
                min_eft = finish_time
                best_host = host_name
        
        return best_host or available_resources[0]

class WASSHeuristicScheduler(HEFTScheduler):
    """The WASS Heuristic scheduler (can inherit from HEFT for this example)."""
    # In a real implementation, this would have more complex data-aware logic.
    pass


# [FIX] The WrenchExperimentRunner class, now in the same file
class WrenchExperimentRunner:
    """
    Handles the execution of multiple WRENCH simulations to compare schedulers.
    """
    def __init__(self, schedulers: Dict[str, Any], config: Dict[str, Any]):
        self.schedulers_map = schedulers
        self.config = config
        self.platform_file = config.get("platform_file")
        self.workflow_dir = Path(config.get("workflow_dir", "workflows"))
        self.workflow_sizes = config.get("workflow_sizes", [20, 50, 100])
        self.repetitions = config.get("repetitions", 3)

    def _run_single_simulation(self, scheduler_name: str, scheduler_class: Any, workflow_file: str) -> Dict[str, Any]:
        """Runs a single WRENCH simulation and returns the results."""
        try:
            sim = wrench.Simulation()
            sim.add_platform(self.platform_file)
            
            all_hosts = list(sim.get_platform().get_compute_hosts().keys())
            controller = all_hosts[0]
            compute_hosts = all_hosts[1:] if len(all_hosts) > 1 else all_hosts
            
            cs = wrench.BareMetalComputeService(controller, compute_hosts, "/scratch")
            sim.add_compute_service(cs)
            
            # Instantiate the scheduler
            if scheduler_name in ["WASS-DRL", "WASS-RAG"]:
                scheduler_impl = scheduler_class(sim, cs, {h:{} for h in compute_hosts})
            else:
                scheduler_impl = scheduler_class(sim, cs, {h:{} for h in compute_hosts})

            sim.set_scheduler(scheduler_impl)

            workflow = sim.create_workflow_from_json(str(workflow_file))
            job = sim.create_standard_job(workflow.get_tasks())
            cs.submit_job(job)

            sim.run()
            
            return {
                "scheduler": scheduler_name,
                "workflow": workflow_file.name,
                "makespan": sim.get_makespan(),
                "status": "success"
            }
        except Exception as e:
            print(f"ERROR running {scheduler_name} on {workflow_file.name}: {e}")
            return {"scheduler": scheduler_name, "workflow": workflow_file.name, "makespan": float('inf'), "status": "failed"}

    def run_all(self) -> List[Dict[str, Any]]:
        """Runs all configured experiments."""
        results = []
        total_exps = len(self.schedulers_map) * len(self.workflow_sizes) * self.repetitions
        print(f"总实验数: {total_exps}")
        
        exp_count = 0
        for name, sched_class in self.schedulers_map.items():
            for size in self.workflow_sizes:
                for _ in range(self.repetitions):
                    wf_file = self.workflow_dir / f"{size}-tasks-wf.json" # Assumed name
                    if not wf_file.exists(): continue
                    exp_count += 1
                    print(f"运行实验 [{exp_count}/{total_exps}]: {name} on {wf_file.name}")
                    result = self._run_single_simulation(name, sched_class, wf_file)
                    results.append(result)
        return results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyzes and prints a summary of the experiment results."""
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