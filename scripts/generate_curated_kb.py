#!/usr/bin/env python3
"""
净化知识库生成器 - 仅包含HEFT和WassHeuristicScheduler
移除FIFO、Random等其他调度器，专注于核心对比算法
"""

import json
import os
import logging
import sys
from typing import Dict, List, Any
import yaml

import wrench  # type: ignore

sys.path.append('.')
from src.utils import get_logger
from src.wrench_schedulers import WassHeuristicScheduler

logger = get_logger(__name__, logging.INFO)

class WassHeuristicKBGenerator:
    """WassHeuristic调度器知识库生成器"""
    
    def __init__(self):
        self.scheduler = WassHeuristicScheduler()
    
    def generate_heuristic_samples(self, config: Dict) -> List[Dict[str, Any]]:
        """生成WassHeuristic调度器的知识样本"""
        platform_cfg = config.get('platform', {})
        kb_cfg = config.get('kb_generation', {})
        platform_file = platform_cfg.get('platform_file')
        if not platform_file:
            raise ValueError("platform.platform_file missing in config")

        controller_host = platform_cfg.get('controller_host', 'ControllerHost')
        num_tasks = int(kb_cfg.get('num_tasks', 20))
        num_workflows = int(kb_cfg.get('num_workflows', 60))
        context_dim = int(kb_cfg.get('context_dim', 8))

        samples: List[Dict[str, Any]] = []

        logger.info("[KB] Generating curated samples for WassHeuristic")
        sim = wrench.Simulation()
        with open(platform_file, 'r', encoding='utf-8') as f:
            xml = f.read()
        sim.start(xml, controller_host)

        hosts_all = sim.get_all_hostnames()
        hosts = [h for h in hosts_all if 'Controller' not in h and 'Storage' not in h] or hosts_all
        host_specs = {h: (4, 10.0) for h in hosts}
        avg_speed = sum(v[1] for v in host_specs.values()) / max(1, len(host_specs))

        # 生成多样化的工作流
        for wf_idx in range(num_workflows):
            wf = sim.create_workflow()
            tasks = []
            
            # 变化任务数量
            current_num_tasks = max(5, num_tasks + (wf_idx % 10) - 5)
            
            # 创建任务
            for i in range(current_num_tasks):
                flops_base = 100.0 + 25 * i
                flops_variation = flops_base * (0.5 + wf_idx * 0.1 % 1.0)
                t = wf.add_task(f"wf{wf_idx}_task_{i}", flops_variation, 1, 1, 0)
                tasks.append(t)
            
            # 创建依赖关系
            for i in range(1, current_num_tasks):
                if wf_idx % 3 == 0:  # 链式
                    if i > 0:
                        tasks[i-1].add_child(tasks[i])
                elif wf_idx % 3 == 1:  # 树状
                    if i % 2 == 0 and i > 1:
                        tasks[i-2].add_child(tasks[i])
                # else: 并行

            # 使用WassHeuristic调度器生成调度决策
            for t in tasks:
                # 模拟调度决策过程
                state_features = [
                    float(t.get_flops()),
                    float(len(t.get_input_files())),
                    float(len(t.get_children())),
                    float(avg_speed),
                    float(len(host_specs))
                ]
                
                # 为每个可能的节点选择生成样本
                for idx, (host, spec) in enumerate(host_specs.items()):
                    speed = spec[1]
                    action_features = [0.0] * len(host_specs)
                    action_features[idx] = 1.0
                    context_features = [0.0] * context_dim
                    
                    # 计算WassHeuristic的启发式分数
                    heuristic_score = self._calculate_heuristic_score(t, host, sim, speed)
                    
                    samples.append({
                        'scheduler': 'WassHeuristic',
                        'state_features': state_features,
                        'action_features': action_features,
                        'context_features': context_features,
                        'heuristic_score': heuristic_score,
                        'meta': {'task_id': t.get_name(), 'host': host, 'workflow_id': f'wf_{wf_idx}'}
                    })
            
            if (wf_idx + 1) % 10 == 0:
                logger.info(f"[KB] WassHeuristic 已生成 {wf_idx + 1}/{num_workflows} 个工作流")
                
        sim.terminate()
        logger.info(f"[KB] WassHeuristic 累计样本: {len(samples)}")
        return samples
    
    def _calculate_heuristic_score(self, task, sim, node_speed):
        """计算WassHeuristic的启发式分数"""
        # 简化的启发式计算
        computation_time = task.get_flops() / max(1e-6, node_speed)
        return 1.0 / (computation_time + 1.0)  # 分数越高越好

def generate_curated_kb_data(config: Dict) -> List[Dict[str, Any]]:
    """生成净化后的知识库数据 - 仅HEFT和WassHeuristic"""
    
    # 生成HEFT样本（复用原逻辑）
    heft_samples = []
    platform_cfg = config.get('platform', {})
    kb_cfg = config.get('kb_generation', {})
    platform_file = platform_cfg.get('platform_file')
    if not platform_file:
        raise ValueError("platform.platform_file missing in config")

    controller_host = platform_cfg.get('controller_host', 'ControllerHost')
    num_tasks = int(kb_cfg.get('num_tasks', 20))
    num_workflows = int(kb_cfg.get('num_workflows', 60))
    context_dim = int(kb_cfg.get('context_dim', 8))

    logger.info("[KB] Generating curated samples for HEFT")
    sim = wrench.Simulation()
    with open(platform_file, 'r', encoding='utf-8') as f:
        xml = f.read()
    sim.start(xml, controller_host)

    hosts_all = sim.get_all_hostnames()
    hosts = [h for h in hosts_all if 'Controller' not in h and 'Storage' not in h] or hosts_all
    host_specs = {h: (4, 10.0) for h in hosts}
    avg_speed = sum(v[1] for v in host_specs.values()) / max(1, len(host_specs))

    for wf_idx in range(num_workflows):
        wf = sim.create_workflow()
        tasks = []
        
        current_num_tasks = max(5, num_tasks + (wf_idx % 10) - 5)
        
        for i in range(current_num_tasks):
            flops_base = 100.0 + 25 * i
            flops_variation = flops_base * (0.5 + wf_idx * 0.1 % 1.0)
            t = wf.add_task(f"wf{wf_idx}_task_{i}", flops_variation, 1, 1, 0)
            tasks.append(t)
        
        for i in range(1, current_num_tasks):
            if wf_idx % 3 == 0:
                if i > 0:
                    wf.add_dependency(tasks[i-1], tasks[i])
            elif wf_idx % 3 == 1:
                if i % 2 == 0 and i > 1:
                    wf.add_dependency(tasks[i-2], tasks[i])

        for t in tasks:
            state_features = [
                float(t.get_flops()),
                float(len(t.get_input_files())),
                float(len(t.get_children())),
                float(avg_speed),
                float(len(host_specs))
            ]
            for idx, (host, spec) in enumerate(host_specs.items()):
                speed = spec[1]
                action_features = [0.0] * len(host_specs)
                action_features[idx] = 1.0
                context_features = [0.0] * context_dim
                finish_time = t.get_flops() / max(1e-6, speed)
                heft_samples.append({
                    'scheduler': 'HEFT',
                    'state_features': state_features,
                    'action_features': action_features,
                    'context_features': context_features,
                    'achieved_finish_time': finish_time,
                    'meta': {'task_id': t.get_name(), 'host': host, 'workflow_id': f'wf_{wf_idx}'}
                })
        
        if (wf_idx + 1) % 10 == 0:
            logger.info(f"[KB] HEFT 已生成 {wf_idx + 1}/{num_workflows} 个工作流")
            
    sim.terminate()
    logger.info(f"[KB] HEFT 累计样本: {len(heft_samples)}")
    
    # 生成WassHeuristic样本
    heuristic_generator = WassHeuristicKBGenerator()
    heuristic_samples = heuristic_generator.generate_heuristic_samples(config)
    
    # 合并净化后的样本
    curated_samples = heft_samples + heuristic_samples
    
    logger.info(f"[KB] 净化知识库总计样本: {len(curated_samples)}")
    logger.info(f"[KB] 调度器分布: HEFT={len(heft_samples)}, WassHeuristic={len(heuristic_samples)}")
    
    return curated_samples

def load_config(cfg_path: str) -> Dict:
    """加载配置文件"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    if 'include' in cfg:
        base_dir = os.path.dirname(cfg_path)
        for include_file in cfg['include']:
            include_path = os.path.join(base_dir, include_file)
            if os.path.exists(include_path):
                with open(include_path, 'r', encoding='utf-8') as f:
                    include_cfg = yaml.safe_load(f) or {}
                    for key, value in include_cfg.items():
                        if key not in cfg:
                            cfg[key] = value
    
    return cfg

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_curated_kb.py <config.yaml>")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    data = generate_curated_kb_data(cfg)
    
    out_path = 'data/curated_kb_training_dataset.json'
    os.makedirs('data', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
    logger.info(f"[KB] 净化知识库已保存: {out_path}")
    logger.info(f"[KB] 总计样本数: {len(data)}")

if __name__ == '__main__':
    main()