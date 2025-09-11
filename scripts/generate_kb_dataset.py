"""Generate training dataset for performance predictor using wrench-python-api.

This version synthesizes structured (state, action, context) feature triples
for each (task, host) pair across simple synthetic workflows under two
baseline labels (HEFT, FIFO). It does NOT currently emulate the true runtime
selection order—rather, it produces deterministic feature grids suitable for
bootstrapping the predictor.
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

logger = get_logger(__name__, logging.INFO)

def generate_kb_data(config: Dict) -> List[Dict[str, Any]]:
    """Return list of KB samples with schema expected by training script."""
    platform_cfg = config.get('platform', {})
    kb_cfg = config.get('kb_generation', {})
    platform_file = platform_cfg.get('platform_file')
    if not platform_file:
        raise ValueError("platform.platform_file missing in config")

    controller_host = platform_cfg.get('controller_host', 'ControllerHost')
    num_tasks = int(kb_cfg.get('num_tasks', 20))
    context_dim = int(kb_cfg.get('context_dim', 8))
    schedulers = ["HEFT", "FIFO"]

    samples: List[Dict[str, Any]] = []

    for sched_name in schedulers:
        logger.info(f"[KB] Generating synthetic samples for {sched_name}")
        sim = wrench.Simulation()
        with open(platform_file, 'r', encoding='utf-8') as f:
            xml = f.read()
        sim.start(xml, controller_host)

        hosts_all = sim.get_all_hostnames()
        hosts = [h for h in hosts_all if 'Controller' not in h and 'Storage' not in h] or hosts_all
        host_specs = {h: (4, 10.0) for h in hosts}  # (cores, speed)
        avg_speed = sum(v[1] for v in host_specs.values()) / max(1, len(host_specs))

        wf = sim.create_workflow()
        tasks = []
        files = []
        
        # Create tasks and files
        for i in range(num_tasks):
            t = wf.add_task(f"task_{i}", 100.0 + 25 * i, 1, 1, 0)
            tasks.append(t)
            
            # Create output file for each task (except the last one)
            if i < num_tasks - 1:
                output_file = sim.add_file(f"output_{i}", 1024)  # 1KB file
                t.add_output_file(output_file)
                files.append(output_file)
        
        # Create dependencies by connecting files
        for i in range(1, num_tasks):
            if i % 2 == 0 and i > 1:  # Every second task depends on previous
                input_file = files[i-2]  # Use output from task i-2
                tasks[i].add_input_file(input_file)

        for t in tasks:
            state_features = [
                float(t.get_flops()),
                float(len(t.get_input_files())),
                float(t.get_number_of_children()),
                float(avg_speed),
                float(len(host_specs))
            ]
            for idx, (host, spec) in enumerate(host_specs.items()):
                speed = spec[1]
                action_features = [0.0] * len(host_specs)
                action_features[idx] = 1.0
                context_features = [0.0] * context_dim
                finish_time = t.get_flops() / max(1e-6, speed)
                samples.append({
                    'scheduler': sched_name,
                    'state_features': state_features,
                    'action_features': action_features,
                    'context_features': context_features,
                    'achieved_finish_time': finish_time,
                    'meta': {'task_id': t.get_name(), 'host': host}
                })
        sim.terminate()
        logger.info(f"[KB] {sched_name} cumulative samples: {len(samples)}")

    logger.info(f"[KB] Total samples: {len(samples)}")
    return samples

def load_config(cfg_path: str) -> Dict:
    """Load configuration with includes support"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    
    # Process includes
    if 'include' in cfg:
        base_dir = os.path.dirname(cfg_path)
        for include_file in cfg['include']:
            include_path = os.path.join(base_dir, include_file)
            if os.path.exists(include_path):
                with open(include_path, 'r', encoding='utf-8') as f:
                    include_cfg = yaml.safe_load(f) or {}
                    # Merge configurations
                    for key, value in include_cfg.items():
                        if key not in cfg:
                            cfg[key] = value
    
    return cfg

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_kb_dataset.py <config.yaml>")
        sys.exit(1)
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    data = generate_kb_data(cfg)
    out_path = 'data/kb_training_dataset.json'
    os.makedirs('data', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    logger.info(f"[KB] Saved {len(data)} samples to {out_path}")

if __name__ == '__main__':
    main()