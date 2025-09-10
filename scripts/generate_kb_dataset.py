# scripts/generate_kb_dataset.py

import json
import logging
import sys
from typing import Dict, List, Any

import wrench.schedulers as schedulers  # <--- UPDATED IMPORT
import yaml

# Add the project root to the Python path
sys.path.append('.')

from src.factory import PlatformFactory, WorkflowFactory
from src.utils import get_logger, extract_features_for_kb
# Note: We use the standard Wrench Simulation for this script, not our custom one
# because the standard schedulers are designed to work with it.

logger = get_logger(__name__, logging.INFO)

def generate_kb_data(config: Dict) -> List[Dict[str, Any]]:
    """
    Generates a knowledge base dataset by running simulations with various schedulers.
    """
    platform_factory = PlatformFactory(config['platform'])
    platform = platform_factory.get_platform()
    node_names = list(platform.get_nodes().keys())

    workflow_factory = WorkflowFactory(config['workflow'])
    workflow = workflow_factory.get_workflow()

    scheduler_classes = {
        "HEFT": schedulers.HeftScheduler,
        "FIFO": schedulers.FifoScheduler,
    }

    kb_entries = []

    for name, scheduler_class in scheduler_classes.items():
        logger.info(f"Running simulation with {name} scheduler...")

        class RecordingScheduler(scheduler_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.decisions = []

            def schedule(self, ready_tasks, simulation_platform=None):
                original_decision = super().schedule(ready_tasks, platform)
                
                simple_decision = {}
                for node, task_list in original_decision.items():
                    if task_list:
                        simple_decision[node.name] = task_list[0] 
                
                if simple_decision:
                    self.decisions.append(simple_decision)
                
                return original_decision

        scheduler_instance = RecordingScheduler(platform=platform)
        
        simulation = schedulers.Simulation(scheduler_instance, platform, workflow)
        simulation.run()
        final_makespan = simulation.time

        logger.info(f"Finished simulation with {name}. Makespan: {final_makespan:.2f}")

        for decision in scheduler_instance.decisions:
            node_id = list(decision.keys())[0]
            task = list(decision.values())[0]
            
            features = extract_features_for_kb(task, node_id, platform, workflow)
            
            kb_entries.append({
                "features": features,
                "makespan": final_makespan
            })

    logger.info(f"Generated {len(kb_entries)} entries for the knowledge base.")
    return kb_entries

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_kb_dataset.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    kb_data = generate_kb_data(config)

    output_file = config['drl']['knowledge_base_path']
    with open(output_file, 'w') as f:
        for entry in kb_data:
            f.write(json.dumps(entry) + '\n')

    logger.info(f"Knowledge base data successfully saved to {output_file}")