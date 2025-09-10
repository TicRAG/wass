import json
import logging
import sys
from typing import Dict, List, Any

import wrench.standard_schedulers as schedulers
import yaml

# Add the project root to the Python path
sys.path.append('.')

from src.factory import PlatformFactory, WorkflowFactory
from src.utils import get_logger, extract_features_for_kb
from wass_wrench_simulator import WassWrenchSimulator

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

    # Schedulers to generate data from
    scheduler_classes = {
        "HEFT": schedulers.HEFTScheduler,
        "FIFO": schedulers.FIFOScheduler,
    }

    kb_entries = []

    for name, scheduler_class in scheduler_classes.items():
        logger.info(f"Running simulation with {name} scheduler...")

        # We need a special version of the scheduler that records decisions
        class RecordingScheduler(scheduler_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.decisions = []

            def schedule(self, ready_tasks, simulation_platform=None):
                # The standard schedulers have a different signature
                # We adapt it here. The WassWrenchSimulator is not passed to them.
                original_decision = super().schedule(ready_tasks, platform)
                
                # We need to map the Wrench decision format to our simplified format
                simple_decision = {}
                for node, task_list in original_decision.items():
                    if task_list:
                        # Assuming one task per decision for simplicity in KB
                        simple_decision[node.name] = task_list[0] 
                
                if simple_decision:
                    self.decisions.append(simple_decision)
                
                return original_decision

        scheduler_instance = RecordingScheduler(platform=platform, execution_time_estimator=None)
        
        # Standard Wrench simulation for standard schedulers
        simulation = schedulers.WrenchSimulation(scheduler_instance, platform, workflow)
        simulation.run()
        final_makespan = simulation.time

        logger.info(f"Finished simulation with {name}. Makespan: {final_makespan:.2f}")

        # Post-process decisions to create KB entries
        # This part is tricky because we need to reconstruct the state at each decision point.
        # For simplicity, we'll extract features based on the final state, which is an approximation.
        # A more accurate implementation would snapshot the state at each schedule() call.
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