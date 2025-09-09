#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent if current_dir.name in ['scripts', 'experiments'] else current_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from ai_schedulers import WASSRAGScheduler, SchedulingState
    print(">>> Successfully imported AI schedulers")
except ImportError as e:
    print(f"!!! Error: {e}")
    sys.exit(1)

def main():
    """Main function to test predictions with a more realistic scenario."""
    MODEL_PATH = project_root / "models/wass_models.pth"
    KB_PATH = project_root / "data/knowledge_base.pkl"

    try:
        scheduler = WASSRAGScheduler(model_path=str(MODEL_PATH), knowledge_base_path=str(KB_PATH))
        print(">>> Successfully initialized WASS-RAG scheduler\n")
    except Exception as e:
        print(f"!!! Failed to initialize scheduler: {e}")
        return

    # --- Upgraded Test Scenario ---
    print("--- Using a more realistic test scenario...")
    state = SchedulingState(
        workflow_graph={
            "tasks": [
                {"id": f"task_{i}", "flops": np.random.uniform(5e9, 15e9), "memory": np.random.uniform(4.0, 12.0)}
                for i in range(20)
            ],
            "task_requirements": {
                "task_10": {"flops": 12e9, "memory": 8.0, "dependencies": ["task_1", "task_5"]}
            },
        },
        cluster_state={
            "nodes": {
                "node_0_cpu_strong": {"cpu_capacity": 8.0, "memory_capacity": 32.0, "current_load": 0.9},
                "node_1_mem_large": {"cpu_capacity": 2.0, "memory_capacity": 64.0, "current_load": 0.3},
                "node_2_balanced": {"cpu_capacity": 4.0, "memory_capacity": 32.0, "current_load": 0.4},
            }
        },
        pending_tasks=[f"task_{i}" for i in range(20)],
        current_task="task_10",
        available_nodes=["node_0_cpu_strong", "node_1_mem_large", "node_2_balanced"],
        timestamp=0.0
    )

    print(f"--- Testing WASS-RAG predictions for task '{state.current_task}':")
    
    # --- Print individual node predictions for debugging ---
    print("\n--- Individual Node Predictions ---")
    node_makespans = {}
    for node in state.available_nodes:
        # Recreate the feature generation steps from the scheduler for an accurate test
        state_embedding = scheduler._extract_simple_features_fallback(state) if scheduler.base_scheduler is None else scheduler.base_scheduler._extract_simple_features(state)
        action_embedding = scheduler._encode_action(node, state)
        context = scheduler.knowledge_base.retrieve_similar_cases(state_embedding.cpu().numpy())
        
        predicted_makespan = scheduler._predict_performance(state_embedding, action_embedding, context)
        node_makespans[node] = predicted_makespan
        print(f"  - Predicted makespan for '{node}': {predicted_makespan:.2f}s")
    
    print("\n--- Final Scheduler Decision ---")
    decision = scheduler.make_decision(state)

    print(f"\n>>> Scheduling decision: {decision.task_id} -> {decision.target_node}")
    print(f"    Confidence: {decision.confidence:.3f}")
    print(f"    Reasoning: {decision.reasoning}")

if __name__ == "__main__":
    main()