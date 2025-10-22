# scripts/seed_knowledge_base.py
import os
import sys
import json
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.workflows.manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.rag.teacher import KnowledgeBase
from src.drl.utils import workflow_json_to_pyg_data
# --- FINAL FIX: Import the new Recording Schedulers ---
from src.simulation.schedulers import RecordingHEFTScheduler, RecordingRandomScheduler
from src.simulation.experiment_runner import WrenchExperimentRunner

# Configuration
GNN_IN_CHANNELS = 4
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS = 32
KB_DIMENSION = GNN_OUT_CHANNELS
PLATFORM_FILE = "configs/test_platform.xml"
WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"
FEATURE_SCALER_PATH = "models/saved_models/feature_scaler.joblib"

def main():
    print("🚀 [Phase 1] Starting Knowledge Base Seeding (with Decision Recording)...")
    
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    knowledge_base = KnowledgeBase(dimension=KB_DIMENSION)
    config_params = {"platform_file": PLATFORM_FILE}
    wrench_runner = WrenchExperimentRunner(schedulers={}, config=config_params)
    print("✅ Components initialized.")

    seeding_workflows = workflow_manager.generate_training_workflows()
    print(f"✅ Generated {len(seeding_workflows)} workflows.")

    print("\n[Step 3a/6] Extracting features to fit the scaler...")
    all_node_features = []
    for wf_file in seeding_workflows:
        try:
            with open(wf_file, 'r') as f:
                wf_data = json.load(f)
            for task in wf_data['workflow']['tasks']:
                all_node_features.append([
                    float(task.get('runtime', 0.0)),
                    float(task.get('flops', 0.0)),
                    float(task.get('memory', 0.0))
                ])
        except Exception:
            continue

    if not all_node_features:
        print("❌ Could not extract any features. Aborting.")
        return

    feature_scaler = StandardScaler()
    feature_scaler.fit(all_node_features)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    print(f"✅ Feature scaler fitted and saved to {FEATURE_SCALER_PATH}")

    # --- FINAL FIX: Use the new Recording Schedulers ---
    seeding_schedulers = {
        "HEFT": RecordingHEFTScheduler,
        "Random": RecordingRandomScheduler
    }
    print(f"\n[Step 3b/6] Simulating workflows with schedulers: {list(seeding_schedulers.keys())}...")
    
    all_embeddings, all_metadata = [], []

    for scheduler_name, scheduler_class in seeding_schedulers.items():
        print(f"\n--- Seeding with {scheduler_name} Scheduler ---")
        for i, wf_file in enumerate(seeding_workflows):
            wf_path = Path(wf_file)
            print(f"  Processing workflow {i+1}/{len(seeding_workflows)}: {wf_path.name}")
            
            makespan, decisions = wrench_runner.run_single_seeding_simulation(
                scheduler_class=scheduler_class,
                workflow_file=str(wf_path)
            )

            if makespan < 0:
                print(f"  ❌ Simulation failed for {wf_path.name}. Skipping.")
                continue
            
            print(f"  ⏹️ Simulation finished. Makespan: {makespan:.2f}s.")
            
            try:
                pyg_data = workflow_json_to_pyg_data(str(wf_path), feature_scaler)
                graph_embedding = gnn_encoder(pyg_data)
            except Exception as e:
                print(f"  ❌ Error encoding workflow {wf_path.name} to graph: {e}")
                continue

            all_embeddings.append(graph_embedding.detach().numpy().flatten())
            all_metadata.append({
                "workflow_file": wf_path.name,
                "makespan": makespan,
                "scheduler_used": scheduler_name,
                "decisions": json.dumps(decisions) # This will now contain rich data
            })
            
    if not all_embeddings:
        print("❌ No experience was collected. Cannot build knowledge base.")
        return
        
    knowledge_base.add(np.array(all_embeddings), all_metadata)
    knowledge_base.save()
    print(f"\n✅ Knowledge Base saved with {len(all_embeddings)} entries from {len(seeding_schedulers)} schedulers.")
    print("\n🎉 [Phase 1] Completed! 🎉")

if __name__ == "__main__":
    main()