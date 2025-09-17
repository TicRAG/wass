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

from workflow_manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.drl.knowledge_teacher import KnowledgeBase
from src.drl.utils import workflow_json_to_pyg_data
from src.wrench_schedulers import RecordingHEFTScheduler
from src.utils import WrenchExperimentRunner

# --- Configuration ---
GNN_IN_CHANNELS = 3
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS = 32
KB_DIMENSION = GNN_OUT_CHANNELS
PLATFORM_FILE = "configs/test_platform.xml"
WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"
FEATURE_SCALER_PATH = "models/saved_models/feature_scaler.joblib" # Path for the new scaler

def main():
    print("🚀 [Phase 1] Starting Knowledge Base Seeding Process (with Feature Scaling)...")
    
    # --- Initialization ---
    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    gnn_encoder = GNNEncoder(GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS)
    knowledge_base = KnowledgeBase(dimension=KB_DIMENSION)
    config_params = {"platform_file": PLATFORM_FILE}
    wrench_runner = WrenchExperimentRunner(schedulers={}, config=config_params)
    print("✅ Components initialized.")

    # --- Generate Workflows ---
    seeding_workflows = workflow_manager.generate_training_workflows()
    print(f"✅ Generated {len(seeding_workflows)} workflows.")

    # --- Step 3a: Extract ALL node features first to fit the scaler ---
    print("\n[Step 3a/6] Extracting all node features to fit the scaler...")
    all_node_features = []
    successful_workflows = []
    for wf_file in seeding_workflows:
        try:
            with open(wf_file, 'r') as f:
                wf_data = json.load(f)
            # This loop extracts features from every task in every workflow file
            for task in wf_data['workflow']['tasks']:
                features = [
                    float(task.get('runtime', 0.0)),
                    float(task.get('flops', 0.0)),
                    float(task.get('memory', 0.0))
                ]
                all_node_features.append(features)
            successful_workflows.append(wf_file)
        except Exception:
            continue # Skip corrupted files

    if not all_node_features:
        print("❌ Could not extract any features. Aborting.")
        return

    feature_scaler = StandardScaler()
    feature_scaler.fit(all_node_features)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    print(f"✅ Feature scaler fitted and saved to {FEATURE_SCALER_PATH}")

    # --- Simulate Workflows ---
    print("\n[Step 3b/6] Simulating workflows with HEFT...")
    all_embeddings = []
    all_metadata = []

    for i, wf_file in enumerate(successful_workflows):
        wf_path = Path(wf_file)
        print(f"\n--- Processing workflow {i+1}/{len(successful_workflows)}: {wf_path.name} ---")
        
        makespan, decisions = wrench_runner.run_single_seeding_simulation(
            scheduler_class=RecordingHEFTScheduler,
            workflow_file=str(wf_path)
        )

        if makespan < 0:
            print(f"❌ Simulation failed for {wf_path.name}. Skipping.")
            continue
        
        print(f"  ⏹️ Simulation finished. Makespan: {makespan:.2f}s.")
        
        # --- Encode graph using the SCALER ---
        try:
            # Pass the fitted scaler to the conversion function
            pyg_data = workflow_json_to_pyg_data(str(wf_path), feature_scaler)
            graph_embedding = gnn_encoder(pyg_data)
        except Exception as e:
            print(f"❌ Error encoding workflow {wf_path.name} to graph: {e}")
            continue

        all_embeddings.append(graph_embedding.detach().numpy().flatten())
        all_metadata.append({
            "workflow_file": wf_path.name,
            "makespan": makespan,
            "decisions": json.dumps(decisions) 
        })
            
    # --- Finalize ---
    if not all_embeddings:
        print("❌ No experience was collected. Cannot build knowledge base.")
        return
        
    knowledge_base.add(np.array(all_embeddings), all_metadata)
    knowledge_base.save()
    print(f"\n✅ Knowledge Base saved successfully with {len(all_embeddings)} entries.")
    print("\n🎉 [Phase 1] Completed! 🎉")

if __name__ == "__main__":
    main()