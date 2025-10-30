import os
import sys
import json
import time
from pathlib import Path
import re
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.workflows.manager import WorkflowManager
from src.drl.gnn_encoder import GNNEncoder
from src.rag.teacher import KnowledgeBase
from src.simulation.experiment_runner import WrenchExperimentRunner
from src.simulation.schedulers import RecordingHEFTScheduler, RecordingRandomScheduler, RecordingMinMinScheduler

WORKFLOW_CONFIG_FILE = "configs/workflow_config.yaml"
FEATURE_SCALER_PATH = "models/saved_models/feature_scaler.joblib"
SEED_GNN_WEIGHTS_PATH = "models/saved_models/gnn_encoder_kb.pth"
KB_DIM = 32  # Will be validated against encoder out_channels
TRAINING_WORKFLOWS_DIR = Path("data/workflows/training")
AUGMENTED_WORKFLOWS_DIR = Path("data/workflows/training_aug")

# Reseed options
SCHEDULERS_TO_RUN = ["HEFT", "MINMIN", "Random"]  # Order matters for deterministic logging
LIMIT_WORKFLOWS = None  # Set to an int for quick debug
PRINT_PROGRESS_EVERY = 1


def load_or_fit_feature_scaler(workflow_files):
    if Path(FEATURE_SCALER_PATH).exists():
        try:
            scaler = joblib.load(FEATURE_SCALER_PATH)
            print(f"✅ Loaded existing feature scaler: {FEATURE_SCALER_PATH}")
            return scaler
        except Exception as e:
            print(f"⚠️ Failed to load scaler ({e}), will refit.")
    print("🔧 Fitting feature scaler from WFCommons task features...")
    all_features = []
    for wf in workflow_files:
        try:
            with open(wf, 'r') as f:
                data = json.load(f)
            wf_section = data.get('workflow', {})
            tasks = wf_section.get('tasks') or wf_section.get('specification', {}).get('tasks', [])
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                all_features.append([
                    float(t.get('runtime', 0.0)),
                    float(t.get('flops', 0.0)),
                    float(t.get('memory', 0.0))
                ])
        except Exception:
            continue
    if not all_features:
        raise RuntimeError("No task features extracted; cannot fit scaler.")
    scaler = StandardScaler()
    scaler.fit(all_features)
    Path(FEATURE_SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, FEATURE_SCALER_PATH)
    print(f"✅ Feature scaler fitted and saved to {FEATURE_SCALER_PATH}")
    return scaler


def main():
    start_time = time.time()
    print("🚀 Reseeding Knowledge Base with WFCommons workflows (real task names)...")

    if not TRAINING_WORKFLOWS_DIR.exists():
        print(f"❌ Training workflows directory missing: {TRAINING_WORKFLOWS_DIR}")
        return
    workflow_files = sorted(str(p) for p in TRAINING_WORKFLOWS_DIR.glob("*.json"))
    # Include augmented variants if present
    if AUGMENTED_WORKFLOWS_DIR.exists():
        aug_files = sorted(str(p) for p in AUGMENTED_WORKFLOWS_DIR.glob("*.json"))
        if aug_files:
            print(f"➕ Including {len(aug_files)} augmented workflow variants from {AUGMENTED_WORKFLOWS_DIR}.")
            workflow_files.extend(aug_files)
    if not workflow_files:
        print(f"❌ No workflow JSON files found in {TRAINING_WORKFLOWS_DIR}")
        return
    if LIMIT_WORKFLOWS:
        workflow_files = workflow_files[:LIMIT_WORKFLOWS]
    print(f"📁 Found {len(workflow_files)} training workflows.")

    workflow_manager = WorkflowManager(WORKFLOW_CONFIG_FILE)
    platform_file = workflow_manager.get_platform_file()

    # Build components
    gnn_encoder = GNNEncoder(in_channels=4, hidden_channels=64, out_channels=KB_DIM)
    if Path(SEED_GNN_WEIGHTS_PATH).exists():
        try:
            gnn_encoder.load_state_dict(torch.load(SEED_GNN_WEIGHTS_PATH, map_location=torch.device('cpu')))
            print(f"🔐 Loaded seed GNN weights from {SEED_GNN_WEIGHTS_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load seed GNN weights: {e}")
    gnn_encoder.eval()
    for param in gnn_encoder.parameters():
        param.requires_grad = False

    kb = KnowledgeBase(dimension=KB_DIM)
    # Reset KB (start fresh)
    if kb.index.ntotal > 0 or not kb.metadata.empty:
        print("🧹 Clearing existing KB index & metadata...")
        kb.index = type(kb.index)(KB_DIM)  # new flat index
        kb.metadata = kb.metadata.iloc[0:0]

    scaler = load_or_fit_feature_scaler(workflow_files)

    runner = WrenchExperimentRunner(schedulers={}, config={"platform_file": platform_file})
    scheduler_map = {
        "HEFT": RecordingHEFTScheduler,
        "MINMIN": RecordingMinMinScheduler,
        "Random": RecordingRandomScheduler
    }
    scheduler_kwargs = {
        'knowledge_encoder': gnn_encoder,
        'feature_scaler': scaler,
    }

    all_embeddings = []
    all_metadata = []

    for sched_name in SCHEDULERS_TO_RUN:
        sched_cls = scheduler_map.get(sched_name)
        if not sched_cls:
            print(f"⚠️ Unknown scheduler '{sched_name}', skipping.")
            continue
        print(f"\n=== Seeding with {sched_name} Scheduler ===")
        for i, wf in enumerate(workflow_files, 1):
            wf_path = Path(wf)
            if i % PRINT_PROGRESS_EVERY == 0:
                print(f"[{sched_name}] {i}/{len(workflow_files)} -> {wf_path.name}")
            makespan, details = runner.run_single_seeding_simulation(
                scheduler_class=sched_cls,
                workflow_file=str(wf_path),
                scheduler_kwargs=scheduler_kwargs,
            )
            knowledge_records = details.get('knowledge_records', []) if isinstance(details, dict) else []
            if makespan < 0:
                print(f"  ❌ Simulation failed for {wf_path.name}, skipping.")
                continue
            # Augment decisions with remaining makespan estimates
            if not knowledge_records:
                print(f"  ⚠️ No knowledge records captured for {wf_path.name} using {sched_name}.")
                continue
            aug_suffix_pattern = r'_AUG\d+$'
            for record in knowledge_records:
                embedding_list = record.get('embedding', [])
                if not embedding_list:
                    continue
                embedding_vec = np.asarray(embedding_list, dtype=np.float32)
                decision_time = float(record.get('decision_time', 0.0))
                remaining_makespan = record.get('remaining_makespan')
                if remaining_makespan is None:
                    remaining_makespan = max(makespan - decision_time, 0.0)
                normalized_q = (remaining_makespan / makespan) if makespan > 0 else 0.0
                metadata_entry = {
                    "workflow_file": wf_path.name,
                    "scheduler_used": sched_name,
                    "makespan": makespan,
                    "task_name": record.get('task_name'),
                    "host": record.get('host'),
                    "decision_time": decision_time,
                    "remaining_makespan": remaining_makespan,
                    "q_value": normalized_q,
                }
                all_embeddings.append(embedding_vec)
                all_metadata.append(metadata_entry)
                task_name = record.get('task_name') or ""
                if re.search(aug_suffix_pattern, task_name or ""):
                    base_name = re.sub(aug_suffix_pattern, '', task_name)
                    metadata_entry_base = {
                        **metadata_entry,
                        "task_name": base_name,
                    }
                    all_embeddings.append(embedding_vec.copy())
                    all_metadata.append(metadata_entry_base)
    if not all_embeddings:
        print("❌ No embeddings collected; aborting KB save.")
        return

    kb.add(np.array(all_embeddings, dtype=np.float32), all_metadata)
    kb.save()

    # Save GNN encoder (update seed for consistency)
    Path(SEED_GNN_WEIGHTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gnn_encoder.state_dict(), SEED_GNN_WEIGHTS_PATH)
    print(f"💾 Saved updated seed GNN weights to {SEED_GNN_WEIGHTS_PATH}")

    elapsed = time.time() - start_time
    print(f"\n✅ Reseeding complete: {len(all_embeddings)} entries across {len(SCHEDULERS_TO_RUN)} schedulers (elapsed {elapsed:.2f}s).")
    print("Next: run a short training to verify non-zero RAG rewards.")

if __name__ == "__main__":
    main()