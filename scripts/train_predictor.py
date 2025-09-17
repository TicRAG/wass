# scripts/train_predictor.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import joblib # For saving the scaler object
from sklearn.preprocessing import StandardScaler

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from src.drl.knowledge_teacher import PerformancePredictor
from src.drl.knowledge_teacher import KnowledgeBase

# --- Configuration ---
KB_PATH = "data/knowledge_base"
MODEL_SAVE_DIR = "models/saved_models"
PREDICTOR_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "performance_predictor.pth")
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "makespan_scaler.joblib") # Path to save the scaler

STATE_DIM = 32
EPOCHS = 150 # Increased epochs for the larger dataset
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def main():
    """Main function to train and save the performance predictor."""
    print("🚀 [Phase 2.2] Starting Performance Predictor Training...")
    
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ Model save directory ensured at: {MODEL_SAVE_DIR}")

    print(f"\n[Step 1/5] Loading Knowledge Base from {KB_PATH}...")
    knowledge_base = KnowledgeBase(dimension=STATE_DIM, storage_path=KB_PATH)
    embeddings = knowledge_base.index.reconstruct_n(0, knowledge_base.index.ntotal)
    metadata = knowledge_base.metadata
    
    if metadata.empty or embeddings.shape[0] < BATCH_SIZE:
        print(f"❌ Knowledge Base is too small ({len(metadata)} entries). Please generate more data by running seed_knowledge_base.py.")
        return
    
    print(f"✅ Loaded {len(metadata)} entries from the Knowledge Base.")

    print("\n[Step 2/5] Normalizing target values (makespans)...")
    makespans = metadata['makespan'].values.reshape(-1, 1)
    scaler = StandardScaler()
    normalized_makespans = scaler.fit_transform(makespans)
    print(f"✅ Makespans normalized. Original Mean: {np.mean(makespans):.2f}, New Mean: {np.mean(normalized_makespans):.2f}")

    print("\n[Step 3/5] Preparing dataset for PyTorch...")
    X_train = torch.tensor(embeddings, dtype=torch.float32)
    y_train = torch.tensor(normalized_makespans, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Dataset ready. Number of samples: {len(dataset)}")

    print("\n[Step 4/5] Initializing model and optimizer...")
    predictor_model = PerformancePredictor(state_dim=STATE_DIM)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(predictor_model.parameters(), lr=LEARNING_RATE)
    print(f"✅ Model initialized.")

    print(f"\n[Step 5/5] Starting training for {EPOCHS} epochs...")
    predictor_model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = predictor_model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], Average Normalized Loss: {avg_loss:.6f}")

    print("✅ Training finished.")

    torch.save(predictor_model.state_dict(), PREDICTOR_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH) # Save the scaler
    print(f"\n💾 Trained Performance Predictor model saved to: {PREDICTOR_MODEL_PATH}")
    print(f"💾 Makespan Normalization Scaler saved to: {SCALER_PATH}")
    print("\n🎉 [Phase 2.2] Completed! 🎉")

if __name__ == "__main__":
    main()