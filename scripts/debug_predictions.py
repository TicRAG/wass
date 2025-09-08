#!/usr/bin/env python3
"""
调试RAG调度器的实际预测值
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from src.ai_schedulers import WASSRAGScheduler, PerformancePredictor
    print("✓ Successfully imported schedulers")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_performance_predictor():
    """直接测试PerformancePredictor的输出"""
    
    print("=== Testing PerformancePredictor Directly ===")
    
    try:
        # 加载模型
        checkpoint = torch.load("models/wass_models.pth", map_location="cpu")
        
        model = PerformancePredictor(input_dim=96, hidden_dim=128)
        model.load_state_dict(checkpoint["performance_predictor"])
        model.eval()
        
        # 获取归一化参数
        metadata = checkpoint["metadata"]["performance_predictor"]
        y_mean = metadata["y_mean"]
        y_std = metadata["y_std"]
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Normalization params: mean={y_mean:.2f}, std={y_std:.2f}")
        
        # 生成测试特征向量
        print(f"\nTesting with different feature vectors:")
        
        for i in range(5):
            # 生成随机特征向量（96维）
            features = torch.randn(96)
            
            with torch.no_grad():
                # 模型预测（归一化值）
                pred_normalized = model(features).item()
                
                # 反归一化
                pred_denormalized = pred_normalized * y_std + y_mean
                
                print(f"Test {i+1}:")
                print(f"  Normalized prediction: {pred_normalized:.6f}")
                print(f"  Denormalized prediction: {pred_denormalized:.2f}")
                
                # 检查是否会触发降级逻辑
                trigger_condition = abs(pred_normalized) < 0.2 and abs(pred_normalized + 0.1) < 0.05
                if trigger_condition:
                    print(f"  ⚠️  Would trigger degradation logic!")
                else:
                    print(f"  ✓ Normal prediction")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test predictor: {e}")
        return False

def test_rag_scheduler_prediction():
    """测试RAG调度器中的预测过程"""
    
    print(f"\n=== Testing RAG Scheduler Prediction Process ===")
    
    try:
        # 创建RAG调度器
        rag_scheduler = WASSRAGScheduler(
            model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        
        print(f"✓ RAG scheduler created")
        print(f"✓ Normalization params: mean={rag_scheduler._y_mean:.2f}, std={rag_scheduler._y_std:.2f}")
        
        # 创建假的输入
        state_embedding = torch.randn(32)
        action_embedding = torch.randn(32) 
        context = {"similar_cases": []}  # 空上下文
        
        # 调用预测函数
        print(f"\nTesting _predict_performance method:")
        predicted_makespan = rag_scheduler._predict_performance(
            state_embedding, action_embedding, context
        )
        
        print(f"Final predicted makespan: {predicted_makespan:.2f}")
        
        # 多次测试看是否有变化
        print(f"\nTesting prediction diversity:")
        predictions = []
        for i in range(10):
            # 稍微不同的输入
            state_emb = torch.randn(32)
            action_emb = torch.randn(32)
            
            pred = rag_scheduler._predict_performance(state_emb, action_emb, context)
            predictions.append(pred)
            print(f"  Prediction {i+1}: {pred:.2f}")
        
        # 分析多样性
        pred_std = np.std(predictions)
        pred_range = max(predictions) - min(predictions)
        unique_preds = len(set([round(p, 2) for p in predictions]))
        
        print(f"\nPrediction analysis:")
        print(f"  Standard deviation: {pred_std:.2f}")
        print(f"  Range: {pred_range:.2f}")
        print(f"  Unique predictions (rounded): {unique_preds}/10")
        
        if pred_std > 1.0:
            print(f"  ✅ Good prediction diversity")
            return True
        else:
            print(f"  ❌ Low prediction diversity")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test RAG scheduler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("WASS-RAG Prediction Debugging")
    print("="*50)
    
    # 测试PerformancePredictor
    predictor_ok = test_performance_predictor()
    
    # 测试RAG调度器
    scheduler_ok = test_rag_scheduler_prediction()
    
    print(f"\n{'='*50}")
    if predictor_ok and scheduler_ok:
        print("🎉 SUCCESS: Both tests passed!")
        print("   The issue may be elsewhere in the scheduling pipeline")
    elif predictor_ok:
        print("⚠️  PARTIAL: Predictor works but scheduler has issues")
        print("   Check _predict_performance implementation")
    else:
        print("❌ FAILURE: Core predictor issues detected")
        print("   Model loading or architecture problems")
    
    print(f"{'='*50}")
