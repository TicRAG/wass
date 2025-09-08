#!/usr/bin/env python3
"""
快速测试模型训练修复
重新训练并验证效果
"""

import os
import sys
import subprocess
import time

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def test_training_fix():
    """测试训练修复"""
    
    print("=== Testing Training Fix ===")
    print("This will retrain the model with fixed architecture and parameters\n")
    
    # 运行训练脚本
    print("1. Running model training...")
    try:
        result = subprocess.run([
            sys.executable, "scripts/initialize_ai_models.py"
        ], cwd=parent_dir, capture_output=True, text=True, timeout=300)
        
        print("Training output:")
        print(result.stdout)
        if result.stderr:
            print("Training errors:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"✗ Training failed with return code {result.returncode}")
            return False
        
    except subprocess.TimeoutExpired:
        print("✗ Training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # 分析训练结果
    print("\n2. Analyzing training results...")
    
    # 检查关键指标
    output = result.stdout
    
    # 提取R²值
    r2_found = False
    final_r2 = None
    for line in output.split('\n'):
        if "R²:" in line and "PerformancePredictor" in line:
            try:
                r2_str = line.split("R²:")[1].split()[0]
                final_r2 = float(r2_str)
                r2_found = True
                break
            except:
                pass
    
    # 提取预测多样性
    diversity_found = False
    prediction_std = None
    for line in output.split('\n'):
        if "Prediction diversity:" in line:
            try:
                std_str = line.split("diversity:")[1].split()[0]
                prediction_std = float(std_str)
                diversity_found = True
                break
            except:
                pass
    
    # 检查是否有CRITICAL错误
    has_critical_error = "CRITICAL: Model produces identical predictions!" in output
    
    # 生成报告
    print(f"\n=== Training Analysis ===")
    
    if r2_found:
        print(f"R² Score: {final_r2:.4f}")
        if final_r2 > 0.7:
            print("✅ Excellent model fit")
        elif final_r2 > 0.3:
            print("⚠️  Moderate model fit")
        else:
            print("❌ Poor model fit")
    else:
        print("❌ Could not extract R² score")
    
    if diversity_found:
        print(f"Prediction Diversity: {prediction_std:.2f}")
        if prediction_std > 5.0:
            print("✅ Good prediction diversity")
        elif prediction_std > 1.0:
            print("⚠️  Moderate prediction diversity")
        else:
            print("❌ Low prediction diversity")
    else:
        print("❌ Could not extract prediction diversity")
    
    if has_critical_error:
        print("❌ CRITICAL: Model still produces identical predictions")
        success = False
    else:
        print("✅ No critical errors detected")
        success = r2_found and final_r2 > 0.3 and diversity_found and prediction_std > 1.0
    
    return success

def test_rag_scheduler():
    """快速测试RAG调度器"""
    
    print(f"\n=== Testing RAG Scheduler ===")
    
    try:
        from src.ai_schedulers import create_scheduler
        
        # 创建调度器
        rag_scheduler = create_scheduler(
            "WASS-RAG",
            model_path="models/wass_models.pth",
            knowledge_base_path="data/knowledge_base.pkl"
        )
        
        print("✅ RAG scheduler loaded successfully")
        
        # 检查是否有归一化参数
        if hasattr(rag_scheduler, '_y_mean') and hasattr(rag_scheduler, '_y_std'):
            print(f"✅ Normalization parameters loaded: mean={rag_scheduler._y_mean:.2f}, std={rag_scheduler._y_std:.2f}")
            return True
        else:
            print("⚠️  Normalization parameters not found")
            return False
            
    except Exception as e:
        print(f"✗ RAG scheduler test failed: {e}")
        return False

if __name__ == "__main__":
    print("WASS-RAG Training Fix Test")
    print("This script tests the fixes for model training issues\n")
    
    # 测试训练
    training_success = test_training_fix()
    
    # 测试调度器
    scheduler_success = test_rag_scheduler()
    
    # 最终结果
    print(f"\n{'='*60}")
    
    if training_success and scheduler_success:
        print("🎉 SUCCESS: Training fix appears to work!")
        print("   Model now shows good R² and prediction diversity")
        print("   RAG scheduler loaded with normalization parameters")
        print("\n   Next step: Run full experiments to confirm fix")
        print("   Command: python experiments/real_experiment_framework.py")
    elif training_success:
        print("⚠️  PARTIAL SUCCESS: Training improved but scheduler issues remain")
        print("   Model training is better but may need scheduler fixes")
    else:
        print("❌ FAILURE: Training issues persist")
        print("   Additional debugging needed")
        
        # 提供诊断建议
        print(f"\n   Debugging steps:")
        print(f"   1. Run: python scripts/diagnose_training.py")
        print(f"   2. Check data quality and feature generation")
        print(f"   3. Verify model architecture changes")
    
    print(f"{'='*60}")
