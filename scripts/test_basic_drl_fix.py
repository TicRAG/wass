#!/usr/bin/env python3
"""
简化的DRL修复测试 - 最小依赖版本
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_import():
    """测试基本的导入和类实例化"""
    
    try:
        print("=== 基本导入测试 ===")
        
        # 1. 测试导入
        print("1. 测试导入AI调度器...")
        from src.ai_schedulers import WASSSmartScheduler, WASSRAGScheduler, SchedulingState, SchedulingAction
        print("   ✓ 成功导入所有类")
        
        # 2. 测试WASSSmartScheduler实例化
        print("\n2. 测试WASSSmartScheduler实例化...")
        try:
            smart_scheduler = WASSSmartScheduler("models/wass_models.pth")
            print(f"   ✓ 成功创建{smart_scheduler.name}调度器")
            
            # 检查关键方法是否存在
            if hasattr(smart_scheduler, '_build_graph_data'):
                print("   ✓ _build_graph_data方法存在")
            else:
                print("   ❌ _build_graph_data方法缺失")
                return False
                
        except Exception as e:
            print(f"   ❌ WASSSmartScheduler实例化失败: {e}")
            return False
        
        # 3. 测试WASSRAGScheduler实例化
        print("\n3. 测试WASSRAGScheduler实例化...")
        try:
            rag_scheduler = WASSRAGScheduler(
                model_path="models/wass_models.pth",
                knowledge_base_path="data/wass_knowledge_base.pkl"
            )
            print(f"   ✓ 成功创建{rag_scheduler.name}调度器")
            
            # 检查base_scheduler是否正确设置
            if hasattr(rag_scheduler, 'base_scheduler') and hasattr(rag_scheduler.base_scheduler, '_build_graph_data'):
                print("   ✓ base_scheduler._build_graph_data方法存在")
            else:
                print("   ❌ base_scheduler._build_graph_data方法缺失")
                return False
                
        except Exception as e:
            print(f"   ❌ WASSRAGScheduler实例化失败: {e}")
            return False
        
        # 4. 测试SchedulingState创建
        print("\n4. 测试SchedulingState创建...")
        try:
            state = SchedulingState(
                workflow_graph={"tasks": [], "name": "test"},
                cluster_state={"nodes": {}},
                pending_tasks=[],
                current_task="task_0",
                available_nodes=["node_0", "node_1"],
                timestamp=1234567890.0
            )
            print("   ✓ 成功创建SchedulingState")
        except Exception as e:
            print(f"   ❌ SchedulingState创建失败: {e}")
            return False
        
        # 5. 测试_build_graph_data方法调用
        print("\n5. 测试_build_graph_data方法调用...")
        try:
            graph_data = smart_scheduler._build_graph_data(state)
            print(f"   ✓ _build_graph_data调用成功，返回: {type(graph_data)}")
        except Exception as e:
            print(f"   ❌ _build_graph_data调用失败: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_import()
    
    if success:
        print("\n🎉 基本修复测试成功!")
        print("_build_graph_data方法已正确添加")
    else:
        print("\n❌ 仍有问题需要解决")
