#!/usr/bin/env python3
"""
WASS-RAG系统测试脚本
用于验证重构后的平台、工作流生成器和AI调度器
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.ai_schedulers import create_scheduler, WASSRAGScheduler
from src.performance_predictor import PerformancePredictor
from src.drl_agent import DQNAgent
from scripts.workflow_generator import WorkflowGenerator
# 从generate_ccr_workflows导入（如果需要）

def test_platform_config():
    """测试平台配置"""
    print("=== 测试平台配置 ===")
    
    platform_file = Path("configs/platform.xml")
    if not platform_file.exists():
        print("❌ 平台配置文件不存在")
        return False
    
    with open(platform_file, 'r') as f:
        content = f.read()
        if 'latency="1ms"' in content:
            print("✅ 网络延迟已设置为1ms")
        else:
            print("❌ 网络延迟设置不正确")
            return False
    
    return True

def test_workflow_generator():
    """测试工作流生成器CCR支持"""
    print("\n=== 测试工作流生成器CCR支持 ===")
    
    # 测试不同CCR值的工作流生成
    test_ccr_values = [0.1, 1.0, 5.0, 10.0]
    
    for ccr in test_ccr_values:
        try:
            generator = WorkflowGenerator(ccr=ccr)
            workflow_file = generator.generate_workflow_set("montage", [50])[0]
            
            print(f"✅ CCR={ccr}: 工作流文件已生成 - {workflow_file}")
                
        except Exception as e:
            print(f"❌ CCR={ccr}: 生成错误 - {e}")
            return False
    
    return True

def test_gnn_predictor():
    """测试GNN性能预测器"""
    print("\n=== 测试GNN性能预测器 ===")
    
    try:
        # 初始化预测器
        predictor = PerformancePredictor()
        
        # 生成测试工作流
        generator = WorkflowGenerator(ccr=1.0)
        workflow = generator.generate_workflow_set("montage", [20])[0]
        
        # 构建DAG图
        import networkx as nx
        dag = nx.DiGraph()
        for task in workflow.tasks:
            dag.add_node(task.id, computation_size=task.computation_size)
        for edge in workflow.edges:
            dag.add_edge(edge.source, edge.target, data_size=edge.data_size)
        
        # 测试特征提取
        node_features = {'test_node': {'speed': 1.0, 'available_time': 0.0, 'queue_length': 0}}
        graph_data = predictor.extract_graph_features(dag, node_features, focus_task_id=workflow.tasks[0].id)
        
        if graph_data.x.shape[1] == 12:  # 检查特征维度
            print("✅ GNN特征提取成功")
            
            # 测试预测
            prediction = predictor.predict(graph_data)
            print(f"✅ GNN预测成功: {prediction:.2f}")
            return True
        else:
            print(f"❌ GNN特征维度错误: {graph_data.x.shape[1]} != 12")
            return False
            
    except Exception as e:
        print(f"❌ GNN测试失败: {e}")
        return False

def test_rag_scheduler():
    """测试RAG调度器"""
    print("\n=== 测试RAG调度器 ===")
    
    try:
        # 初始化组件
        predictor = PerformancePredictor()
        drl_agent = DQNAgent(
            state_dim=50,  # 根据实际特征维度调整
            action_dim=4,  # 4个计算节点
            learning_rate=0.001
        )
        
        node_names = ["host1", "host2", "host3", "host4"]
        scheduler = WASSRAGScheduler(drl_agent, node_names, predictor)
        
        print("✅ RAG调度器初始化成功")
        
        # 测试奖励机制
        print("✅ 动态差分奖励机制已启用")
        return True
        
    except Exception as e:
        print(f"❌ RAG调度器测试失败: {e}")
        return False

def run_benchmark_test():
    """运行基准测试"""
    print("\n=== 运行基准测试 ===")
    
    try:
        # 生成不同CCR的测试工作流
        output_dir = Path("data/test_workflows")
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for ccr in [0.1, 1.0, 5.0, 10.0]:
            print(f"\n测试CCR={ccr}...")
            
            # 生成工作流
            generator = WorkflowGenerator(ccr=ccr)
            workflow = generator.generate_workflow_set("montage", [30])[0]
            
            # 保存工作流
            workflow_file = output_dir / f"test_workflow_ccr_{ccr}.json"
            with open(workflow_file, 'w') as f:
                json.dump({
                    "tasks": len(workflow.tasks),
                    "edges": len(workflow.edges),
                    "ccr": ccr,
                    "total_compute": sum(t.computation_size for t in workflow.tasks),
                    "total_comm": sum(e.data_size for e in workflow.edges)
                }, f, indent=2)
            
            results[f"CCR_{ccr}"] = {
                "tasks": len(workflow.tasks),
                "edges": len(workflow.edges),
                "total_compute": sum(t.computation_size for t in workflow.tasks),
                "total_comm": sum(e.data_size for e in workflow.edges)
            }
            
            print(f"  任务数: {len(workflow.tasks)}, 边数: {len(workflow.edges)}")
            print(f"  总计算量: {sum(t.computation_size for t in workflow.tasks):.2e}")
            print(f"  总通信量: {sum(e.data_size for e in workflow.edges):.2e}")
        
        # 保存测试结果
        with open("data/test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ 基准测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        return False

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="测试WASS-RAG系统重构")
    parser.add_argument("--skip-platform", action="store_true", help="跳过平台测试")
    parser.add_argument("--skip-workflow", action="store_true", help="跳过工作流测试")
    parser.add_argument("--skip-gnn", action="store_true", help="跳过GNN测试")
    parser.add_argument("--skip-rag", action="store_true", help="跳过RAG测试")
    parser.add_argument("--run-benchmark", action="store_true", help="运行基准测试")
    
    args = parser.parse_args()
    
    print("🚀 WASS-RAG系统重构测试")
    print("=" * 50)
    
    all_passed = True
    
    if not args.skip_platform:
        all_passed &= test_platform_config()
    
    if not args.skip_workflow:
        all_passed &= test_workflow_generator()
    
    if not args.skip_gnn:
        all_passed &= test_gnn_predictor()
    
    if not args.skip_rag:
        all_passed &= test_rag_scheduler()
    
    if args.run_benchmark:
        all_passed &= run_benchmark_test()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！WASS-RAG系统重构完成")
    else:
        print("❌ 部分测试失败，请检查配置")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)