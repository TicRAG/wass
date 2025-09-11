#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 工作流生成器
支持生成不同规模和复杂度的科学工作流，参考真实HPC应用场景
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Task:
    """工作流任务定义"""
    id: str
    name: str
    runtime: float  # 秒
    memory: int     # MB
    flops: float    # 浮点运算次数
    input_files: List[str]
    output_files: List[str]
    dependencies: List[str]  # 依赖的任务ID

@dataclass
class File:
    """文件定义"""
    id: str
    name: str
    size: int  # bytes

@dataclass
class Workflow:
    """完整工作流定义"""
    name: str
    description: str
    tasks: List[Task]
    files: List[File]
    entry_task: str
    exit_task: str

class WorkflowPattern:
    """工作流模式定义"""
    
    @staticmethod
    def generate_montage_like(num_tasks: int) -> Workflow:
        """生成类Montage（天文学图像拼接）工作流"""
        tasks = []
        files = []
        
        # 第一阶段：预处理任务（并行）
        preprocess_tasks = min(num_tasks // 3, 20)
        for i in range(preprocess_tasks):
            task_id = f"preprocess_{i}"
            input_file = f"raw_image_{i}.fits"
            output_file = f"processed_image_{i}.fits"
            
            files.append(File(input_file, input_file, random.randint(100, 500) * 1024 * 1024))  # 100-500MB
            files.append(File(output_file, output_file, random.randint(80, 400) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Preprocess Image {i}",
                runtime=random.uniform(300, 900),  # 5-15分钟
                memory=random.randint(2000, 4000),  # 2-4GB
                flops=random.uniform(1e10, 5e10),
                input_files=[input_file],
                output_files=[output_file],
                dependencies=[]
            ))
        
        # 第二阶段：差异检测（需要前阶段输出）
        diff_tasks = min((num_tasks - preprocess_tasks) // 2, 15)
        for i in range(diff_tasks):
            task_id = f"diff_{i}"
            # 随机选择两个预处理的输出作为输入
            deps = random.sample(tasks[:preprocess_tasks], 2)
            input_files = [dep.output_files[0] for dep in deps]
            output_file = f"diff_{i}.fits"
            
            files.append(File(output_file, output_file, random.randint(10, 50) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Difference Detection {i}",
                runtime=random.uniform(120, 480),  # 2-8分钟
                memory=random.randint(1000, 2000),
                flops=random.uniform(5e9, 2e10),
                input_files=input_files,
                output_files=[output_file],
                dependencies=[dep.id for dep in deps]
            ))
        
        # 第三阶段：最终拼接（需要所有前阶段输出）
        remaining_tasks = num_tasks - preprocess_tasks - diff_tasks
        for i in range(remaining_tasks):
            task_id = f"mosaic_{i}"
            # 需要所有差异检测的输出
            input_files = [task.output_files[0] for task in tasks if task.id.startswith('diff_')]
            output_file = f"final_mosaic_{i}.fits"
            
            files.append(File(output_file, output_file, random.randint(500, 1000) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Final Mosaic {i}",
                runtime=random.uniform(600, 1800),  # 10-30分钟
                memory=random.randint(4000, 8000),  # 4-8GB
                flops=random.uniform(2e10, 1e11),
                input_files=input_files,
                output_files=[output_file],
                dependencies=[task.id for task in tasks if task.id.startswith('diff_')]
            ))
        
        return Workflow(
            name=f"Montage-like-{num_tasks}",
            description=f"天文学图像拼接工作流，{num_tasks}个任务",
            tasks=tasks,
            files=files,
            entry_task=tasks[0].id,
            exit_task=tasks[-1].id
        )
    
    @staticmethod
    def generate_ligo_like(num_tasks: int) -> Workflow:
        """生成类LIGO（引力波检测）工作流"""
        tasks = []
        files = []
        
        # 数据切分阶段
        split_tasks = min(num_tasks // 4, 10)
        for i in range(split_tasks):
            task_id = f"split_{i}"
            input_file = f"raw_data_{i}.dat"
            output_files = [f"segment_{i}_{j}.dat" for j in range(4)]
            
            files.append(File(input_file, input_file, random.randint(1000, 2000) * 1024 * 1024))  # 1-2GB
            for out_file in output_files:
                files.append(File(out_file, out_file, random.randint(200, 400) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Data Split {i}",
                runtime=random.uniform(60, 180),
                memory=random.randint(1000, 2000),
                flops=random.uniform(1e9, 5e9),
                input_files=[input_file],
                output_files=output_files,
                dependencies=[]
            ))
        
        # 分析阶段（高并行度）
        analyze_tasks = num_tasks - split_tasks - split_tasks  # 剩余大部分用于分析
        for i in range(analyze_tasks):
            task_id = f"analyze_{i}"
            # 随机选择一个切分任务的输出
            split_task = random.choice(tasks[:split_tasks])
            input_file = random.choice(split_task.output_files)
            output_file = f"analysis_result_{i}.json"
            
            files.append(File(output_file, output_file, random.randint(1, 10) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Signal Analysis {i}",
                runtime=random.uniform(1200, 3600),  # 20-60分钟，计算密集
                memory=random.randint(3000, 6000),
                flops=random.uniform(5e10, 2e11),  # 高计算量
                input_files=[input_file],
                output_files=[output_file],
                dependencies=[split_task.id]
            ))
        
        # 汇总阶段
        for i in range(split_tasks):
            task_id = f"merge_{i}"
            # 收集所有分析结果
            input_files = [task.output_files[0] for task in tasks if task.id.startswith('analyze_')]
            output_file = f"detection_report_{i}.pdf"
            
            files.append(File(output_file, output_file, random.randint(5, 20) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Result Merge {i}",
                runtime=random.uniform(300, 900),
                memory=random.randint(2000, 4000),
                flops=random.uniform(1e10, 5e10),
                input_files=input_files,
                output_files=[output_file],
                dependencies=[task.id for task in tasks if task.id.startswith('analyze_')]
            ))
        
        return Workflow(
            name=f"LIGO-like-{num_tasks}",
            description=f"引力波检测工作流，{num_tasks}个任务",
            tasks=tasks,
            files=files,
            entry_task=tasks[0].id,
            exit_task=tasks[-1].id
        )
    
    @staticmethod
    def generate_cybershake_like(num_tasks: int) -> Workflow:
        """生成类CyberShake（地震模拟）工作流"""
        tasks = []
        files = []
        
        # 预处理阶段
        prep_tasks = min(num_tasks // 5, 8)
        for i in range(prep_tasks):
            task_id = f"prep_{i}"
            input_file = f"seismic_model_{i}.dat"
            output_file = f"preprocessed_model_{i}.dat"
            
            files.append(File(input_file, input_file, random.randint(200, 800) * 1024 * 1024))
            files.append(File(output_file, output_file, random.randint(150, 600) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Model Preprocessing {i}",
                runtime=random.uniform(600, 1200),
                memory=random.randint(2000, 4000),
                flops=random.uniform(2e10, 8e10),
                input_files=[input_file],
                output_files=[output_file],
                dependencies=[]
            ))
        
        # 仿真阶段（计算密集）
        sim_tasks = num_tasks - prep_tasks - prep_tasks
        for i in range(sim_tasks):
            task_id = f"simulate_{i}"
            # 依赖一个预处理任务
            prep_task = random.choice(tasks[:prep_tasks])
            input_file = prep_task.output_files[0]
            output_file = f"simulation_result_{i}.dat"
            
            files.append(File(output_file, output_file, random.randint(300, 1200) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Earthquake Simulation {i}",
                runtime=random.uniform(3600, 7200),  # 1-2小时，高计算量
                memory=random.randint(6000, 12000),  # 6-12GB
                flops=random.uniform(1e11, 5e11),  # 极高计算量
                input_files=[input_file],
                output_files=[output_file],
                dependencies=[prep_task.id]
            ))
        
        # 后处理阶段
        for i in range(prep_tasks):
            task_id = f"postprocess_{i}"
            # 收集部分仿真结果
            sim_subset = random.sample([task for task in tasks if task.id.startswith('simulate_')], 
                                     min(3, len([task for task in tasks if task.id.startswith('simulate_')])))
            input_files = [task.output_files[0] for task in sim_subset]
            output_file = f"hazard_map_{i}.png"
            
            files.append(File(output_file, output_file, random.randint(50, 200) * 1024 * 1024))
            
            tasks.append(Task(
                id=task_id,
                name=f"Hazard Analysis {i}",
                runtime=random.uniform(900, 1800),
                memory=random.randint(3000, 6000),
                flops=random.uniform(5e9, 2e10),
                input_files=input_files,
                output_files=[output_file],
                dependencies=[task.id for task in sim_subset]
            ))
        
        return Workflow(
            name=f"CyberShake-like-{num_tasks}",
            description=f"地震模拟工作流，{num_tasks}个任务",
            tasks=tasks,
            files=files,
            entry_task=tasks[0].id,
            exit_task=tasks[-1].id
        )

class WorkflowGenerator:
    """工作流生成器主类"""
    
    def __init__(self, output_dir: str = "data/workflows"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.patterns = {
            'montage': WorkflowPattern.generate_montage_like,
            'ligo': WorkflowPattern.generate_ligo_like,
            'cybershake': WorkflowPattern.generate_cybershake_like
        }
    
    def generate_workflow_set(self, pattern: str, task_counts: List[int]) -> List[str]:
        """生成一组不同规模的工作流"""
        if pattern not in self.patterns:
            raise ValueError(f"未知的工作流模式: {pattern}. 支持的模式: {list(self.patterns.keys())}")
        
        generated_files = []
        pattern_func = self.patterns[pattern]
        
        for count in task_counts:
            workflow = pattern_func(count)
            filename = f"{pattern}_{count}_tasks.json"
            filepath = self.output_dir / filename
            
            # 保存为JSON格式
            workflow_dict = {
                'metadata': {
                    'name': workflow.name,
                    'description': workflow.description,
                    'generated_at': datetime.now().isoformat(),
                    'task_count': len(workflow.tasks),
                    'file_count': len(workflow.files)
                },
                'workflow': {
                    'tasks': [asdict(task) for task in workflow.tasks],
                    'files': [asdict(file) for file in workflow.files],
                    'entry_task': workflow.entry_task,
                    'exit_task': workflow.exit_task
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(workflow_dict, f, indent=2, ensure_ascii=False)
            
            generated_files.append(str(filepath))
            print(f"✅ 生成工作流: {filename} ({count} 任务)")
        
        return generated_files
    
    def generate_all_scales(self) -> Dict[str, List[str]]:
        """生成所有规模的标准工作流集合"""
        # 定义标准规模集合
        small_scale = [10, 20, 30, 50]        # 小规模：快速测试
        medium_scale = [100, 200, 300, 500]   # 中等规模：常规实验
        large_scale = [1000, 1500, 2000]      # 大规模：可扩展性测试
        
        all_files = {}
        
        # 为每种模式生成不同规模
        for pattern in self.patterns:
            print(f"\n🚀 生成 {pattern.upper()} 模式工作流...")
            
            pattern_files = []
            pattern_files.extend(self.generate_workflow_set(pattern, small_scale))
            pattern_files.extend(self.generate_workflow_set(pattern, medium_scale))
            pattern_files.extend(self.generate_workflow_set(pattern, large_scale))
            
            all_files[pattern] = pattern_files
            print(f"📊 {pattern} 模式完成: {len(pattern_files)} 个工作流")
        
        return all_files
    
    def generate_summary(self, generated_files: Dict[str, List[str]]) -> str:
        """生成工作流集合摘要"""
        summary_path = self.output_dir / "workflow_summary.json"
        
        summary = {
            'generation_info': {
                'generated_at': datetime.now().isoformat(),
                'total_patterns': len(generated_files),
                'total_workflows': sum(len(files) for files in generated_files.values())
            },
            'patterns': {}
        }
        
        for pattern, files in generated_files.items():
            workflow_info = []
            for file_path in files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    workflow_info.append({
                        'filename': Path(file_path).name,
                        'task_count': data['metadata']['task_count'],
                        'file_count': data['metadata']['file_count']
                    })
            
            summary['patterns'][pattern] = {
                'count': len(files),
                'workflows': workflow_info
            }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(summary_path)

def main():
    parser = argparse.ArgumentParser(description='WASS-RAG 工作流生成器')
    parser.add_argument('--pattern', choices=['montage', 'ligo', 'cybershake', 'all'], 
                       default='all', help='工作流模式')
    parser.add_argument('--tasks', nargs='+', type=int, 
                       help='任务数量列表，例如：--tasks 50 100 200')
    parser.add_argument('--output', default='data/workflows', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    generator = WorkflowGenerator(args.output)
    
    if args.pattern == 'all':
        print("🌟 生成完整工作流集合...")
        generated_files = generator.generate_all_scales()
        summary_path = generator.generate_summary(generated_files)
        
        print(f"\n📋 工作流摘要已保存: {summary_path}")
        print(f"🎉 总计生成 {sum(len(files) for files in generated_files.values())} 个工作流文件")
        
    else:
        if not args.tasks:
            args.tasks = [50, 100, 200]  # 默认规模
            
        print(f"🚀 生成 {args.pattern} 模式工作流...")
        files = generator.generate_workflow_set(args.pattern, args.tasks)
        
        print(f"✅ 完成! 生成了 {len(files)} 个工作流文件")
        for file_path in files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    main()
