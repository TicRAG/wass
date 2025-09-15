#!/bin/bash
# 使用真实HEFT和WASS-Heuristic案例的WASS-RAG实验脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_environment() {
    log_info "检查Python环境..."
    
    if ! python -c "import torch" 2>/dev/null; then
        log_error "PyTorch未安装"
        exit 1
    fi
    
    log_success "Python环境检查通过"
}

# 步骤1: 提取真实HEFT和WASS-Heuristic案例
extract_real_cases() {
    log_info "第1步: 从实验结果中提取真实HEFT和WASS-Heuristic案例..."
    
    if python scripts/extract_real_heuristic_cases.py; then
        log_success "真实案例提取完成"
        
        # 检查输出文件
        if [[ -f "data/heuristic_only_real_cases.json" ]]; then
            cases=$(python -c "import json; data=json.load(open('data/heuristic_only_real_cases.json')); print(len(data))")
            log_info "提取了 $cases 个真实案例"
        fi
    else
        log_error "真实案例提取失败"
        exit 1
    fi
}

# 步骤2: 更新RAG知识库
update_rag_kb() {
    log_info "第2步: 使用真实案例更新RAG知识库..."
    
    if python scripts/update_rag_with_real_cases.py; then
        log_success "RAG知识库更新完成"
        
        # 检查输出文件
        if [[ -f "data/real_heuristic_kb.json" ]]; then
            cases=$(python -c "import json; data=json.load(open('data/real_heuristic_kb.json')); print(len(data['cases']))")
            log_info "知识库包含 $cases 个真实案例"
        fi
    else
        log_error "RAG知识库更新失败"
        exit 1
    fi
}

# =================================================================
# 新增步骤: 训练DRL模型
# =================================================================
train_drl_model() {
    log_info "第3步: 训练适应当前环境的DRL模型..."
    
    # 确保DRL配置文件存在
    if [[ ! -f "configs/drl.yaml" ]]; then
        log_error "DRL配置文件 configs/drl.yaml 不存在!"
        exit 1
    fi

    # 运行改进版的DRL训练器
    if python scripts/improved_drl_trainer.py --config configs/drl.yaml; then
        log_success "DRL模型训练完成"
        
        # 检查输出模型文件
        if [[ -f "models/improved_wass_drl.pth" ]]; then
            log_info "新的DRL模型已保存到 models/improved_wass_drl.pth"
        fi
    else
        log_error "DRL模型训练失败"
        exit 1
    fi
}
# =================================================================

# 步骤4: 重新训练RAG模型 (原步骤3)
retrain_rag() {
    log_info "第4步: 使用真实案例重新训练RAG模型..."
    
    if python scripts/train_rag_wrench.py configs/rag.yaml; then
        log_success "RAG模型重新训练完成"
        
        # 检查输出文件
        if [[ -f "data/wrench_rag_knowledge_base.json" ]]; then
            cases=$(python -c "import json; data=json.load(open('data/wrench_rag_knowledge_base.json')); print(len(data['cases']))")
            log_info "训练后的RAG知识库包含 $cases 个案例"
        fi
    else
        log_warning "RAG模型重新训练失败，但继续执行后续步骤"
    fi
}

# 步骤5: 运行对比实验 (原步骤4)
run_comparison_experiments() {
    log_info "第5步: 运行使用真实案例的对比实验..."
    
    # 创建实验配置
    cat > configs/real_heuristic_experiment.yaml << EOF
# 真实案例实验配置
experiment:
  name: "real_heuristic_comparison"
  description: "使用真实HEFT和WASS-Heuristic案例的对比实验"

# 调度器配置
schedulers:
  - "HEFT"
  - "WASS-Heuristic"
  - "WASS-DRL"
  - "WASS-RAG"

# 实验规模
experiment_scale:
  num_workflows: 50
  workflow_sizes: [5, 10, 15, 20, 25]
  platforms: ["test_platform.xml"]

# RAG配置
rag:
  knowledge_base_path: "data/real_heuristic_kb.json"
  retriever: "wrench_similarity"
  top_k: 5
  fusion: "weighted"

# 评估配置
evaluation:
  metrics: ["makespan", "cpu_utilization", "load_balance"]
  output_dir: "results/real_heuristic_experiments"
  generate_charts: true
EOF
    
    # 运行实验
    if python experiments/wrench_real_experiment.py configs/real_heuristic_experiment.yaml; then
        log_success "对比实验运行完成"
        
        # 检查输出文件
        if [[ -f "results/real_heuristic_experiments/experiment_results.json" ]]; then
            log_info "实验结果已保存到 results/real_heuristic_experiments/"
        fi
    else
        log_error "对比实验运行失败"
        exit 1
    fi
}

# 步骤6: 生成结果摘要 (原步骤5)
generate_summary() {
    log_info "第6步: 生成实验结果摘要..."
    
    # 创建结果分析脚本
    cat > analyze_real_results.py << 'EOF'
import json
import os
import numpy as np

# 加载实验结果
results_path = "results/real_heuristic_experiments/experiment_results.json"
if not os.path.exists(results_path):
    print("实验结果文件不存在")
    exit(1)

with open(results_path, 'r') as f:
    results = json.load(f)

# 分析结果
scheduler_results = {}
for experiment in results.get("results", []):
    scheduler = experiment.get("scheduler", "unknown")
    makespan = experiment.get("makespan", 0)
    
    if scheduler not in scheduler_results:
        scheduler_results[scheduler] = []
    scheduler_results[scheduler].append(makespan)

# 计算平均性能
print("=== 使用真实案例的实验结果摘要 ===")
print()
print("调度器性能对比:")
print("-" * 40)

for scheduler, makespans in scheduler_results.items():
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    count = len(makespans)
    
    print(f"{scheduler:15} | 平均: {avg_makespan:8.2f}s | 标准差: {std_makespan:6.2f}s | 样本: {count:3d}")

print()
print("基于真实HEFT和WASS-Heuristic案例的RAG知识库已部署完成!")
EOF
    
    if python analyze_real_results.py; then
        rm analyze_real_results.py
        log_success "结果摘要生成完成"
    else
        log_error "结果摘要生成失败"
    fi
}

# 显示最终摘要
show_final_summary() {
    log_info "=============== 真实案例实验完成摘要 ==============="
    
    echo -e "${GREEN}知识库更新:${NC}"
    if [[ -f "data/real_heuristic_kb.json" ]]; then
        cases=$(python -c "import json; data=json.load(open('data/real_heuristic_kb.json')); print(len(data['cases']))")
        heft_cases=$(python -c "import json; data=json.load(open('data/real_heuristic_kb.json')); print(len([c for c in data['cases'] if c.get('scheduler_type') == 'HEFT']))")
        wass_cases=$(python -c "import json; data=json.load(open('data/real_heuristic_kb.json')); print(len([c for c in data['cases'] if c.get('scheduler_type') == 'WASS-Heuristic']))")
        echo "  • 总案例数: $cases 个"
        echo "  • HEFT案例: $heft_cases 个"
        echo "  • WASS-Heuristic案例: $wass_cases 个"
    fi
    
    echo -e "${GREEN}模型训练:${NC}"
    if [[ -f "models/improved_wass_drl.pth" ]]; then
        echo "  • DRL模型: models/improved_wass_drl.pth (已重新训练)"
    fi
    if [[ -f "data/wrench_rag_knowledge_base.json" ]]; then
        cases=$(python -c "import json; data=json.load(open('data/wrench_rag_knowledge_base.json')); print(len(data['cases']))")
        echo "  • RAG知识库: $cases 个案例"
    fi
    
    echo -e "${GREEN}实验结果:${NC}"
    if [[ -f "results/real_heuristic_experiments/experiment_results.json" ]]; then
        echo "  • 实验数据: results/real_heuristic_experiments/"
    fi
    
    log_success "使用真实案例的WASS-RAG实验流程执行完成! 🎉"
}

# 主函数
main() {
    log_info "开始 使用真实HEFT和WASS-Heuristic案例的WASS-RAG实验流程..."
    log_info "预计用时: 30-60分钟 (包含DRL模型训练)"
    echo
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 执行各个步骤
    check_environment
    extract_real_cases
    update_rag_kb
    train_drl_model  # <--- 调用新增的DRL训练函数
    retrain_rag
    run_comparison_experiments
    generate_summary
    
    # 计算总用时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo
    log_info "总执行时间: ${minutes}分${seconds}秒"
    
    # 显示结果摘要
    show_final_summary
}

# 检查命令行参数
if [[ $# -gt 0 ]]; then
    case $1 in
        "extract")
            extract_real_cases
            ;;
        "update")
            update_rag_kb
            ;;
        "train_drl") # <--- 新增的单独执行选项
            train_drl_model
            ;;
        "retrain")
            retrain_rag
            ;;
        "experiments")
            run_comparison_experiments
            ;;
        "summary")
            generate_summary
            ;;
        *)
            echo "用法: $0 [extract|update|train_drl|retrain|experiments|summary]"
            echo "无参数运行完整流程"
            exit 1
            ;;
    esac
else
    main
fi