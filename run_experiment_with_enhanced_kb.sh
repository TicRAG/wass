#!/bin/bash
# 使用已有增强RAG知识库的实验脚本

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
    
    if ! python -c "import wrench" 2>/dev/null; then
        log_error "WRENCH未安装或未激活虚拟环境"
        log_info "请运行: source wrench-env/bin/activate"
        exit 1
    fi
    
    if ! python -c "import torch" 2>/dev/null; then
        log_error "PyTorch未安装"
        log_info "请运行: pip install torch"
        exit 1
    fi
    
    log_success "Python环境检查通过"
}

# 验证WRENCH环境
verify_wrench() {
    log_info "验证WRENCH环境..."
    if python wrenchtest/test_simple_wrech.py > /tmp/wrench_test.log 2>&1; then
        log_success "WRENCH环境验证成功"
    else
        log_error "WRENCH环境验证失败"
        cat /tmp/wrench_test.log
        exit 1
    fi
}

# 验证知识库存在
verify_kb() {
    log_info "验证增强RAG知识库..."
    if [[ -f "data/enhanced_rag_kb.json" ]]; then
        # 检查知识库内容
        cases=$(python -c "import json; data=json.load(open('data/enhanced_rag_kb.json')); print(len(data['cases']))")
        log_info "找到增强RAG知识库，包含 $cases 个案例"
        
        if [[ $cases -lt 5000 ]]; then
            log_warning "知识库案例数量不足5000，可能需要重新生成"
        fi
    else
        log_error "未找到增强RAG知识库文件: data/enhanced_rag_kb.json"
        log_info "请先运行: python scripts/enhanced_rag_kb_generator.py"
        exit 1
    fi
}

# 训练性能预测器
train_predictor() {
    log_info "第1步: 训练性能预测器（使用增强知识库）..."
    if python scripts/train_predictor_from_kb.py configs/experiment.yaml; then
        log_success "性能预测器训练完成"
        # 检查模型性能
        if [[ -f "models/wass_models.pth" ]]; then
            r2=$(python -c "import torch; cp=torch.load('models/wass_models.pth', map_location='cpu', weights_only=False); print(f\"{cp['metadata']['performance_predictor']['validation_results']['r2']:.4f}\")")
            log_info "验证R²: $r2"
        fi
    else
        log_error "性能预测器训练失败"
        exit 1
    fi
}

# 训练DRL智能体
train_drl() {
    log_info "第2步: 训练DRL智能体（使用增强知识库）..."
    if python scripts/train_drl_wrench.py configs/experiment.yaml; then
        log_success "DRL智能体训练完成"
        # 检查训练结果
        if [[ -f "models/wass_optimized_models.pth" ]]; then
            makespan=$(python -c "import torch; cp=torch.load('models/wass_optimized_models.pth', map_location='cpu', weights_only=False); print(f\"{cp['drl_metadata']['avg_makespan']:.2f}\")")
            epsilon=$(python -c "import torch; cp=torch.load('models/wass_optimized_models.pth', map_location='cpu', weights_only=False); print(f\"{cp['drl_metadata']['final_epsilon']:.3f}\")")
            log_info "最终性能: ${makespan}s, ε: $epsilon"
        fi
    else
        log_error "DRL智能体训练失败"
        exit 1
    fi
}

# 训练RAG知识库（使用已有知识库）
train_rag() {
    log_info "第3步: 加载RAG知识库（使用已有的5000个案例）..."
    
    # 创建一个简单的脚本来加载并验证知识库
    python3 << 'EOF'
import sys
import os
sys.path.append('src')
from knowledge_base.wrench_full_kb import WRENCHRAGKnowledgeBase
import json

# 加载知识库
print("📚 加载增强RAG知识库...")
try:
    with open('data/enhanced_rag_kb.json', 'r') as f:
        data = json.load(f)
    
    # 创建知识库对象
    kb = WRENCHRAGKnowledgeBase(embedding_dim=64)
    
    # 加载案例
    from knowledge_base.wrench_full_kb import WRENCHKnowledgeCase
    import numpy as np
    
    for case_dict in data['cases']:
        # 转换列表为numpy数组
        case_dict['workflow_embedding'] = np.array(case_dict['workflow_embedding'])
        case_dict['task_features'] = np.array(case_dict['task_features'])
        case_dict['node_features'] = np.array(case_dict['node_features'])
        
        # 创建案例对象
        case = WRENCHKnowledgeCase(**case_dict)
        kb.add_case(case)
    
    print(f"✅ 成功加载知识库，包含 {len(kb.cases)} 个案例")
    
    # 统计调度器类型
    scheduler_types = {}
    for case in kb.cases:
        scheduler = case.scheduler_type
        scheduler_types[scheduler] = scheduler_types.get(scheduler, 0) + 1
    
    print("📊 调度器分布:")
    for scheduler, count in scheduler_types.items():
        print(f"   {scheduler}: {count} 个案例")
    
    # 测试检索功能
    if len(kb.cases) > 0:
        test_case = kb.cases[0]
        similar_cases = kb.retrieve_similar_cases(
            test_case.workflow_embedding, 
            test_case.task_features, 
            k=5
        )
        print(f"🔍 检索测试: 找到 {len(similar_cases)} 个相似案例")
    
    # 保存为标准格式，以便其他脚本使用
    import pickle
    with open('data/wrench_rag_knowledge_base.pkl', 'wb') as f:
        pickle.dump(kb, f)
    
    print("💾 知识库已保存为标准格式: data/wrench_rag_knowledge_base.pkl")
    
except Exception as e:
    print(f"❌ 加载知识库失败: {e}")
    sys.exit(1)
EOF

    if [[ $? -eq 0 ]]; then
        log_success "RAG知识库加载完成"
    else
        log_error "RAG知识库加载失败"
        exit 1
    fi
}

# 运行实验
run_experiments() {
    log_info "第4步: 运行基于WRENCH的真实实验对比..."
    if python experiments/wrench_real_experiment.py; then
        log_success "WRENCH实验运行完成"
        # 检查实验结果
        if [[ -f "results/wrench_experiments/detailed_results.json" ]]; then
            log_info "实验结果已保存到 results/wrench_experiments/"
        fi
    else
        log_error "WRENCH实验运行失败"
        exit 1
    fi
}

# 生成图表
generate_charts() {
    log_info "第5步: 生成学术论文图表..."
    if python charts/paper_charts.py; then
        log_success "图表生成完成"
        # 检查生成的图表
        chart_count=$(find charts/ -name "*.png" 2>/dev/null | wc -l)
        log_info "生成了 $chart_count 个图表文件"
    else
        log_error "图表生成失败"
        exit 1
    fi
}

# 显示结果摘要
show_summary() {
    log_info "=============== 实验完成摘要 ==============="
    
    echo -e "${GREEN}训练模型:${NC}"
    if [[ -f "models/wass_optimized_models.pth" ]]; then
        python -c "
import torch
cp = torch.load('models/wass_optimized_models.pth', map_location='cpu', weights_only=False)
print('  • DRL智能体: 最终性能 = {:.2f}s'.format(cp['drl_metadata']['avg_makespan']))
"
    fi
    if [[ -f "models/wass_models.pth" ]]; then
        python -c "
import torch
cp = torch.load('models/wass_models.pth', map_location='cpu', weights_only=False)
print('  • 性能预测器: R² = {:.4f}'.format(cp['metadata']['performance_predictor']['validation_results']['r2']))
"
    fi
    
    echo -e "${GREEN}知识库:${NC}"
    if [[ -f "data/enhanced_rag_kb.json" ]]; then
        cases=$(python -c "import json; data=json.load(open('data/enhanced_rag_kb.json')); print(len(data['cases']))")
        echo "  • 增强RAG知识库: $cases 个案例"
    fi
    if [[ -f "data/wrench_rag_knowledge_base.pkl" ]]; then
        echo "  • 标准格式知识库: data/wrench_rag_knowledge_base.pkl"
    fi
    
    echo -e "${GREEN}实验结果:${NC}"
    if [[ -f "results/final_experiments_discrete_event/experiment_results.json" ]]; then
        echo "  • 实验数据: results/final_experiments_discrete_event/"
    fi
    
    echo -e "${GREEN}生成图表:${NC}"
    chart_count=$(find charts/ -name "*.png" 2>/dev/null | wc -l)
    echo "  • 图表文件: $chart_count 个"
    
    log_success "使用增强RAG知识库的实验流程执行完成! 🎉"
}

# 主函数
main() {
    log_info "开始使用增强RAG知识库的实验流程..."
    log_info "预计用时: 20-40分钟"
    echo
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 执行各个步骤
    check_environment
    verify_wrench
    verify_kb
    train_predictor
    train_drl
    train_rag
    run_experiments
    generate_charts
    
    # 计算总用时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo
    log_info "总执行时间: ${minutes}分${seconds}秒"
    
    # 显示结果摘要
    show_summary
}

# 检查命令行参数
if [[ $# -gt 0 ]]; then
    case $1 in
        "check")
            check_environment
            verify_wrench
            verify_kb
            ;;
        "predictor")
            train_predictor
            ;;
        "drl")
            train_drl
            ;;
        "rag")
            train_rag
            ;;
        "experiments")
            run_experiments
            ;;
        "charts")
            generate_charts
            ;;
        "summary")
            show_summary
            ;;
        *)
            echo "用法: $0 [check|predictor|drl|rag|experiments|charts|summary]"
            echo "无参数运行完整流程"
            exit 1
            ;;
    esac
else
    main
fi