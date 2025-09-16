#!/bin/bash
# ==============================================================================
#                 WASS-RAG 全流程训练与实验脚本
#
# 该脚本实现了 "学习者-导师" 思想下的三阶段训练流程，并最终进行性能评估。
# 流程:
# 1. 阶段一: 知识库播种 - 从启发式算法的运行结果中提取经验。
# 2. 阶段二: 性能预测器训练 (导师) - 训练一个能预测调度性能的导师模型。
# 3. 阶段三: DRL智能体训练 (学习者) - 在导师的指导下训练DRL决策模型。
# 4. 最终评估: 使用新训练的模型和知识库进行对比实验。
#
# ==============================================================================

set -e # 遇到错误立即退出

# --- 配置区 ---
# 定义所有关键文件路径，方便管理
KB_SEED_DATA="data/heuristic_only_real_cases.json"
MAIN_KB_JSON="data/real_heuristic_kb.json"
PREDICTOR_MODEL="models/performance_predictor.pth"
DRL_MODEL="models/improved_wass_drl.pth"
DRL_CONFIG="configs/drl.yaml"
PREDICTOR_CONFIG="configs/predictor.yaml" # 假设预测器有自己的配置文件
EXPERIMENT_CONFIG="configs/real_heuristic_experiment.yaml"
PLATFORM_FILE="test_platform.xml"
WORKFLOW_MANAGER="scripts/workflow_manager.py"

# --- 颜色和日志函数 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- 阶段函数定义 ---

# 阶段0: 准备工作流
stage_0_prepare_workflows() {
    log_info "--- [阶段0] 开始：准备工作流文件 ---"
    
    # 确保workflow_manager.py可执行
    chmod +x "${WORKFLOW_MANAGER}"
    
    # 生成工作流文件
    if python "${WORKFLOW_MANAGER}" --action generate; then
        log_success "工作流文件生成完成"
    else
        log_error "工作流文件生成失败"
        exit 1
    fi
    
    # 更新所有配置文件以确保训练和实验的一致性
    if python "${WORKFLOW_MANAGER}" --action update_all_configs; then
        log_success "所有配置文件更新完成"
    else
        log_error "配置文件更新失败"
        exit 1
    fi
    
    log_success "--- [阶段0] 完成 ---"
}

# 阶段一: 知识库播种
stage_1_seed_knowledge_base() {
    log_info "--- [阶段一] 开始：知识库播种 ---"
    
    log_info "步骤 1/2: 从历史运行中提取启发式算法案例..."
    if python scripts/extract_real_heuristic_cases.py; then
        log_success "案例提取完成，数据保存在: ${KB_SEED_DATA}"
    else
        log_error "案例提取失败"
        exit 1
    fi

    log_info "步骤 2/2: 将案例更新并构建到主知识库..."
    if python scripts/update_rag_with_real_cases.py; then
        log_success "主知识库构建完成: ${MAIN_KB_JSON}"
    else
        log_error "主知识库构建失败"
        exit 1
    fi
    log_success "--- [阶段一] 完成 ---"
}

# 阶段二: 训练性能预测器 (导师)
stage_2_train_predictor() {
    log_info "--- [阶段二] 开始：训练性能预测器 (导师) ---"
    
    # 确保阶段一的产出存在
    if [[ ! -f "$MAIN_KB_JSON" ]]; then
        log_error "找不到主知识库文件: ${MAIN_KB_JSON}，请先执行阶段一"
        exit 1
    fi

    # 确保预测器的配置文件存在
    if [[ ! -f "$PREDICTOR_CONFIG" ]]; then
        log_error "找不到性能预测器的配置文件: ${PREDICTOR_CONFIG}"
        log_error "请创建一个名为 predictor.yaml 的配置文件在 configs/ 目录下。"
        exit 1
    fi
    
    log_info "使用 ${MAIN_KB_JSON} 中的数据训练性能预测器..."
    
    # 修复：使用正确的命令行参数调用训练脚本
    # 它需要一个配置文件，而不是 --output-model 参数
    log_info "scripts/train_predictor_from_kb.py --kb-path ${MAIN_KB_JSON} ${PREDICTOR_CONFIG}"
    if python scripts/train_predictor_from_kb.py --kb-path "${MAIN_KB_JSON}" "${PREDICTOR_CONFIG}"; then
        log_success "性能预测器训练完成，模型已根据 ${PREDICTOR_CONFIG} 中的配置保存"
    else
        log_error "性能预测器训练失败"
        exit 1
    fi
    log_success "--- [阶段二] 完成 ---"
}
# 阶段三: 训练DRL智能体 (学习者)
stage_3_train_drl_agent() {
    log_info "--- [阶段三] 开始：训练DRL智能体 (学习者) ---"
    
    # 确保配置文件存在
    if [[ ! -f "$DRL_CONFIG" ]]; then
        log_error "找不到DRL配置文件: ${DRL_CONFIG}"
        exit 1
    fi

    log_info "确保DRL训练配置与实验环境一致..."
    # 临时修改配置文件，使其指向正确的知识库和平台
    # 创建备份
    cp "${DRL_CONFIG}" "${DRL_CONFIG}.bak"
    
    # 使用 sed 命令进行修改 (兼容macOS和Linux)
    sed -i.sedbak "s|platform_file:.*|platform_file: \"${PLATFORM_FILE}\"|" "${DRL_CONFIG}"
    # 假设DRL配置中有一个 knowledge_base -> path 的字段
    sed -i.sedbak "s|path:.*knowledge_base.json|path: \"${MAIN_KB_JSON}\"|" "${DRL_CONFIG}"
    rm -f "${DRL_CONFIG}.sedbak" # 清理sed产生的备份

    log_info "配置文件已临时更新，开始使用 improved_drl_trainer.py 进行训练..."
    log_info "scripts/improved_drl_trainer.py --config ${DRL_CONFIG}"
    if python scripts/improved_drl_trainer.py --config "${DRL_CONFIG}"; then
        log_success "DRL智能体训练完成，模型保存在: ${DRL_MODEL}"
    else
        log_error "DRL智能体训练失败"
        # 恢复原始配置文件
        mv "${DRL_CONFIG}.bak" "${DRL_CONFIG}"
        exit 1
    fi
    
    # 训练成功后，恢复原始配置文件
    mv "${DRL_CONFIG}.bak" "${DRL_CONFIG}"
    log_info "原始DRL配置文件已恢复"
    log_success "--- [阶段三] 完成 ---"
}

# 最终评估: 运行对比实验
final_stage_run_experiments() {
    log_info "--- [最终评估] 开始：运行对比实验 ---"

    # 确保训练好的DRL模型存在
    if [[ ! -f "$DRL_MODEL" ]]; then
        log_error "找不到训练好的DRL模型: ${DRL_MODEL}，请先执行阶段三"
        exit 1
    fi

    # 运行实验
    if python experiments/wrench_real_experiment.py; then
        log_success "对比实验运行完成"
    else
        log_error "对比实验运行失败"
        exit 1
    fi

    log_info "生成最终结果摘要..."
    # 调用一个独立的分析脚本，如果存在的话
    if [[ -f "analyze_real_results.py" ]]; then
        python analyze_real_results.py
    else
        log_warning "未找到结果分析脚本 analyze_real_results.py，跳过摘要生成。"
    fi
    
    log_success "--- [最终评估] 完成 ---"
}


# --- 主函数 ---
main() {
    log_info "启动 WASS-RAG 全流程训练与实验..."
    log_info "预计总用时: 30-60分钟"
    echo

    start_time=$(date +%s)
    
    # 依次执行所有阶段
    stage_0_prepare_workflows
    echo
    stage_1_seed_knowledge_base
    echo
    stage_2_train_predictor
    echo
    stage_3_train_drl_agent
    echo
    final_stage_run_experiments
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo
    log_success "🎉 WASS-RAG 全流程执行完毕! 总耗时: ${minutes}分${seconds}秒"
}

# --- 脚本入口 ---
# 允许单独执行某个阶段，方便调试
if [[ $# -gt 0 ]]; then
    case $1 in
        "stage0")
            stage_0_prepare_workflows
            ;;
        "stage1")
            stage_1_seed_knowledge_base
            ;;
        "stage2")
            stage_2_train_predictor
            ;;
        "stage3")
            stage_3_train_drl_agent
            ;;
        "eval")
            final_stage_run_experiments
            ;;
        *)
            echo "用法: $0 [stage0|stage1|stage2|stage3|eval]"
            echo "无参数则运行完整流程"
            exit 1
            ;;
    esac
else
    # 默认运行完整流程
    main
fi