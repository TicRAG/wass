"""改进的Pipeline运行脚本 - 带完整日志和统计."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any
import json

from .factory import (
    build_dataset, build_label_functions, build_label_matrix_builder,
    build_label_model, build_graph_builder, build_gnn, build_retriever,
    build_fusion, build_drl_env, build_drl_policy
)
from .labeling.label_matrix import SimpleLabelMatrixBuilder
from .eval.metrics import accuracy, f1_binary
from .config_loader import load_experiment_config
from .utils import setup_logger, time_stage


def run_enhanced_pipeline(config_path: str):
    """运行增强版pipeline，包含详细日志和统计."""
    path = Path(config_path)
    if path.name == 'experiment.yaml' and path.parent.name == 'configs':
        cfg = load_experiment_config(str(path.parent))
    else:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))

    # 设置日志
    results_dir = Path(cfg['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / 'pipeline.log'
    logger = setup_logger('pipeline', str(log_file))
    
    logger.info(f"开始运行增强版pipeline, 配置文件: {config_path}")
    
    results: Dict[str, Any] = {
        "config_path": config_path,
        "experiment_name": cfg.get('experiment_name', 'unnamed'),
        "stages": {}
    }

    # 1. 数据加载
    with time_stage("data_loading", logger) as stage_info:
        ds_cfg = {
            'adapter': cfg['data']['adapter'],
            'data_dir': cfg['paths']['data_dir'],
            'train_file': cfg['data']['train_file'],
            'valid_file': cfg['data']['valid_file'],
            'test_file': cfg['data']['test_file'],
        }
        dataset = build_dataset(ds_cfg)
        train_data = dataset.train_split()
        valid_data = dataset.valid_split()
        test_data = dataset.test_split()
        
        data_stats = {
            'train_size': len(train_data),
            'valid_size': len(valid_data),
            'test_size': len(test_data)
        }
        results['data_stats'] = data_stats
        results['stages']['data_loading'] = stage_info
        logger.info(f"数据加载完成: {data_stats}")

    # 2. Label Functions & Label Matrix
    with time_stage("labeling", logger) as stage_info:
        lfs = build_label_functions(cfg['labeling'])
        L_builder: SimpleLabelMatrixBuilder = build_label_matrix_builder({})
        L = L_builder.build(train_data, lfs)
        label_stats = L_builder.stats(L)
        
        results['labeling_stats'] = label_stats
        results['stages']['labeling'] = stage_info
        logger.info(f"标签矩阵构建完成: {label_stats}")

    # 3. Label Model
    with time_stage("label_model", logger) as stage_info:
        label_model = build_label_model(cfg['label_model'])
        label_model.fit(L)
        proba = label_model.predict_proba(L)
        
        label_model_stats = {
            'proba_shape': getattr(proba, 'shape', None),
            'model_type': cfg['label_model']['type']
        }
        results['label_model_stats'] = label_model_stats
        results['stages']['label_model'] = stage_info
        logger.info(f"标签模型训练完成: {label_model_stats}")

    # 4. Graph + GNN
    with time_stage("graph_gnn", logger) as stage_info:
        graph = build_graph_builder(cfg['graph']).build(train_data, proba)
        gnn = build_gnn(cfg['graph'])
        gnn.train(graph, proba)
        gnn_preds = gnn.predict(graph)
        
        graph_stats = {
            'graph_nodes': len(gnn_preds),
            'gnn_model': cfg['graph']['gnn_model']
        }
        results['graph_stats'] = graph_stats
        results['stages']['graph_gnn'] = stage_info
        logger.info(f"图构建和GNN训练完成: {graph_stats}")

    # 5. RAG (可选)
    with time_stage("rag", logger) as stage_info:
        retriever = build_retriever(cfg['rag'])
        retriever.index(train_data)
        sample_query = train_data[0]['text'] if train_data else ''
        retrieved = retriever.retrieve(sample_query, top_k=cfg['rag'].get('top_k', 5)) if sample_query else []
        fusion = build_fusion(cfg['rag'])
        fused_example = fusion.fuse(train_data[0], retrieved) if train_data else {}
        
        rag_stats = {
            'retrieved_count': len(retrieved),
            'has_context': 'retrieved_context' in fused_example,
            'retriever_type': cfg['rag']['retriever']
        }
        results['rag_stats'] = rag_stats
        results['stages']['rag'] = stage_info
        logger.info(f"RAG处理完成: {rag_stats}")

    # 6. DRL (简单回合)
    with time_stage("drl", logger) as stage_info:
        drl_env = build_drl_env(cfg['drl'], unlabeled_pool=train_data)
        drl_policy = build_drl_policy(cfg['drl'])
        state = drl_env.reset()
        total_reward = 0.0
        episodes = cfg['drl'].get('episodes', 3)
        for episode in range(episodes):
            action = drl_policy.act(state)
            state, reward, done, _ = drl_env.step(action)
            total_reward += reward
            if done:
                break
        
        drl_stats = {
            'total_reward': total_reward,
            'episodes': episodes,
            'policy_type': cfg['drl']['policy']
        }
        results['drl_stats'] = drl_stats
        results['stages']['drl'] = stage_info
        logger.info(f"DRL训练完成: {drl_stats}")

    # 7. Eval (占位: 使用伪 gold = argmax(proba))
    with time_stage("evaluation", logger) as stage_info:
        if len(proba):
            import numpy as np
            preds = proba.argmax(axis=1).tolist()
            gold = preds  # 占位 (真实应来自数据)
            
            eval_stats = {
                'accuracy': accuracy(preds, gold),
                'f1': f1_binary(preds, gold),
                'n_predictions': len(preds)
            }
            results['eval_stats'] = eval_stats
            results['stages']['evaluation'] = stage_info
            logger.info(f"评估完成: {eval_stats}")

    # 8. 保存结果
    with time_stage("save_results", logger) as stage_info:
        summary_path = results_dir / 'summary.json'
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        
        # 保存配置副本
        config_copy_path = results_dir / 'config_used.yaml'
        config_copy_path.write_text(yaml.dump(cfg, allow_unicode=True, default_flow_style=False), encoding='utf-8')
        
        results['stages']['save_results'] = stage_info
        logger.info(f"结果保存完成: {summary_path}")

    logger.info("Pipeline运行完成")
    print(f"\n=== Enhanced Pipeline Summary ===")
    print(f"实验名称: {results['experiment_name']}")
    print(f"配置文件: {config_path}")
    print(f"结果目录: {results_dir}")
    
    # 打印关键统计
    if 'data_stats' in results:
        print(f"数据: 训练集{results['data_stats']['train_size']}样本")
    if 'labeling_stats' in results:
        stats = results['labeling_stats']
        print(f"标签: 覆盖率{stats['coverage']:.3f}, 冲突率{stats['conflict_rate']:.3f}")
    if 'eval_stats' in results:
        stats = results['eval_stats']
        print(f"评估: 准确率{stats['accuracy']:.3f}, F1{stats['f1']:.3f}")

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs_example.yaml'
    run_enhanced_pipeline(config_path)
