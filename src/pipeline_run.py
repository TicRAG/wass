"""统一 Pipeline 运行脚本 (占位执行).
读取单一 YAML (如 configs_example.yaml) 或 multi-file experiment.yaml 并按阶段执行, 输出统计.
"""
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


def run_pipeline(config_path: str):
    path = Path(config_path)
    if path.name == 'experiment.yaml' and path.parent.name == 'configs':
        cfg = load_experiment_config(str(path.parent))
    else:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))

    results: Dict[str, Any] = {"config_path": config_path}

    # 1. 数据
    ds_cfg = {
        'adapter': cfg['data']['adapter'],
        'data_dir': cfg['paths']['data_dir'],
        'train_file': cfg['data']['train_file'],
        'valid_file': cfg['data']['valid_file'],
        'test_file': cfg['data']['test_file'],
    }
    dataset = build_dataset(ds_cfg)
    train_data = dataset.train_split()
    results['train_size'] = len(train_data)

    # 2. Label Functions & Label Matrix
    lfs = build_label_functions(cfg['labeling'])
    L_builder: SimpleLabelMatrixBuilder = build_label_matrix_builder({})
    L = L_builder.build(train_data, lfs)
    stats = L_builder.stats(L)
    results['label_matrix_stats'] = stats

    # 3. Label Model
    label_model = build_label_model(cfg['label_model'])
    label_model.fit(L)
    proba = label_model.predict_proba(L)
    results['label_proba_shape'] = getattr(proba, 'shape', None)

    # 4. Graph + GNN
    graph = build_graph_builder(cfg['graph']).build(train_data, proba)
    gnn = build_gnn(cfg['graph'])
    gnn.train(graph, proba)
    gnn_preds = gnn.predict(graph)
    results['graph_nodes'] = len(gnn_preds)

    # 5. RAG (可选)
    retriever = build_retriever(cfg['rag'])
    retriever.index(train_data)
    sample_query = train_data[0]['text'] if train_data else ''
    retrieved = retriever.retrieve(sample_query, top_k=cfg['rag'].get('top_k', 5)) if sample_query else []
    fusion = build_fusion(cfg['rag'])
    fused_example = fusion.fuse(train_data[0], retrieved) if train_data else {}
    results['rag_retrieved'] = len(retrieved)
    results['rag_fused_has_context'] = 'retrieved_context' in fused_example

    # 6. DRL (简单回合)
    drl_env = build_drl_env(cfg['drl'], unlabeled_pool=train_data)
    drl_policy = build_drl_policy(cfg['drl'])
    state = drl_env.reset()
    total_reward = 0.0
    for _ in range(3):
        action = drl_policy.act(state)
        state, reward, done, _ = drl_env.step(action)
        total_reward += reward
        if done:
            break
    results['drl_total_reward'] = total_reward

    # 7. Eval (占位: 使用伪 gold = argmax(proba))
    if len(proba):
        import numpy as np
        preds = proba.argmax(axis=1).tolist()
        gold = preds  # 占位 (真实应来自数据)
        results['eval_accuracy'] = accuracy(preds, gold)
        results['eval_f1'] = f1_binary(preds, gold)

    # 8. 保存结果
    out_dir = Path(cfg['paths']['results_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    print("Pipeline summary ->", json.dumps(results, ensure_ascii=False, indent=2))

    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # 默认尝试多文件配置
        config_path = 'configs/experiment.yaml'
    run_pipeline(config_path)
