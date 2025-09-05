"""组件工厂：根据配置字符串创建对象 (占位).

依赖配置字段 (参见 configs_example.yaml)。
"""
from __future__ import annotations
from typing import Dict, Any, List

# 数据
from .data.jsonl_adapter import JSONLAdapter
# Labeling
from .labeling.lf_base import build_lfs
from .labeling.label_matrix import SimpleLabelMatrixBuilder
# Label Model
from .label_model.majority_vote import MajorityVote
# Graph
from .graph.graph_builder import CooccurrenceGraphBuilder
from .graph.gnn_model import DummyGNN
# RAG
from .rag.retriever import SimpleBM25Retriever
from .rag.fusion import ConcatFusion
# DRL
from .drl.env import ActiveLearningEnv
from .drl.policy import RandomPolicy

# 可选 Wrench wrapper (延迟导入)
try:
    from .label_model.wrench_wrapper import WrenchLabelModelWrapper  # type: ignore
except Exception:  # noqa
    WrenchLabelModelWrapper = None  # type: ignore


def build_dataset(cfg: Dict[str, Any]):
    a = cfg.get('adapter')
    if a == 'simple_jsonl':
        return JSONLAdapter(
            data_dir=cfg.get('data_dir', 'data'),
            train_file=cfg.get('train_file', 'train.jsonl'),
            valid_file=cfg.get('valid_file', 'valid.jsonl'),
            test_file=cfg.get('test_file', 'test.jsonl'),
        )
    raise ValueError(f"Unknown dataset adapter: {a}")


def build_label_functions(cfg: Dict[str, Any]):
    lfs_cfg: List[Dict[str, Any]] = cfg.get('lfs', [])
    return build_lfs(lfs_cfg)


def build_label_matrix_builder(_: Dict[str, Any]):
    return SimpleLabelMatrixBuilder()


def build_label_model(cfg: Dict[str, Any]):
    t = cfg.get('type')
    if t == 'majority_vote':
        return MajorityVote()
    if t == 'wrench':
        if WrenchLabelModelWrapper is None:
            raise ImportError('Wrench not available in this environment.')
        model_name = cfg.get('model_name', 'MajorityVoting')
        return WrenchLabelModelWrapper(model_name=model_name, params=cfg.get('params', {}))
    raise ValueError(f"Unknown label model: {t}")


def build_graph_builder(cfg: Dict[str, Any]):
    t = cfg.get('builder')
    if t == 'cooccurrence':
        return CooccurrenceGraphBuilder(**cfg.get('params', {}))
    raise ValueError(f"Unknown graph builder: {t}")


def build_gnn(cfg: Dict[str, Any]):
    model_type = cfg.get('gnn_model')
    if model_type == 'gcn':  # 占位所有映射到 Dummy
        return DummyGNN(**cfg.get('gnn_params', {}))
    raise ValueError(f"Unknown gnn model: {model_type}")


def build_retriever(cfg: Dict[str, Any]):
    t = cfg.get('retriever')
    if t == 'simple_bm25':
        return SimpleBM25Retriever()
    raise ValueError(f"Unknown retriever: {t}")


def build_fusion(cfg: Dict[str, Any]):
    t = cfg.get('fusion')
    if t == 'concat':
        return ConcatFusion()
    raise ValueError(f"Unknown fusion: {t}")


def build_drl_env(cfg: Dict[str, Any], unlabeled_pool):
    env_name = cfg.get('env')
    if env_name in ['active_learning_env', 'active_learning']:
        return ActiveLearningEnv(unlabeled_pool)
    raise ValueError(f"Unknown env: {env_name}")


def build_drl_policy(cfg: Dict[str, Any]):
    policy_name = cfg.get('policy')
    if policy_name == 'dqn':  # 占位仍返回随机策略
        return RandomPolicy()
    if policy_name == 'random':
        return RandomPolicy()
    raise ValueError(f"Unknown policy: {policy_name}")
