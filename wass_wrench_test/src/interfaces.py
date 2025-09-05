"""核心接口占位.
在仅文档/设计阶段，用于约束后续实现。
"""
from __future__ import annotations
from typing import Protocol, List, Dict, Any, Iterable, Tuple, Optional

class DataSample(dict):
    """简单表示单条数据，后续可替换为pydantic/dataclass"""
    pass

class DatasetAdapter(Protocol):
    def load(self) -> Iterable[DataSample]: ...
    def train_split(self) -> List[DataSample]: ...
    def valid_split(self) -> List[DataSample]: ...
    def test_split(self) -> List[DataSample]: ...

class LabelFunction(Protocol):
    name: str
    abstain: int
    def __call__(self, x: DataSample) -> int: ...

class LabelMatrixBuilder(Protocol):
    def build(self, data: List[DataSample], lfs: List[LabelFunction]) -> Any: ...  # returns L

class LabelModel(Protocol):
    def fit(self, L, **kwargs) -> None: ...
    def predict_proba(self, L) -> Any: ...  # numpy array

class GraphBuilder(Protocol):
    def build(self, data: List[DataSample], labels) -> Any: ...  # returns graph object

class GNNModel(Protocol):
    def train(self, graph, labels, **kwargs) -> Any: ...
    def predict(self, graph) -> Any: ...

class Retriever(Protocol):
    def index(self, data: List[DataSample]) -> None: ...
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]: ...

class RAGFusion(Protocol):
    def fuse(self, sample: DataSample, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]: ...

class DRLEnvironment(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]: ...

class DRLPolicy(Protocol):
    def act(self, state: Any) -> Any: ...
    def update(self, *batch) -> None: ...

class ExperimentLogger(Protocol):
    def log(self, **kwargs) -> None: ...
    def finalize(self) -> None: ...

class PipelineStage(Protocol):
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]: ...

class Pipeline:
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(self, initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = initial or {}
        for s in self.stages:
            ctx = s.run(ctx)
        return ctx
