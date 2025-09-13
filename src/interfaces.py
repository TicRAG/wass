"""
Interfaces for the WASS system.
Used only in documentation/design phase to constrain subsequent implementations.
"""
import math
from typing import Dict, List, Protocol, Tuple, Iterable, Optional, Any, Union
from typing import NamedTuple
from abc import ABC, abstractmethod

DataSample = Dict[str, Any]

class PredictedValue(NamedTuple):
    """Represents a predicted value with confidence."""
    value: float
    confidence: float

# Type alias for scheduling decisions
SchedulingDecision = Dict[str, Any]  # Maps node names to tasks

class Scheduler(ABC):
    """Abstract base class for all schedulers."""
    
    @abstractmethod
    def schedule(self, ready_tasks: List[Any], simulation: Any) -> SchedulingDecision:
        """Schedule ready tasks on available nodes.
        
        Args:
            ready_tasks: List of tasks that are ready to be scheduled
            simulation: The simulation environment
            
        Returns:
            A dictionary mapping node names to tasks to be scheduled on them
        """
        pass


class DataReader(Protocol):
    def read(self) -> Iterable[DataSample]: ...
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
