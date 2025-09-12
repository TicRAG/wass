"""Shared workflow generation utilities to unify training and evaluation environments.

The goal is to remove distribution shift between train and eval:
- Same flops range
- Same file size range
- Same dependency probability
- Optional fixed workflow sizes list
"""
from __future__ import annotations
import random
from typing import Sequence, Tuple, Dict, Any, List, Optional

try:
    import wrench  # type: ignore
except Exception:  # pragma: no cover
    wrench = None  # type: ignore

DEFAULT_FLOPS_RANGE = (1e9, 12e9)
DEFAULT_FILE_SIZE_RANGE = (5_000, 50_000)
DEFAULT_DEP_PROB = 0.35

class WorkflowSpec:
    def __init__(self, size: int, flops_range: Tuple[float,float], dep_prob: float, file_size_range: Tuple[int,int]):
        self.size = size
        self.flops_range = flops_range
        self.dep_prob = dep_prob
        self.file_size_range = file_size_range

    def as_dict(self) -> Dict[str, Any]:
        return {
            'size': self.size,
            'flops_range': self.flops_range,
            'dep_prob': self.dep_prob,
            'file_size_range': self.file_size_range
        }

def generate_workflow(sim, size: int, flops_range=DEFAULT_FLOPS_RANGE, dep_prob: float=DEFAULT_DEP_PROB, file_size_range=DEFAULT_FILE_SIZE_RANGE):
    """Generate a workflow inside an existing simulation with unified parameters.
    Returns (workflow, tasks, files)"""
    if wrench is None:
        raise RuntimeError('WRENCH not available')
    workflow = sim.create_workflow()
    tasks = []
    files = []
    for i in range(size):
        flops = random.uniform(*flops_range)
        task = workflow.add_task(f"task_{i}", flops, 1, 1, 0)
        tasks.append(task)
        if i < size - 1:
            fsize = random.randint(*file_size_range)
            ofile = sim.add_file(f"f_{i}", fsize)
            task.add_output_file(ofile)
            files.append(ofile)
    for i in range(1, len(tasks)):
        if random.random() < dep_prob:
            dep_idx = random.randint(0, i - 1)
            if dep_idx < len(files):
                tasks[i].add_input_file(files[dep_idx])
    return workflow, tasks, files

def generate_richer_workflow(
    sim,
    size: int,
    flops_range=DEFAULT_FLOPS_RANGE,
    file_size_range=DEFAULT_FILE_SIZE_RANGE,
    layers: Optional[int]=None,
    layer_width_range: Tuple[int,int]=(3,10),
    max_parents: int = 3,
    parent_edge_prob: float = 0.45,
    merge_prob: float = 0.25,
    seed: Optional[int] = None,
):
    """Generate a more structurally diverse DAG with optional multi-parent tasks.
    Returns (workflow, tasks, files, adjacency) where adjacency={'children': {task_name:[child_names]}, 'parents':{...}}

    Strategy:
      - Build layer by layer; each layer size sampled within layer_width_range, truncated to remaining tasks.
      - For each task in new layer choose up to max_parents from previous K layers (currently only immediate previous) with probability parent_edge_prob.
      - Ensure at least one parent for non-first-layer tasks (except if none exist yet) to avoid isolated chains.
      - With merge_prob create convergence by reusing existing parents for multiple siblings.
    """
    if wrench is None:
        raise RuntimeError('WRENCH not available')
    rnd = random.Random(seed)
    workflow = sim.create_workflow()
    tasks: List[Any] = []
    files: List[Any] = []
    # Layer construction
    remaining = size
    layer_defs: List[List[Any]] = []
    while remaining > 0:
        if layers is not None and len(layer_defs) >= layers:
            # force all remaining into last layer
            w = remaining
        else:
            low, high = layer_width_range
            w = min(remaining, rnd.randint(low, high))
        layer: List[Any] = []
        for i in range(w):
            idx = len(tasks)
            flops = rnd.uniform(*flops_range)
            t = workflow.add_task(f"task_{idx}", flops, 1, 1, 0)
            tasks.append(t); layer.append(t)
            if idx < size - 1:  # output file for potential dependency
                fsize = rnd.randint(*file_size_range)
                of = sim.add_file(f"f_{idx}", fsize)
                t.add_output_file(of)
                files.append(of)
        layer_defs.append(layer)
        remaining -= w
    # Build adjacency
    parents: Dict[str,List[str]] = {t.get_name(): [] for t in tasks}
    children: Dict[str,List[str]] = {t.get_name(): [] for t in tasks}
    # Connect layers sequentially (previous layer candidates)
    for li in range(1, len(layer_defs)):
        prev = layer_defs[li-1]
        cur = layer_defs[li]
        for t in cur:
            # choose parents
            cand = prev[:]  # could extend to earlier layers later
            rnd.shuffle(cand)
            chosen: List[Any] = []
            for p in cand:
                if len(chosen) >= max_parents:
                    break
                if rnd.random() < parent_edge_prob:
                    chosen.append(p)
            if not chosen:
                # ensure at least one parent (pick fastest random) when not first non-empty prev
                chosen = [rnd.choice(prev)] if prev else []
            # Optionally add merge: reuse one parent's outputs for additional siblings (already implicit)
            for p in chosen:
                # find one of p's output files (if any) to link
                outputs = p.get_output_files()
                if outputs:
                    # pick first (could randomize)
                    t.add_input_file(outputs[0])
                parents[t.get_name()].append(p.get_name())
                children[p.get_name()].append(t.get_name())
    adjacency = {'parents': parents, 'children': children}
    return workflow, tasks, files, adjacency

__all__ = ['WorkflowSpec','generate_workflow','generate_richer_workflow','DEFAULT_FLOPS_RANGE','DEFAULT_FILE_SIZE_RANGE','DEFAULT_DEP_PROB']
