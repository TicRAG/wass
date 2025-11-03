#!/usr/bin/env python3
"""Utility to programmatically synthesize WFCommons-compatible workflows.

Generates layered DAGs with controllable width/depth so baseline heuristics
(HEFT, MIN-MIN, DRL, RAG) exhibit divergent behaviour during simulation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Dict, List, Set


@dataclass
class GeneratorConfig:
    count: int
    output_dir: Path
    seed: int
    min_layers: int
    max_layers: int
    min_width: int
    max_width: int
    min_runtime: float
    max_runtime: float
    flop_rate: float
    memory_min: float
    memory_max: float
    fanout_min: int
    fanout_max: int
    file_size_min: float
    file_size_max: float
    critical_multiplier: float
    heavy_layer_multiplier: float
    decoy_branches: int
    decoy_runtime_scale: float
    decoy_file_multiplier: float
    decoy_file_size_mode: str
    file_size_cap: float | None


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate synthetic workflows with rich parallel structure.")
    parser.add_argument("--count", type=int, default=3, help="Number of workflows to generate.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/workflows/synthetic"), help="Destination directory for JSON outputs.")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed for reproducibility.")
    parser.add_argument("--min-layers", type=int, default=5, help="Minimum number of DAG layers per workflow.")
    parser.add_argument("--max-layers", type=int, default=7, help="Maximum number of DAG layers per workflow.")
    parser.add_argument("--min-width", type=int, default=3, help="Minimum tasks per intermediate layer.")
    parser.add_argument("--max-width", type=int, default=6, help="Maximum tasks per intermediate layer.")
    parser.add_argument("--min-runtime", type=float, default=2.5, help="Lower bound (seconds) for task runtimes.")
    parser.add_argument("--max-runtime", type=float, default=18.0, help="Upper bound (seconds) for task runtimes.")
    parser.add_argument("--flop-rate", type=float, default=8e8, help="Average flop/s used to convert runtime to FLOPs.")
    parser.add_argument("--memory-min", type=float, default=80e6, help="Minimum per-task memory footprint (bytes).")
    parser.add_argument("--memory-max", type=float, default=320e6, help="Maximum per-task memory footprint (bytes).")
    parser.add_argument("--fanout-min", type=int, default=1, help="Minimum number of parents drawn from previous layer.")
    parser.add_argument("--fanout-max", type=int, default=3, help="Maximum number of parents drawn from previous layer.")
    parser.add_argument("--file-size-min", type=float, default=64e6, help="Minimum file size in bytes for generated data artifacts.")
    parser.add_argument("--file-size-max", type=float, default=256e6, help="Maximum file size in bytes for generated data artifacts.")
    parser.add_argument("--critical-multiplier", type=float, default=4.0, help="Factor applied to the longest (critical) path to amplify runtime and communication cost.")
    parser.add_argument("--heavy-layer-multiplier", type=float, default=6.0, help="Scale factor applied to the middle layer to create bursty parallel heavy tasks.")
    parser.add_argument("--decoy-branches", type=int, default=2, help="Number of low-runtime decoy branches to attach to each critical-path task (set 0 to disable).")
    parser.add_argument("--decoy-runtime-scale", type=float, default=0.25, help="Scale factor applied to parent runtime/flops when creating decoy tasks.")
    parser.add_argument("--decoy-file-multiplier", type=float, default=5.0, help="Multiplier for decoy output file sizes to enforce heavy downstream transfers.")
    parser.add_argument("--decoy-file-size-mode", type=str, choices=["sum", "avg", "min"], default="sum", help="How to aggregate parent input file sizes for decoy base size before multiplier: sum | avg | min (default: sum).")
    parser.add_argument("--file-size-cap", type=float, default=None, help="Optional hard cap (bytes) applied to any generated file (including decoy outputs) after multipliers.")
    args = parser.parse_args()

    if args.min_layers < 2:
        parser.error("--min-layers must be at least 2 to create parallel structure.")
    if args.max_layers < args.min_layers:
        parser.error("--max-layers must be >= --min-layers.")
    if args.max_width < args.min_width:
        parser.error("--max-width must be >= --min-width.")
    if args.fanout_max < args.fanout_min or args.fanout_min < 1:
        parser.error("Invalid fanout bounds; ensure 1 <= fanout_min <= fanout_max.")
    if args.file_size_max < args.file_size_min:
        parser.error("--file-size-max must be >= --file-size-min.")
    if args.critical_multiplier < 1.0:
        parser.error("--critical-multiplier must be >= 1.0.")
    if args.heavy_layer_multiplier < 1.0:
        parser.error("--heavy-layer-multiplier must be >= 1.0.")
    if args.decoy_branches < 0:
        parser.error("--decoy-branches must be >= 0.")
    if args.decoy_runtime_scale <= 0.0:
        parser.error("--decoy-runtime-scale must be > 0.")
    if args.decoy_file_multiplier <= 0.0:
        parser.error("--decoy-file-multiplier must be > 0.")

    return GeneratorConfig(
        count=args.count,
        output_dir=args.output_dir,
        seed=args.seed,
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        min_width=args.min_width,
        max_width=args.max_width,
        min_runtime=args.min_runtime,
        max_runtime=args.max_runtime,
        flop_rate=args.flop_rate,
        memory_min=args.memory_min,
        memory_max=args.memory_max,
        fanout_min=args.fanout_min,
    fanout_max=args.fanout_max,
    file_size_min=args.file_size_min,
    file_size_max=args.file_size_max,
    critical_multiplier=args.critical_multiplier,
    heavy_layer_multiplier=args.heavy_layer_multiplier,
    decoy_branches=args.decoy_branches,
    decoy_runtime_scale=args.decoy_runtime_scale,
    decoy_file_multiplier=args.decoy_file_multiplier,
    decoy_file_size_mode=args.decoy_file_size_mode,
    file_size_cap=args.file_size_cap,
    )


def _random_runtime(rng: random.Random, cfg: GeneratorConfig, layer: int, total_layers: int) -> float:
    base = rng.uniform(cfg.min_runtime, cfg.max_runtime)
    if layer == 0:
        return max(cfg.min_runtime * 0.5, base * 0.6)
    if layer == total_layers - 1:
        return base * 1.4
    return base


def _random_memory(rng: random.Random, cfg: GeneratorConfig) -> float:
    return rng.uniform(cfg.memory_min, cfg.memory_max)


def _make_task_id(layer: int, index: int) -> str:
    return f"L{layer:02d}_T{index:03d}"


def _make_file_id(task_id: str, suffix: str) -> str:
    return f"{task_id}_{suffix}.dat"


def _create_task(
    rng: random.Random,
    cfg: GeneratorConfig,
    layer: int,
    layer_index: int,
    total_layers: int,
    parents: List[str],
) -> Dict[str, object]:
    task_id = _make_task_id(layer, layer_index)
    runtime = _random_runtime(rng, cfg, layer, total_layers)
    flops = runtime * cfg.flop_rate * rng.uniform(0.85, 1.25)
    memory = _random_memory(rng, cfg)
    output_file = _make_file_id(task_id, "out")
    if parents:
        input_files = [_make_file_id(parent, "out") for parent in parents]
    else:
        input_files = [_make_file_id(task_id, "input")]
    spec_task = {
        "name": task_id,
        "id": task_id,
        "children": [],
        "inputFiles": input_files,
        "outputFiles": [output_file],
        "parents": parents,
        "flops": float(flops),
        "runtime": float(runtime),
        "memory": float(memory),
    }
    exec_task = {
        "id": task_id,
        "runtimeInSeconds": float(runtime),
        "cores": 1,
        "avgCPU": rng.uniform(0.75, 0.95),
    }
    file_entries = {file_id: rng.uniform(cfg.file_size_min, cfg.file_size_max) for file_id in input_files + [output_file]}
    return spec_task, exec_task, file_entries


def _longest_path_by_runtime(tasks_by_layer: List[List[str]], spec_tasks: Dict[str, Dict[str, object]]) -> List[str]:
    if not tasks_by_layer:
        return []
    longest_cost: Dict[str, float] = {}
    successor: Dict[str, str | None] = {}

    for layer_ids in reversed(tasks_by_layer):
        for task_id in layer_ids:
            task_spec = spec_tasks[task_id]
            runtime = float(task_spec.get("runtime", 0.0))
            children = task_spec.get("children", []) or []
            if not children:
                longest_cost[task_id] = runtime
                successor[task_id] = None
            else:
                best_child = None
                best_cost = -1.0
                for child in children:
                    child_cost = longest_cost.get(child, 0.0)
                    if child_cost > best_cost:
                        best_cost = child_cost
                        best_child = child
                longest_cost[task_id] = runtime + max(best_cost, 0.0)
                successor[task_id] = best_child

    start_candidates = tasks_by_layer[0]
    if not start_candidates:
        return []
    best_start = max(start_candidates, key=lambda tid: longest_cost.get(tid, 0.0))
    path: List[str] = []
    seen: Set[str] = set()
    current = best_start
    while current is not None and current not in seen:
        path.append(current)
        seen.add(current)
        current = successor.get(current)
    return path


def _inject_decoy_branches(
    cfg: GeneratorConfig,
    tasks_by_layer: List[List[str]],
    spec_tasks: Dict[str, Dict[str, object]],
    exec_tasks: Dict[str, Dict[str, object]],
    file_sizes: Dict[str, float],
    critical_path: List[str],
) -> None:
    if cfg.decoy_branches <= 0 or not critical_path:
        return
    if len(tasks_by_layer) < 2:
        return

    sink_layer = tasks_by_layer[-1]
    if not sink_layer:
        return

    task_to_layer = {}
    for layer_idx, layer_ids in enumerate(tasks_by_layer):
        for tid in layer_ids:
            task_to_layer[tid] = layer_idx

    sink_count = len(sink_layer)
    sink_rotation = 0
    for task_id in critical_path[:-1]:
        parent_layer_idx = task_to_layer.get(task_id)
        if parent_layer_idx is None:
            continue
        target_layer_idx = min(parent_layer_idx + 1, len(tasks_by_layer) - 1)
        parent_spec = spec_tasks.get(task_id)
        if not parent_spec:
            continue
        parent_outputs = parent_spec.get("outputFiles", [])
        if not parent_outputs:
            continue

        for branch_idx in range(cfg.decoy_branches):
            sink_id = sink_layer[sink_rotation % sink_count]
            sink_rotation += 1
            decoy_suffix = f"DECOY_{sink_rotation:03d}"
            decoy_id = f"{task_id}_{decoy_suffix}"
            input_files = list(parent_outputs)
            output_file = _make_file_id(decoy_id, "out")

            base_runtime = float(parent_spec.get("runtime", cfg.min_runtime)) * cfg.decoy_runtime_scale
            runtime = max(cfg.min_runtime * 0.25, base_runtime)
            base_flops = float(parent_spec.get("flops", cfg.flop_rate)) * cfg.decoy_runtime_scale
            flops = max(cfg.flop_rate * 0.1, base_flops)
            memory = float(parent_spec.get("memory", cfg.memory_min))

            parent_spec.setdefault("children", [])
            if decoy_id not in parent_spec["children"]:
                parent_spec["children"].append(decoy_id)

            sink_spec = spec_tasks.get(sink_id)
            if sink_spec is None:
                continue
            sink_spec.setdefault("parents", [])
            if decoy_id not in sink_spec["parents"]:
                sink_spec["parents"].append(decoy_id)
            sink_inputs = sink_spec.setdefault("inputFiles", [])
            if output_file not in sink_inputs:
                sink_inputs.append(output_file)

            decoy_spec = {
                "name": decoy_id,
                "id": decoy_id,
                "children": [sink_id],
                "inputFiles": input_files,
                "outputFiles": [output_file],
                "parents": [task_id],
                "flops": float(flops),
                "runtime": float(runtime),
                "memory": float(memory),
            }
            decoy_exec = {
                "id": decoy_id,
                "runtimeInSeconds": float(runtime),
                "cores": 1,
                "avgCPU": 0.5,
            }

            parent_sizes = [float(file_sizes.get(fid, cfg.file_size_min)) for fid in input_files]
            if not parent_sizes:
                parent_sizes = [cfg.file_size_min]
            if cfg.decoy_file_size_mode == 'avg':
                base_output_size = sum(parent_sizes) / len(parent_sizes)
            elif cfg.decoy_file_size_mode == 'min':
                base_output_size = min(parent_sizes)
            else:  # 'sum'
                base_output_size = sum(parent_sizes)
            scaled_size = float(base_output_size * cfg.decoy_file_multiplier)
            if cfg.file_size_cap is not None and scaled_size > cfg.file_size_cap:
                scaled_size = cfg.file_size_cap
            file_sizes[output_file] = scaled_size

            tasks_by_layer[target_layer_idx].append(decoy_id)
            spec_tasks[decoy_id] = decoy_spec
            exec_tasks[decoy_id] = decoy_exec


def build_workflow(idx: int, cfg: GeneratorConfig) -> Dict[str, object]:
    rng = random.Random(cfg.seed + idx * 17)
    layers = rng.randint(cfg.min_layers, cfg.max_layers)
    tasks_by_layer: List[List[str]] = []
    spec_tasks: Dict[str, Dict[str, object]] = {}
    exec_tasks: Dict[str, Dict[str, object]] = {}
    file_sizes: Dict[str, float] = {}

    for layer in range(layers):
        if layer == 0 or layer == layers - 1:
            width = max(2, cfg.min_width)
        else:
            width = rng.randint(cfg.min_width, cfg.max_width)
        current_layer_ids: List[str] = []
        for pos in range(width):
            if layer == 0:
                parents: List[str] = []
            else:
                prev_layer_ids = tasks_by_layer[layer - 1]
                fanout = rng.randint(cfg.fanout_min, min(cfg.fanout_max, len(prev_layer_ids)))
                parents = rng.sample(prev_layer_ids, k=fanout)
            spec_task, exec_task, file_entries = _create_task(rng, cfg, layer, pos, layers, parents)
            task_id = spec_task["id"]
            current_layer_ids.append(task_id)
            spec_tasks[task_id] = spec_task
            exec_tasks[task_id] = exec_task
            for fid, size in file_entries.items():
                file_sizes.setdefault(fid, size)
            for parent in parents:
                spec_tasks[parent]["children"].append(task_id)
        tasks_by_layer.append(current_layer_ids)

    critical_path = _longest_path_by_runtime(tasks_by_layer, spec_tasks)

    if cfg.critical_multiplier > 1.0 and critical_path:
        for task_id in critical_path:
            spec = spec_tasks[task_id]
            exec_spec = exec_tasks[task_id]
            spec['runtime'] = float(spec['runtime']) * cfg.critical_multiplier
            spec['flops'] = float(spec['flops']) * cfg.critical_multiplier
            exec_spec['runtimeInSeconds'] = float(exec_spec['runtimeInSeconds']) * cfg.critical_multiplier
            for outfile in spec.get('outputFiles', []):
                if outfile in file_sizes:
                    file_sizes[outfile] = float(file_sizes[outfile]) * cfg.critical_multiplier

    if cfg.heavy_layer_multiplier > 1.0 and tasks_by_layer:
        heavy_layer_idx = len(tasks_by_layer) // 2
        heavy_layer_tasks = tasks_by_layer[heavy_layer_idx]
        for task_id in heavy_layer_tasks:
            spec = spec_tasks[task_id]
            exec_spec = exec_tasks[task_id]
            spec['runtime'] = float(spec['runtime']) * cfg.heavy_layer_multiplier
            spec['flops'] = float(spec['flops']) * cfg.heavy_layer_multiplier
            exec_spec['runtimeInSeconds'] = float(exec_spec['runtimeInSeconds']) * cfg.heavy_layer_multiplier
            for outfile in spec.get('outputFiles', []):
                if outfile in file_sizes:
                    file_sizes[outfile] = float(file_sizes[outfile]) * cfg.heavy_layer_multiplier

    if cfg.decoy_branches > 0:
        _inject_decoy_branches(cfg, tasks_by_layer, spec_tasks, exec_tasks, file_sizes, critical_path)

    name = f"synthetic_workflow_{idx:03d}"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    files_section = [
        {
            "id": fid,
            "name": fid,
            "sizeInBytes": float(size)
        }
        for fid, size in file_sizes.items()
    ]

    workflow_payload = {
        "name": name,
        "description": "Synthetic workflow with controllable parallelism (generated programmatically).",
        "createdAt": created_at,
        "schemaVersion": "1.5",
        "author": {
            "name": "synthetic-generator",
            "email": "support@example.com"
        },
        "workflow_name": name,
        "workflow": {
            "specification": {
                "tasks": list(spec_tasks.values()),
                "files": files_section,
            },
            "execution": {
                "tasks": list(exec_tasks.values()),
            },
        },
    }
    return workflow_payload


def save_workflow(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    for idx in range(cfg.count):
        workflow = build_workflow(idx, cfg)
        filename = f"{workflow['name']}.json"
        out_path = cfg.output_dir / filename
        save_workflow(workflow, out_path)
        generated_paths.append(out_path)
        print(f"✅ Generated {out_path}")

    print(f"Done. {len(generated_paths)} workflow(s) saved under {cfg.output_dir}.")
    print("Reminder: copy or reference them from configs/workflow_config.yaml if you want to include them in experiments.")


if __name__ == "__main__":
    main()
