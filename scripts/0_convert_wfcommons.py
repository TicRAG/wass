#!/usr/bin/env python3
"""Batch convert wfcommons workflow JSON files to project format with flops and memory fields.

Usage:
    python scripts/0_convert_wfcommons.py --input_dir configs/wfcommons --output_dir data/workflows

For each .json in input_dir, this script will:
  * Load wfcommons workflow JSON
  * Build helper dictionaries from execution + specification sections
  * Add `flops` and `memory` to each task in `workflow.specification.tasks`
  * Save full modified JSON to output_dir preserving filename

Memory estimation heuristic:
  memory = sum(input file sizes) + sum(output file sizes) + base_overhead (100MB)
FLOPS estimation heuristic:
  flops = runtimeInSeconds * (avgCPU/100) * (cpu_speed_mhz * 1e6)
Assumes task ran on first machine listed in execution.tasks[machines].
"""
from __future__ import annotations
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

BASE_OVERHEAD = 1e8  # 100MB in bytes


def build_machine_info(data: Dict[str, Any]) -> Dict[str, Any]:
    machines = data.get("workflow", {}).get("execution", {}).get("machines", [])
    machine_info = {}
    for m in machines:
        node_name = m.get("nodeName")
        if node_name:
            machine_info[node_name] = m
    return machine_info


def build_exec_tasks_info(data: Dict[str, Any]) -> Dict[str, Any]:
    exec_tasks = data.get("workflow", {}).get("execution", {}).get("tasks", [])
    exec_tasks_info = {}
    for t in exec_tasks:
        t_id = t.get("id")
        if t_id is None:
            continue
        exec_tasks_info[t_id] = {
            "runtimeInSeconds": t.get("runtimeInSeconds", 0.0),
            "avgCPU": t.get("avgCPU", 0.0),
            "machines": t.get("machines", []),
        }
    return exec_tasks_info


def build_file_sizes(data: Dict[str, Any]) -> Dict[str, int]:
    spec_files = data.get("workflow", {}).get("specification", {}).get("files", [])
    file_sizes = {}
    for f in spec_files:
        f_id = f.get("id")
        if f_id is None:
            continue
        file_sizes[f_id] = f.get("sizeInBytes", 0)
    return file_sizes


def compute_flops(exec_info: Dict[str, Any], machine_info: Dict[str, Any], task_id: Any, filename: str) -> float:
    if not exec_info:
        return 0.0
    runtime = exec_info.get("runtimeInSeconds", 0.0)
    cpu_util = exec_info.get("avgCPU", 0.0) / 100.0
    machines = exec_info.get("machines", [])
    if not machines:
        print(f"[WARN] Task {task_id} in {filename} has no machines listed; flops=0")
        return 0.0
    machine_name = machines[0]
    machine = machine_info.get(machine_name)
    if not machine:
        print(f"[WARN] Machine {machine_name} not found for task {task_id} in {filename}; flops=0")
        return 0.0
    speed_mhz = machine.get("cpu", {}).get("speedInMHz", 0.0)
    estimated_flops = runtime * cpu_util * (speed_mhz * 1e6)
    return estimated_flops


def compute_memory(task: Dict[str, Any], file_sizes: Dict[str, int]) -> float:
    input_size = sum(file_sizes.get(f_id, 0) for f_id in task.get("inputFiles", []))
    output_size = sum(file_sizes.get(f_id, 0) for f_id in task.get("outputFiles", []))
    memory = input_size + output_size + BASE_OVERHEAD
    return float(memory)


def process_file(path: Path, output_dir: Path) -> None:
    filename = path.name
    with path.open('r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse {filename}: {e}")
            return

    machine_info = build_machine_info(data)
    exec_tasks_info = build_exec_tasks_info(data)
    file_sizes = build_file_sizes(data)

    tasks: List[Dict[str, Any]] = data.get("workflow", {}).get("specification", {}).get("tasks", [])

    for task in tasks:
        task_id = task.get("id")
        exec_info = exec_tasks_info.get(task_id)
        if exec_info is None:
            print(f"[WARN] Execution info missing for task {task_id} in {filename}; skipping flops computation")
            task["flops"] = 0.0
            # Runtime fallback if execution info missing
            task.setdefault("runtime", 0.0)
        else:
            task["flops"] = compute_flops(exec_info, machine_info, task_id, filename)
            # Populate runtime from execution info for consistency with model features
            task["runtime"] = float(exec_info.get("runtimeInSeconds", 0.0))
        task["memory"] = compute_memory(task, file_sizes)

    # Write back full modified JSON
    output_path = output_dir / filename
    with output_path.open('w') as out_f:
        json.dump(data, out_f, indent=4)
    print(f"[INFO] Converted {filename} -> {output_path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert wfcommons JSON workflows to project format.")
    parser.add_argument("--input_dir", default="configs/wfcommons", help="Directory containing wfcommons JSON files")
    parser.add_argument("--output_dir", default="data/workflows", help="Directory to write converted JSON files")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory {input_dir} does not exist or is not a directory")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in input_dir.glob('*.json') if p.is_file())
    if not json_files:
        print(f"[WARN] No JSON files found in {input_dir}")
        return

    for jf in json_files:
        process_file(jf, output_dir)


if __name__ == "__main__":
    main()
