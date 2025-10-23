#!/usr/bin/env python3
"""Quick validation for converted wfcommons workflow JSON files.

Checks each *.json in a directory to ensure every task has non-negative
flops and memory fields. Optionally reports summary statistics.

Usage:
  python scripts/validate_workflows.py --dir data/workflows
"""
import json
import argparse
from pathlib import Path
import sys
from collections import deque, defaultdict
from statistics import mean


def _extract_tasks(data: dict):
    wf = data.get('workflow', {})
    if 'tasks' in wf and isinstance(wf.get('tasks'), list):
        return wf.get('tasks', [])
    spec = wf.get('specification', {})
    tasks = spec.get('tasks', [])
    return tasks if isinstance(tasks, list) else []


def compute_stats(tasks):
    # Build dependencies (parents) normalized list
    id_to_task = {t.get('id', f'idx_{i}'): t for i, t in enumerate(tasks) if isinstance(t, dict)}
    parents_map = {tid: set() for tid in id_to_task}
    children_map = {tid: set() for tid in id_to_task}
    for tid, t in id_to_task.items():
        deps = t.get('dependencies') or t.get('parents') or []
        if isinstance(deps, list):
            for p in deps:
                if p in parents_map:  # only consider known tasks
                    parents_map[tid].add(p)
                    children_map[p].add(tid)

    # Sources & sinks
    sources = [tid for tid, ps in parents_map.items() if not ps]
    sinks = [tid for tid, cs in children_map.items() if not cs]

    # Width (max parallel ready set) approximation via level decomposition (BFS over dag)
    in_degree = {tid: len(parents_map[tid]) for tid in id_to_task}
    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    level_widths = []
    visited = set()
    while queue:
        level_size = len(queue)
        level_widths.append(level_size)
        next_queue = deque()
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for child in children_map[cur]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        queue = next_queue
    width = max(level_widths) if level_widths else 0
    depth = len(level_widths)

    flops_list = [float(t.get('flops', 0.0)) for t in id_to_task.values()]
    mem_list = [float(t.get('memory', 0.0)) for t in id_to_task.values()]
    runtime_list = [float(t.get('runtime', 0.0)) for t in id_to_task.values()]

    stats = {
        'task_count': len(id_to_task),
        'sources': len(sources),
        'sinks': len(sinks),
        'width': width,
        'depth': depth,
        'total_flops': sum(flops_list),
        'avg_flops': mean(flops_list) if flops_list else 0.0,
        'max_flops': max(flops_list) if flops_list else 0.0,
        'total_memory': sum(mem_list),
        'avg_memory': mean(mem_list) if mem_list else 0.0,
        'max_memory': max(mem_list) if mem_list else 0.0,
        'avg_runtime': mean(runtime_list) if runtime_list else 0.0,
        'max_runtime': max(runtime_list) if runtime_list else 0.0,
    }
    return stats


def validate_file(path: Path):
    with path.open('r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"JSON decode error: {e}"], {}
    tasks = _extract_tasks(data)
    errors = []
    for idx, t in enumerate(tasks):
        if not isinstance(t, dict):
            errors.append(f"Task index {idx} not a dict")
            continue
        if 'flops' not in t:
            errors.append(f"Task {t.get('id', idx)} missing 'flops'")
        if 'memory' not in t:
            errors.append(f"Task {t.get('id', idx)} missing 'memory'")
        if float(t.get('flops', -1)) < 0:
            errors.append(f"Task {t.get('id', idx)} has negative flops: {t.get('flops')}")
        if float(t.get('memory', -1)) < 0:
            errors.append(f"Task {t.get('id', idx)} has negative memory: {t.get('memory')}")
    stats = compute_stats(tasks)
    return len(errors) == 0, errors, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate workflow JSON files for flops & memory.")
    parser.add_argument('--dir', default='data/workflows', help='Directory containing workflow JSONs')
    parser.add_argument('--summary', action='store_true', help='Print aggregate stats across all workflows.')
    parser.add_argument('--csv', metavar='OUT.csv', help='Optional path to write per-workflow stats CSV.')
    args = parser.parse_args(argv)
    directory = Path(args.dir)
    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        sys.exit(1)
    json_files = sorted(p for p in directory.glob('*.json'))
    if not json_files:
        print(f"[WARN] No JSON files in {directory}")
        return
    total = len(json_files)
    passed = 0
    all_stats = []
    for jf in json_files:
        ok, errs, stats = validate_file(jf)
        status = 'PASS' if ok else 'FAIL'
        print(f"[{status}] {jf.name} | tasks={stats.get('task_count', 0)} width={stats.get('width', 0)} depth={stats.get('depth', 0)} sources={stats.get('sources', 0)} sinks={stats.get('sinks', 0)} total_flops={int(stats.get('total_flops', 0))} total_mem={int(stats.get('total_memory', 0))}")
        if not ok:
            for e in errs:
                print(f"       - {e}")
        else:
            passed += 1
        stats_row = {'file': jf.name}
        stats_row.update(stats)
        stats_row['status'] = status
        all_stats.append(stats_row)
    print(f"\nSummary: {passed}/{total} files passed validation.")
    if args.summary and all_stats:
        agg = defaultdict(float)
        for s in all_stats:
            agg['total_tasks'] += s['task_count']
            agg['total_flops'] += s['total_flops']
            agg['total_memory'] += s['total_memory']
        avg_width = mean([s['width'] for s in all_stats]) if all_stats else 0
        avg_depth = mean([s['depth'] for s in all_stats]) if all_stats else 0
        print("Aggregate Stats:")
        print(f"  Total Tasks: {int(agg['total_tasks'])}")
        print(f"  Total FLOPs: {int(agg['total_flops'])}")
        print(f"  Total Memory: {int(agg['total_memory'])}")
        print(f"  Avg Width: {avg_width:.2f}")
        print(f"  Avg Depth: {avg_depth:.2f}")
    if args.csv and all_stats:
        import csv
        fieldnames = list(all_stats[0].keys())
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"[INFO] Stats CSV written to {args.csv}")
    if passed != total:
        sys.exit(2)

if __name__ == '__main__':
    main()
