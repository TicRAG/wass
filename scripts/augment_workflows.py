import os
import sys
import json
import random
from pathlib import Path
from copy import deepcopy
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Simple augmentation strategies:
# 1. Feature perturbation: runtime, flops, memory multiplied by (1 + noise)
# 2. Task duplication: optionally create new tasks derived from existing leaves with scaled features
# 3. ID renaming: append synthetic suffix _AUGk to ensure uniqueness
# Structure (parents/children) preserved to maintain DAG semantics.

DEFAULT_SOURCE_DIR = Path("data/workflows/training")
DEFAULT_OUTPUT_DIR = Path("data/workflows/training_aug")


def load_workflow(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def save_workflow(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force top-level workflow_name included for WRENCH compatibility
    data['workflow_name'] = data.get('workflow_name', data.get('name', path.stem))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    # Debug: confirm write
    if os.environ.get('AUG_DEBUG') == '1':
        print(f"[AUG] Saved {path.name} with workflow_name={data.get('workflow_name')}")


def perturb_task(task: dict, runtime_noise: float, flops_noise: float, memory_noise: float):
    # Apply multiplicative noise; ensure non-negative
    for key, noise in [("runtime", runtime_noise), ("flops", flops_noise), ("memory", memory_noise)]:
        if key in task and isinstance(task[key], (int, float)):
            base = task[key]
            task[key] = max(0.0, base * (1.0 + noise))
    return task


def duplicate_leaf_tasks(tasks: list, max_duplicates: int, duplicate_scale: float):
    # Identify leaves: tasks that are not parents of any other (i.e., appear only as parents list is empty or not in other parents?)
    child_set = set()
    for t in tasks:
        for c in t.get('children', []):
            child_set.add(c)
    leaves = [t for t in tasks if t['id'] not in child_set]
    new_tasks = []
    counter = 0
    for leaf in leaves:
        if counter >= max_duplicates:
            break
        clone = deepcopy(leaf)
        counter += 1
        clone_suffix = f"_AUGD{counter}"
        clone['id'] = clone['id'] + clone_suffix
        clone['name'] = clone['name'] + clone_suffix
        # Scale features
        for k in ['runtime', 'flops', 'memory']:
            if k in clone and isinstance(clone[k], (int, float)):
                clone[k] = clone[k] * duplicate_scale
        # Keep children empty; treat as additional terminal task
        clone['children'] = []
        # Parents remain same to keep dependency chain; or we can detach so it's independent
        # Here: keep parents to maintain coherence
        new_tasks.append(clone)
    return tasks + new_tasks


def augment_workflow(data: dict, variant_index: int, cfg: dict):
    wf = deepcopy(data)
    spec = wf.get('workflow', {}).get('specification', {})
    tasks = spec.get('tasks', [])
    if not tasks:
        return wf
    # Capture execution tasks if present to keep runtimes aligned after ID renaming
    execution_section = wf.get('workflow', {}).get('execution', {})
    exec_tasks = execution_section.get('tasks', []) if isinstance(execution_section, dict) else []
    exec_task_map = {t.get('id'): t for t in exec_tasks if isinstance(t, dict)}

    # Random seed for reproducibility per variant
    random.seed(cfg['seed'] + variant_index)

    # Perturbation
    for t in tasks:
        runtime_noise = random.uniform(-cfg['runtime_jitter'], cfg['runtime_jitter'])
        flops_noise = random.uniform(-cfg['flops_jitter'], cfg['flops_jitter'])
        memory_noise = random.uniform(-cfg['memory_jitter'], cfg['memory_jitter'])
        perturb_task(t, runtime_noise, flops_noise, memory_noise)
        # Rename IDs/names to ensure uniqueness globally
        t['id'] = f"{t['id']}_AUG{variant_index}"
        t['name'] = f"{t['name']}_AUG{variant_index}"
        # Update children references (they refer to original IDs) -> rename accordingly
        t['children'] = [f"{cid}_AUG{variant_index}" for cid in t.get('children', [])]
        # Parents will be updated after a mapping is built
        # If execution task exists adjust its id and runtime with same noise factor (approx runtime_noise multiplier)
        original_exec = exec_task_map.get(t['id'][:-len(f"_AUG{variant_index}")])
        if original_exec:
            # store for later rebuild
            pass

    # Build ID mapping for parents update
    id_map = {original['id'][:-len(f"_AUG{variant_index}")]: original['id'] for original in tasks}
    for t in tasks:
        new_parents = []
        for p in t.get('parents', []):
            mapped = id_map.get(p, f"{p}_AUG{variant_index}")
            if mapped not in new_parents:
                new_parents.append(mapped)
        t['parents'] = new_parents

    # Rebuild execution tasks with renamed IDs if execution section existed
    if exec_tasks:
        new_exec_tasks = []
        for et in exec_tasks:
            base_id = et.get('id')
            new_id = f"{base_id}_AUG{variant_index}"
            new_et = deepcopy(et)
            new_et['id'] = new_id
            # Optionally jitter runtime similar to task runtime noise using uniform small factor
            if 'runtimeInSeconds' in new_et and isinstance(new_et['runtimeInSeconds'], (int,float)):
                rt = new_et['runtimeInSeconds']
                jitter = random.uniform(-cfg['runtime_jitter'], cfg['runtime_jitter'])
                new_et['runtimeInSeconds'] = max(0.001, rt * (1.0 + jitter))
            new_exec_tasks.append(new_et)
        wf.setdefault('workflow', {}).setdefault('execution', {})['tasks'] = new_exec_tasks

    # Optional duplication of leaves
    if cfg['duplicate_leaves']:
        tasks = duplicate_leaf_tasks(tasks, cfg['max_leaf_duplicates'], cfg['duplicate_scale'])
        spec['tasks'] = tasks

    wf['name'] = wf.get('name', 'workflow') + f"-aug{variant_index}"
    # Preserve or create workflow_name (required by WRENCH parser) mirroring name
    original_wfname = data.get('workflow_name', data.get('name', 'workflow'))
    wf['workflow_name'] = f"{original_wfname}-aug{variant_index}"
    spec['tasks'] = tasks
    wf['workflow']['specification'] = spec
    return wf


def main():
    parser = argparse.ArgumentParser(description="Generate augmented WFCommons workflows for training expansion.")
    parser.add_argument('--source_dir', type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--variants_per_workflow', type=int, default=5)
    parser.add_argument('--runtime_jitter', type=float, default=0.3, help='Max +/- fractional change for runtime')
    parser.add_argument('--flops_jitter', type=float, default=0.25, help='Max +/- fractional change for flops')
    parser.add_argument('--memory_jitter', type=float, default=0.15, help='Max +/- fractional change for memory')
    parser.add_argument('--duplicate_leaves', action='store_true', help='Duplicate leaf tasks to increase terminal diversity.')
    parser.add_argument('--max_leaf_duplicates', type=int, default=3)
    parser.add_argument('--duplicate_scale', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = {
        'runtime_jitter': args.runtime_jitter,
        'flops_jitter': args.flops_jitter,
        'memory_jitter': args.memory_jitter,
        'duplicate_leaves': args.duplicate_leaves,
        'max_leaf_duplicates': args.max_leaf_duplicates,
        'duplicate_scale': args.duplicate_scale,
        'seed': args.seed
    }

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_files = sorted(source_dir.glob('*.json'))
    if not workflow_files:
        print(f"No workflows found in {source_dir}")
        return

    print(f"Found {len(workflow_files)} source workflows. Generating {args.variants_per_workflow} variants each -> {len(workflow_files)*args.variants_per_workflow} total.")

    for wf_file in workflow_files:
        try:
            data = load_workflow(wf_file)
        except Exception as e:
            print(f"Failed to load {wf_file.name}: {e}")
            continue
        for v in range(1, args.variants_per_workflow + 1):
            aug = augment_workflow(data, v, cfg)
            out_name = wf_file.stem + f"_aug{v}.json"
            out_path = output_dir / out_name
            save_workflow(aug, out_path)
        print(f"Generated {args.variants_per_workflow} variants for {wf_file.name}.")

    print(f"Augmentation complete. Files written to {output_dir}")

if __name__ == '__main__':
    main()
