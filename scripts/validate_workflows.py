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


def validate_file(path: Path):
    with path.open('r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"JSON decode error: {e}"]
    tasks = []
    wf = data.get('workflow', {})
    if 'tasks' in wf:  # internal format
        tasks = wf.get('tasks', [])
    else:
        spec = wf.get('specification', {})
        tasks = spec.get('tasks', [])
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
    return len(errors) == 0, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate workflow JSON files for flops & memory.")
    parser.add_argument('--dir', default='data/workflows', help='Directory containing workflow JSONs')
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
    for jf in json_files:
        ok, errs = validate_file(jf)
        if ok:
            print(f"[PASS] {jf.name}")
            passed += 1
        else:
            print(f"[FAIL] {jf.name}")
            for e in errs:
                print(f"       - {e}")
    print(f"\nSummary: {passed}/{total} files passed validation.")
    if passed != total:
        sys.exit(2)

if __name__ == '__main__':
    main()
