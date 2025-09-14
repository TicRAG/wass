#!/usr/bin/env bash
# WASS-RAG end-to-end benchmark runner (no environment setup / no chart generation)
# Usage: bash scripts/run_wass_scheduler_benchmark.sh [--force-train] [--episodes 1000]
# Optional env vars:
#   CONFIG=configs/experiment.yaml
#   MODEL=models/improved_wass_drl.pth
#   EPISODES=1000
#   RAG_JSON=data/wrench_rag_knowledge_base.json
#   RAG_PKL=data/wrench_rag_knowledge_base.pkl
#   EXTRA_TRAIN_ARGS=""  (e.g. "--episodes 1200")

set -Eeuo pipefail
IFS=$'\n\t'

# -------------- Configuration --------------
CONFIG=${CONFIG:-configs/experiment.yaml}
MODEL=${MODEL:-models/improved_wass_drl.pth}
EPISODES=${EPISODES:-1000}
RAG_JSON=${RAG_JSON:-data/wrench_rag_knowledge_base.json}
RAG_PKL=${RAG_PKL:-data/wrench_rag_knowledge_base.pkl}
RESULTS_DIR=results/wrench_experiments
RESULTS_JSON=${RESULTS_DIR}/detailed_results.json
SUMMARY_TXT=${RESULTS_DIR}/summary.txt
FORCE_TRAIN=false

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --force-train) FORCE_TRAIN=true; shift ;;
    --episodes) EPISODES=$2; shift 2 ;;
    --*) echo "[WARN] Unknown option: $1"; shift ;;
    *) echo "[WARN] Ignoring positional arg: $1"; shift ;;
  esac
done

# -------------- Helpers --------------
color() { local c="$1"; shift; printf "\033[%sm%s\033[0m" "$c" "$*"; }
info()  { echo "$(color 36 [INFO]) $*"; }
warn()  { echo "$(color 33 [WARN]) $*"; }
err()   { echo "$(color 31 [ERR ]) $*" >&2; }
succ()  { echo "$(color 32 [OK  ]) $*"; }

require_python() {
  if ! command -v python >/dev/null 2>&1; then
    err "python not found in PATH. Activate your environment first."; exit 1; fi
}

# -------------- Pre-flight --------------
require_python
mkdir -p "$(dirname "$MODEL")" "$RESULTS_DIR" results

if [[ ! -f "$CONFIG" ]]; then
  err "Config not found: $CONFIG"; exit 1; fi

# -------------- Step 1: Train (if needed) --------------
if [[ "$FORCE_TRAIN" == true || ! -f "$MODEL" ]]; then
  info "Training DRL model -> $MODEL (episodes=$EPISODES)"
  python scripts/improved_drl_trainer.py \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --output "$MODEL" || { err "Training failed"; exit 1; }
  succ "Model training complete"
else
  succ "Model exists, skip training (use --force-train to retrain)"
fi

# -------------- Step 2: Knowledge Base Check --------------
if [[ -f "$RAG_JSON" ]]; then
  succ "RAG JSON knowledge base found: $RAG_JSON"
elif [[ -f "$RAG_PKL" ]]; then
  warn "JSON KB missing; PKL present ($RAG_PKL). Consider converting to JSON for best compatibility."
else
  warn "No RAG knowledge base found. WASS-RAG will fall back to default embedded cases."
fi

# -------------- Step 3: Run Scheduler Comparison --------------
info "Running multi-scheduler WRENCH experiment"
export WASS_DRL_MODEL="$MODEL"
python experiments/wrench_real_experiment.py || { err "Experiment run failed"; exit 1; }

if [[ ! -f "$RESULTS_JSON" ]]; then
  err "Results JSON not found: $RESULTS_JSON"; exit 1; fi
succ "Experiment results stored: $RESULTS_JSON"

# -------------- Step 4: Summarize Metrics --------------
info "Aggregating makespan statistics"
python - <<'PY'
import json, numpy as np, os, sys
results_path = os.environ.get('RESULTS_JSON','results/wrench_experiments/detailed_results.json')
summary_path = os.environ.get('SUMMARY_TXT','results/wrench_experiments/summary.txt')
with open(results_path,'r',encoding='utf-8') as f:
    data = json.load(f)
rows = data.get('results', [])
if not rows:
    print('[ERR] No result rows found', file=sys.stderr); sys.exit(2)
# Group by scheduler
from collections import defaultdict
sched = defaultdict(list)
by_size = {}
for r in rows:
    sched[r['scheduler_name']].append(r['makespan'])
    by_size.setdefault(r['task_count'],{}).setdefault(r['scheduler_name'],[]).append(r['makespan'])
# Compute global stats
stats = {}
for k,v in sched.items():
    a=np.array(v)
    stats[k]={
        'avg':float(a.mean()),
        'std':float(a.std()),
        'best':float(a.min()),
        'count':int(a.size)
    }
# Determine baseline (exclude advanced schedulers)
baseline_candidates=[k for k in stats if k not in ('WASS-RAG','WASS-DRL')]
if baseline_candidates:
    baseline_best=min(baseline_candidates, key=lambda k: stats[k]['avg'])
else:
    baseline_best=None
lines=[]
lines.append('== Global Scheduler Performance ==')
lines.append(f"{'Scheduler':<15}{'Avg':>10}{'Std':>10}{'Best':>10}{'Count':>8}")
for k in sorted(stats, key=lambda x: stats[x]['avg']):
    s=stats[k]
    lines.append(f"{k:<15}{s['avg']:>10.2f}{s['std']:>10.2f}{s['best']:>10.2f}{s['count']:>8}")
if 'WASS-RAG' in stats and baseline_best:
    imp=(stats[baseline_best]['avg']-stats['WASS-RAG']['avg'])/stats[baseline_best]['avg']
    lines.append(f"\nWASS-RAG Improvement vs Best Baseline ({baseline_best}): {imp*100:.2f}%")
lines.append('\n== Per Workflow Size (Average Makespan) ==')
for size in sorted(by_size):
    lines.append(f"Workflow Size {size}")
    avgs={k:np.mean(v) for k,v in by_size[size].items()}
    for k in sorted(avgs, key=avgs.get):
        lines.append(f"  {k:<15} {avgs[k]:.2f}")
    if 'WASS-RAG' in avgs:
        base_candidates=[k for k in avgs if k not in ('WASS-RAG','WASS-DRL')]
        if base_candidates:
            base_best=min(base_candidates, key=lambda kk: avgs[kk])
            imp=(avgs[base_best]-avgs['WASS-RAG'])/avgs[base_best]
            lines.append(f"  -> Improvement vs {base_best}: {imp*100:.2f}%")
with open(summary_path,'w',encoding='utf-8') as f:
    f.write('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY

succ "Summary written: ${SUMMARY_TXT}"

# -------------- Completion --------------
info "Pipeline completed. Key artifacts:"
echo "  Model:        $MODEL"
echo "  Knowledge KB: ${RAG_JSON:-'(JSON missing)'}"
echo "  Results:      $RESULTS_JSON"
echo "  Summary:      $SUMMARY_TXT"

succ "Done."