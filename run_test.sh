#!/bin/bash
cd /data/workspace/traespace/wass
export PYTHONPATH="$PYTHONPATH:src"
python test_evaluation_metrics.py