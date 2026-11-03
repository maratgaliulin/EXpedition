#!/usr/bin/env bash
# Ae dense-heatmap folds (sp_train recipe) for the sheet-centre prior study. Runs alongside the LiDAR queue with a
# small batch so both fit in 16 GB; the study itself is scripts/ae_hm_prior_oof.py.
set -u
cd "$(dirname "$0")/.."
PY=./myenv/bin/python
BS=${AE_BS:-4}
for k in 0 1 2 3; do
  if [ ! -f "work/ae/hm_f$k.pt" ]; then
    echo "=== ae fold $k $(date +%H:%M) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -m koz3.sp_train --mod Ae --fold "$k" --iters 12000 --bs "$BS" > "work/ae_hm_f$k.log" 2>&1
  fi
done
echo "=== ae training done $(date +%H:%M) ==="
