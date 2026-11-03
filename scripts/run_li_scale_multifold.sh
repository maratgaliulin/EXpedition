#!/usr/bin/env bash
# Unattended cross-fold validation of the LiDAR scale pyramid (folds 1-3; fold 0 is already done).
# Per fold: train with the augx recipe, then score base vs pyramid on that fold's held-out sites.
# Logs: work/li_x0_f<k>.log (training), work/li_scale_f<k>.log (study).
set -u
cd "$(dirname "$0")/.."
PY=./myenv/bin/python
for k in 1 2 3; do
  if [ ! -f "work/li/li_x0_f$k.pt" ]; then
    echo "=== train fold $k $(date +%H:%M) ==="
    LI_AUG=1 LI_PASTE=1 RELIEF_EXTRA=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY -m koz3.li_train --fold "$k" --iters 12000 --tag _x0 > "work/li_x0_f$k.log" 2>&1
  fi
  echo "=== eval fold $k $(date +%H:%M) ==="
  RELIEF_EXTRA=1 $PY scripts/li_scale_tta_oof.py --fold "$k" --tag _x0 --scales 1.0,0.8,1.25 \
    > "work/li_scale_f$k.log" 2>&1
  grep -E "^== |regions better" "work/li_scale_f$k.log"
done
echo "=== all folds done $(date +%H:%M) ==="
