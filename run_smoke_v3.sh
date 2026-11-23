#!/bin/bash
# Smoke: prove ball_v3 loads and trains (1 epoch, 5% of train). Same flags as run A. Not a result.
cd ~/footy-track-jobs/ball-train-prep
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${used:-9999}" -lt 500 ] || { echo "GPU busy (${used} MiB used); refusing to start"; exit 1; }
PYTHONPATH=$PWD/src ~/code/jobs-cache/footy-venv/bin/python -m footy_track.scripts.train_object_detector \
  --model ~/footy-track-web/repo/model_saves/detector/rtdetr_1920_v11/best.pt \
  --local-dataset ~/footy/staging/ball_v3 --imgsz 1920 --rect --batch 1 --epochs 1 --fraction 0.05 --freeze 0 \
  --optimizer AdamW --lr0 1e-4 --workers 4 --cache none --device 0 \
  --project ~/footy/runs/ball_train --name-prepend ball_v3_smoke "$@" 2>&1 | tee smoke_v3.log
