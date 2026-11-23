#!/bin/bash
# Run B: ball-only from the COCO-pretrained RT-DETR-L (no joint/ball-in-v11 knowledge), 1-class head from scratch, on ball_v3.
# Needs ~/footy/models/pretrained/rtdetr-l.pt (66.5 MB, ultralytics assets v8.2.0; fetched below if missing; desktop can reach github).
# 40 epochs x ~10.5 min = ~7 GPU-h max (patience 15). Launch only after run A (GPU is single-tenant):
#   tmux new -d -s ball_v3_b 'bash ~/footy-track-jobs/ball-train-prep/run_ball_v3_coco.sh'
# Then on merlin: TAG=ball_v3_b bash post_ball_eval.sh
trap '' INT HUP
cd ~/footy-track-jobs/ball-train-prep
PT=~/footy/models/pretrained/rtdetr-l.pt
if [ ! -s "$PT" ]; then
  mkdir -p "$(dirname "$PT")"
  curl -fsSL -m 600 -o "$PT" https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt || { echo "rtdetr-l.pt fetch failed"; exit 1; }
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${used:-9999}" -lt 500 ] || { echo "GPU busy (${used} MiB used); refusing to start"; exit 1; }
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 > ~/footy-track-jobs/ball-train-prep/vram_ball_v3_b.csv &
VP=$!
PYTHONPATH=$PWD/src ~/code/jobs-cache/footy-venv/bin/python -m footy_track.scripts.train_object_detector \
  --model "$PT" \
  --local-dataset ~/footy/staging/ball_v3 --imgsz 1920 --rect --batch 1 --epochs 40 --patience 15 --freeze 0 \
  --optimizer AdamW --lr0 1e-4 --workers 4 --cache none --device 0 \
  --project ~/footy/runs/ball_train --name-prepend ball_v3_b "$@" 2>&1 | tee ball_v3_b.log
kill $VP 2>/dev/null
