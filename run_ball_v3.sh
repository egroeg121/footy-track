#!/bin/bash
# Run A: ball-only fine-tune of the PRODUCTION JOINT weights (rtdetr_1920_v11) with a reset 1-class head, on ball_v3
# (Roboflow v11 ball boxes + ball_v2, ~1617 train / 340 val). Same recipe as the pilot (imgsz 1920 rect batch 1,
# AdamW lr0 1e-4) but 25 epochs: ~10.5 min/epoch (1617 imgs x ~0.36 s + val) => ~4.4 GPU-h max, patience 10 may stop earlier.
# Launch on the desktop:  tmux new -d -s ball_v3_a 'bash ~/footy-track-jobs/ball-train-prep/run_ball_v3.sh'
# Then, on merlin:        TAG=ball_v3_a bash post_ball_eval.sh
# Preconditions (checked below): GPU idle; also confirm the Prefect gpu pool is empty (pf.sh status / flow-run ls).
trap '' INT HUP   # an ssh hiccup once delivered KeyboardInterrupt into augmentation and killed a run
cd ~/footy-track-jobs/ball-train-prep
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "${used:-9999}" -lt 500 ] || { echo "GPU busy (${used} MiB used); refusing to start"; exit 1; }
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 > ~/footy-track-jobs/ball-train-prep/vram_ball_v3_a.csv &
VP=$!
PYTHONPATH=$PWD/src ~/code/jobs-cache/footy-venv/bin/python -m footy_track.scripts.train_object_detector \
  --model ~/footy-track-web/repo/model_saves/detector/rtdetr_1920_v11/best.pt \
  --local-dataset ~/footy/staging/ball_v3 --imgsz 1920 --rect --batch 1 --epochs 25 --patience 10 --freeze 0 \
  --optimizer AdamW --lr0 1e-4 --workers 4 --cache none --device 0 \
  --project ~/footy/runs/ball_train --name-prepend ball_v3_a "$@" 2>&1 | tee ball_v3_a.log
kill $VP 2>/dev/null
