#!/usr/bin/env bash
# Every Prefect run of the SAM 3 offline-labeller study, in the order they were queued (mini; PF = ~/.claude/skills/prefect-jobs/scripts/pf.sh).
# 0. spec (mini):   python spec.py --out /mnt/storage/exp/sam3/spec_v2.json --rtdetr-extra /mnt/storage/exp/sam3/rtdetr_full/labels ; python cuts.py ...
# 1. RT-DETR labels for the clips without machine labels / seed-finder comparison (stride 1)
PF=${PF:-~/.claude/skills/prefect-jobs/scripts/pf.sh}
SPEC=spec_v2; DEV="mancity_seg010,mancity_seg050,bournemouth_1st_seg070"
$PF run sam3-rtdetr/sam3-rtdetr spec=$SPEC exp=rtdetr_full clips=arsenal_bournemouth_1st_seg009,arsenal_bournemouth_1st_seg015,arsenal_bournemouth_1st_seg021,arsenal_bournemouth_1st_seg031,arsenal_bournemouth_1st_seg035,arsenal_bournemouth_1st_seg060,arsenal_mancity_part093,arsenal_mancity_seg101,arsenal_mancity_seg134,arsenal_mancity_seg135,arsenal_example
$PF run sam3-rtdetr/sam3-rtdetr spec=$SPEC exp=rtdetr_dev clips=bournemouth_1st_seg070,mancity_seg010,mancity_seg031,mancity_seg050
# 2. dev sweep (tune on dev only; 2 oracle seeds per clip, 120 frames each way)
dev() { $PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=$DEV "config=$1" exp=dev_sweep max_reach=${2:-120} n_seeds=2; }
dev '{"model":"sam3","prompt":"box","crop":512}'
$PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=$DEV engine=sam2 'config={"model":"sam2.1-small","prompt":"box","crop":512}' exp=dev_sweep max_reach=120 n_seeds=2
for p in point boxpoint; do dev "{\"model\":\"sam3\",\"prompt\":\"$p\",\"crop\":512}"; done
for c in 384 256 0 768; do dev "{\"model\":\"sam3\",\"prompt\":\"box\",\"crop\":$c}"; done
dev '{"model":"sam3.1","prompt":"box","crop":512}'; dev '{"model":"sam3.1","prompt":"point","crop":512}'
dev '{"model":"sam3","prompt":"text:ball","crop":512,"tag":"t"}' 80; dev '{"model":"sam3","prompt":"text:yellow ball","crop":512,"tag":"t"}' 80
dev '{"model":"sam3","prompt":"box","crop":512,"frame_step":2,"tag":"step2"}'
for c in '{"tag":"base"}' '{"dtype":"fp32","tag":"fp32"}' '{"offload":true,"tag":"offload"}' '{"memsel":false,"tag":"nomemsel"}'; do
  $PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=mancity_seg050,bournemouth_1st_seg070 "config=$(python3 -c "import json,sys;print(json.dumps({'model':'sam3','prompt':'box','crop':512,**json.loads('$c')}))")" exp=dev_prec max_reach=60 n_seeds=1
done
# 3. text-prompt seed finder (final = held-out clips + astonvilla subset; dev for prompt choice; arsenal_example held-out frames)
$PF run sam3-seedfind/sam3-seedfind spec=$SPEC clips=final schemes=full,t540,t384 exp=seedfind
$PF run sam3-seedfind/sam3-seedfind spec=$SPEC clips=dev schemes=full,t384 exp=seedfind_dev max_per_clip=20
$PF run sam3-seedfind/sam3-seedfind spec=$SPEC clips=arsenal_example schemes=full,t540,t384 exp=seedfind 'only_frames={"arsenal_example":[59,160,162,163,165,166,167,168,169,171,173,175,177,178,190,194,195,196,197,198]}'
# 4. final: oracle seeds (reach 400) and RT-DETR conf>=0.8 seeds (reach 250)
$PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=final,example engine=sam2 'config={"model":"sam2.1-small","prompt":"box","crop":512}' exp=final_oracle max_reach=400
$PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=final,example 'config={"model":"sam3","prompt":"box","crop":512}' exp=final_oracle max_reach=400
$PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=final,example protocol=rtdetr 'config={"model":"sam3","prompt":"box","crop":512,"seed_check":"ball"}' exp=final_rtdetr max_reach=250
$PF run sam3-track-eval/sam3-track-eval spec=$SPEC clips=final,example protocol=rtdetr engine=sam2 'config={"model":"sam2.1-small","prompt":"box","crop":512}' exp=final_rtdetr max_reach=250
# 5. production demo
$PF run sam3-label-match/sam3-label-match match=demo seeds=/mnt/storage/exp/sam3/demo/seeds.jsonl clips_dir=/mnt/storage/footy_data/ballcheck_clips 'clip_files={"arsenal_bournemouth_1st_seg035":"bournemouth_seg035.mp4"}' 'config={"model":"sam3","prompt":"box","crop":512}' reach=150 max_seeds=3
