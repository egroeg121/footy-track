# SAM 3 / SAM 3.1 as a slow, offline background ball labeller

Write-up: `docs/experiments/sam3_offline_labeller.md` (numbers: `docs/experiments/sam3_offline_labeller.json`).  Prefect flows: `~/code/jobs/footy_sam3.py` + `sam3.prefect.yaml`.

| file | where it runs | what |
|---|---|---|
| `spec.py`, `cuts.py` | mini (stdlib / ffmpeg) | evaluation spec: clips, GT (ball_gt_marks), oracle and RT-DETR seeds, scene cuts -> `spec_v2.json` |
| `sam3_track.py` | desktop GPU (`~/sam3-exp/venv`) | SAM 3 / 3.1 crop-following box / point tracker from a seed (+ text-only tracking, text recovery, seed check, frame step), fp16, sm_75 patches |
| `sam2_adapter.py` | desktop GPU (`~/sam2-exp/venv`) | the round-3 SAM2.1-small crop512 baseline on the same seeds / windows |
| `sam3_seedfind.py` | desktop GPU | text-prompt ball detection on GT frames (full frame / 540 px / 384 px tiles, 6 prompts, optional exemplar box) |
| `sam3_label.py`, `make_seeds.py` | desktop GPU / mini | **production**: seeds file -> per-clip forward / backward tracks, S1 gate, labels JSONL (`sam3-label-match` flow) |
| `score.py`, `analyze.py`, `final_tables.py`, `seedfind_score.py`, `eval_labels.py`, `assemble.py`, `plot.py` | mini CPU (numpy) | scoring, paired bootstraps, tables |
| `gtcheck.py`, `viz.py`, `viz_labels.py` | mini CPU | GT sanity sheets, track contact sheets / GIFs, label sheets |
| `run_all.sh` | mini | every Prefect run of the study in queue order |
| `prof.py` | desktop GPU | backbone profiling (the bf16 finding) |
