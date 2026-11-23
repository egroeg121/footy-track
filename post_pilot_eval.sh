#!/bin/bash
# Post-pilot evaluation of the ball_pilot detector on the ball-eval-v1 panel (7 labelled heldout matches).
# Run ON MERLIN (needs the panel, human marks and segment videos) AFTER the desktop training in tmux `ball_pilot`
# has finished. GPU suggest runs on the desktop; scoring runs on merlin (CPU, nice'd).
#   bash post_pilot_eval.sh [best.pt path on desktop]     (default: newest ball_pilot_* run's weights/best.pt)
# Baseline to beat (rtdetr_1920_v11@b4ea4b22, --blind-only, match-averaged r10):
#   recall 0.316 / precision 0.365   (scratchpad/base--blind-only.json)
# NOTE (2026-11-23): the old baseline.json (0.316/0.365) is invalid: 4 of 7 matches scored 0.00 (predictions missing in that scoring run), and the live panel dir has no b4ea4b22 file for leeds. Compare pilot against the base1920 row this script produces.
# NOTE: the baseline suggestions were made by the nightly panel job at imgsz 1280 (default). This script follows the
# request (imgsz 1920), so the comparison is confounded by input size; set BASE1920=1 to also re-suggest the baseline
# weights at 1920 for a like-for-like row (costs one more GPU pass, ~10 min).
set -euo pipefail
D=george@100.74.96.107
FD=/mnt/storage/footy_data
PANEL=ball-eval-v1
CODE=/home/george/footy-track-wt/ball-train-prep            # merlin: prep/ball-training worktree (score code)
PY=/mnt/storage/venvs/footy-track/bin/python
RPY='~/code/jobs-cache/footy-venv/bin/python'
STAGE='~/footy/eval_pilot'                                   # desktop scratch (mirrors the data-root layout)
OUT=/home/george/pilot_eval; mkdir -p $OUT
MATCHES=(arsenal_nottinghamforest brentford-vs-chelsea-2026-27 everton-vs-crystal-palace-2026-27 ipswich-town-vs-liverpool-2026-27 manchester-united-vs-manchester-city-2026-27 nottingham-forest-vs-leeds-united-2026-27 sunderland-vs-fulham-2026-27)
BASELINE_JSON=/tmp/claude-1000/-home-george-footy-track/b6618b2c-88f2-49c4-a884-5b935c4f0324/scratchpad/base--blind-only.json
cp -n "$BASELINE_JSON" $OUT/baseline.json 2>/dev/null || true

# 0. training must be finished, weights must exist
if ssh $D 'tmux has-session -t ball_pilot 2>/dev/null'; then echo "ball_pilot still running; wait"; exit 1; fi
W=${1:-$(ssh $D 'ls -dt ~/footy/runs/ball_train/ball_pilot_*/weights/best.pt | head -1')}
echo "weights: $W"; ssh $D "test -f $W && nvidia-smi --query-gpu=memory.used --format=csv,noheader"
# (also check the gpu pool is idle in Prefect before using the GPU)

ssh $D "mkdir -p $STAGE/fd/ball_eval_panel/$PANEL"
# 1. stage panels + ONLY the needed segment videos on the desktop (~2 GB), rewriting segments_json to desktop paths
$PY - "$STAGE" "${MATCHES[@]}" <<'PY'
import json, os, subprocess, sys, pathlib
stage_arg, matches = sys.argv[1], sys.argv[2:]
home = os.path.expanduser("~")
D = "george@100.74.96.107"; FD = "/mnt/storage/footy_data"; P = "ball-eval-v1"
tmp = pathlib.Path("/tmp/pilot_stage"); (tmp / "ball_eval_panel" / P).mkdir(parents=True, exist_ok=True)
files = []
for m in matches:
    doc = json.load(open(f"{FD}/ball_eval_panel/{P}/{m}.json"))
    sj = doc["segments_json"]; seg = json.load(open(sj))["segments"]
    by = {os.path.splitext(s["file"])[0]: s["file"] for s in seg}
    rel = os.path.relpath(os.path.dirname(sj), FD)
    for c in sorted({w["clip"] for w in doc["windows"]}):
        files.append(f"{rel}/{by[c]}")
    files.append(f"{rel}/segments.json")
    doc["segments_json"] = f"{stage_arg.replace('~', '/home/george')}/fd/{rel}/segments.json"
    json.dump(doc, open(tmp / "ball_eval_panel" / P / f"{m}.json", "w"))
open(tmp / "files.txt", "w").write("\n".join(files) + "\n")
subprocess.check_call(["rsync", "-a", "--files-from", str(tmp / "files.txt"), FD + "/", f"{D}:{stage_arg.replace('~/', '')}/fd/"])
subprocess.check_call(["rsync", "-a", str(tmp / "ball_eval_panel") + "/", f"{D}:{stage_arg.replace('~/', '')}/fd/ball_eval_panel/"])
subprocess.check_call(["rsync", "-a", f"{FD}/match_splits.json", f"{FD}/clip_splits.json", f"{D}:{stage_arg.replace('~/', '')}/fd/"])
PY

# 2. GPU suggest at 1920 on the desktop, one match at a time (new sha8 -> new suggest file in the scratch root only)
MARGS=(); for m in "${MATCHES[@]}"; do MARGS+=(--match "$m"); done
ssh $D "cd ~/footy-track-jobs/ball-train-prep && FOOTY_DATA=$STAGE/fd FOOTY_MATCH_SPLITS=$STAGE/fd/match_splits.json FOOTY_CLIP_SPLITS=$STAGE/fd/clip_splits.json \
  PYTHONPATH=\$PWD/src $RPY -m footy_track.scripts.ball_eval_panel --root $STAGE/fd --panel-version $PANEL suggest ${MARGS[*]} \
  --weights $W --device 0 --imgsz 1920 --batch 4 2>&1 | tee ~/footy/eval_pilot/suggest.log"
if [ "${BASE1920:-0}" = 1 ]; then
  ssh $D "cd ~/footy-track-jobs/ball-train-prep && FOOTY_DATA=$STAGE/fd FOOTY_MATCH_SPLITS=$STAGE/fd/match_splits.json FOOTY_CLIP_SPLITS=$STAGE/fd/clip_splits.json \
    PYTHONPATH=\$PWD/src $RPY -m footy_track.scripts.ball_eval_panel --root $STAGE/fd --panel-version $PANEL suggest ${MARGS[*]} \
    --weights ~/footy-track-web/repo/model_saves/detector/rtdetr_1920_v11/best.pt --device 0 --imgsz 1920 --batch 4"
fi

# 3. pull the suggest files back; convert to pixel jsonl {clip, frame, x, y, score} (centre of the box, px of the panel frame)
rsync -a --include='*.suggest.*.json' --exclude='*' $D:footy/eval_pilot/fd/ball_eval_panel/$PANEL/ $OUT/sugg/
SHA8=$(ssh $D "sha256sum $W | cut -c1-8")
BASE8=b4ea4b22   # production rtdetr_1920_v11 sha8 (the baseline @1280 in baseline.json used this file from the panel dir)
RUNS=("pilot:$SHA8"); [ "${BASE1920:-0}" = 1 ] && RUNS+=("base1920:$BASE8")
M=(); for m in "${MATCHES[@]}"; do M+=(--match "$m"); done
for run in "${RUNS[@]}"; do
  TAG=${run%%:*}; SH=${run##*:}
  $PY - "$TAG" "$SH" "${MATCHES[@]}" <<'PY'
import json, sys
tag, sha8, matches = sys.argv[1], sys.argv[2], sys.argv[3:]
FD = "/mnt/storage/footy_data"; OUT = "/home/george/pilot_eval"
rows = 0
with open(f"{OUT}/{tag}_pred.jsonl", "w") as o:
    for m in matches:
        doc = json.load(open(f"{FD}/ball_eval_panel/ball-eval-v1/{m}.json")); W, H = doc["width"], doc["height"]
        s = json.load(open(f"{OUT}/sugg/{m}.suggest.{sha8}.json"))
        for k, v in s.items():
            clip, frame = k.rsplit(":", 1)
            if v is None or "x" not in v:
                continue  # tracklet extras / no candidate -> no prediction
            o.write(json.dumps({"clip": clip, "frame": int(frame), "x": (v["x"] + v["w"] / 2) * W,
                                "y": (v["y"] + v["h"] / 2) * H, "score": v["conf"]}) + "\n"); rows += 1
print(tag, "pred rows", rows)
PY
  # score on merlin: --blind-only, explicit --match per match, px units, draws 500, seed 0 (same as the baseline)
  (cd $CODE && OMP_NUM_THREADS=2 nice -n 5 env PYTHONPATH=$CODE/src $PY -m footy_track.scripts.ball_eval_panel --root $FD --panel-version $PANEL \
    score "${M[@]}" --pred $OUT/${TAG}_pred.jsonl --pred-format jsonl --pred-units px --blind-only --model "${TAG}@$SH" --out $OUT/${TAG}--blind-only.json >/dev/null)
done

# compare (match-averaged + frame-weighted, r10/r25, with bootstrap CIs; per-match recall/precision r10)
$PY - <<'PY'
import json, os
O = "/home/george/pilot_eval"
runs = {"base1280": "baseline.json", "base1920": "base1920--blind-only.json", "pilot1920": "pilot--blind-only.json"}
docs = {k: json.load(open(f"{O}/{v}")) for k, v in runs.items() if os.path.exists(f"{O}/{v}")}
f = lambda d: f"{d['est']:.3f} ({d['lo']:.3f}-{d['hi']:.3f})"
for k in ("match_averaged", "frame_weighted"):
    for r in ("r10", "r25"):
        for met in ("recall", "precision"):
            print(f"{k:15s} {r} {met:9s} " + "  ".join(f"{n} {f(d['pooled'][k][r][met])}" for n, d in docs.items()))
g = lambda mm, k: (lambda e: "n/a" if e is None else f"{e:.2f}")(((mm.get("metrics") or {}).get("r10") or {}).get(k, {}).get("est") if mm.get("metrics") else None)
print("per match r10 (recall / precision)")
for m in docs["pilot1920"]["matches"]:
    print(f"{m:48s} " + "  ".join(f"{n} {g(d['matches'][m],'recall')}/{g(d['matches'][m],'precision')}" for n, d in docs.items()))
PY
