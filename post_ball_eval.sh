#!/bin/bash
# Evaluate a new ball detector on the ball-eval-v1 panel (7 labelled heldout matches), ALL 7 and the CLEAN subset,
# against the existing production baseline at 1920 (/home/george/pilot_eval/base1920--blind-only.json) and the pilot.
# Run ON MERLIN (needs the panel, human marks and segment videos) AFTER the desktop training session has finished.
# GPU suggest runs on the desktop; scoring on merlin (CPU, nice'd).
#   TAG=ball_v3_a bash post_ball_eval.sh [best.pt path on desktop]
#     TAG      short run name (default ball_v3_a)   SESSION  tmux session to wait on (default = TAG)
#     DATASET  merlin dataset dir whose provenance.jsonl lists the training frames (default ball_v3); used for the leakage check
# "CLEAN" = panel matches with NO training frame (clip or match) in the new dataset's train split AND none in Roboflow v11
# (production's data). Computed, not assumed; if every match is clean the clean row equals the all-7 row.
# NOTE the old baseline.json (0.316/0.365 @1280) is invalid (4/7 matches scored 0.00); compare to base1920 only.
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
TAG=${TAG:-ball_v3_a}; SESSION=${SESSION:-$TAG}; DATASET=${DATASET:-/mnt/storage/footy_data/training_datasets/ball_v3}

# 0. training must be finished, weights must exist
if ssh $D "tmux has-session -t $SESSION 2>/dev/null"; then echo "$SESSION still running; wait"; exit 1; fi
W=${1:-$(ssh $D "ls -dt ~/footy/runs/ball_train/${TAG}_*/weights/best.pt | head -1")}
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

M=(); for m in "${MATCHES[@]}"; do M+=(--match "$m"); done
# 2. GPU suggest at 1920 on the desktop, one match at a time (new sha8 -> new suggest file in the scratch root only)
MARGS=(); for m in "${MATCHES[@]}"; do MARGS+=(--match "$m"); done
ssh $D "cd ~/footy-track-jobs/ball-train-prep && FOOTY_DATA=$STAGE/fd FOOTY_MATCH_SPLITS=$STAGE/fd/match_splits.json FOOTY_CLIP_SPLITS=$STAGE/fd/clip_splits.json \
  PYTHONPATH=\$PWD/src $RPY -m footy_track.scripts.ball_eval_panel --root $STAGE/fd --panel-version $PANEL suggest ${MARGS[*]} \
  --weights $W --device 0 --imgsz 1920 --batch 4 2>&1 | tee ~/footy/eval_pilot/suggest.log"
# 3. pull the suggest files back; convert to pixel jsonl {clip, frame, x, y, score} (centre of the box, px of the panel frame)
rsync -a --include='*.suggest.*.json' --exclude='*' $D:footy/eval_pilot/fd/ball_eval_panel/$PANEL/ $OUT/sugg/
SHA8=$(ssh $D "sha256sum $W | cut -c1-8")
$PY - "$TAG" "$SHA8" "${MATCHES[@]}" <<'PY'
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

# 4. leakage check -> CLEAN matches (no panel clip/match in the new dataset's train images or in Roboflow v11)
CLEAN=$(FOOTY_DATA=$FD PYTHONPATH=$CODE/src $PY - "$DATASET" "${MATCHES[@]}" <<'PY'
import glob, json, os, sys
from footy_track import splits as S
ds, matches = sys.argv[1], sys.argv[2:]
FD = "/mnt/storage/footy_data"
panel = {m: {S.canonical_clip_id(w["clip"]) for w in json.load(open(f"{FD}/ball_eval_panel/ball-eval-v1/{m}.json"))["windows"]} for m in matches}
used = set(); used_m = set()
for r in map(json.loads, open(f"{ds}/provenance.jsonl")):
    if r["split"] == "train":
        used.add(r["clip"]); used_m.add(S.match_key_of(r["clip"]))
for p in glob.glob(f"{FD}/detection_dataset/roboflow_dataset_11/*/images/*"):          # production's own data
    ref = S.parse_image_name(os.path.basename(p)); used.add(ref.clip_id); used_m.add(ref.match_key)
clean = [m for m in matches if not (panel[m] & used) and m not in used_m]
print(" ".join(clean))
sys.stderr.write(f"clean {len(clean)}/{len(matches)}: {clean}; overlapping: {[m for m in matches if m not in clean]}\n")
PY
)
echo "clean matches: $CLEAN"
CM=(); for m in $CLEAN; do CM+=(--match "$m"); done

# 5. score (same settings as the baseline: --blind-only, px units, 500 draws, seed 0): ALL 7 and CLEAN, for all three models
score() {  # score <tag> <sha8> <suffix> <match args...>
  local tag=$1 sha=$2 suf=$3; shift 3
  (cd $CODE && OMP_NUM_THREADS=2 nice -n 5 env PYTHONPATH=$CODE/src $PY -m footy_track.scripts.ball_eval_panel --root $FD --panel-version $PANEL \
    score "$@" --pred $OUT/${tag}_pred.jsonl --pred-format jsonl --pred-units px --blind-only --model "${tag}@$sha" --out $OUT/${tag}--blind-only${suf}.json >/dev/null)
}
score "$TAG" "$SHA8" "" "${M[@]}"
[ -n "$CLEAN" ] && score "$TAG" "$SHA8" ".clean" "${CM[@]}"
if [ -n "$CLEAN" ]; then   # baselines on the clean subset (all-7 versions already exist)
  score base1920 b4ea4b22 ".clean" "${CM[@]}"; score pilot "$(ssh $D "ls -dt ~/footy/runs/ball_train/ball_pilot_*/weights/best.pt | head -1 | xargs sha256sum | cut -c1-8")" ".clean" "${CM[@]}"
fi

# 6. compare (match-averaged + frame-weighted, r10/r25, bootstrap CIs; per-match recall/precision r10)
TAG=$TAG $PY - <<'PY'
import json, os
O = "/home/george/pilot_eval"; T = os.environ["TAG"]
for suf, title in (("", "ALL 7 matches"), (".clean", "CLEAN subset")):
    runs = {"base1920": f"base1920--blind-only{suf}.json", "pilot": f"pilot--blind-only{suf}.json", T: f"{T}--blind-only{suf}.json"}
    docs = {k: json.load(open(f"{O}/{v}")) for k, v in runs.items() if os.path.exists(f"{O}/{v}")}
    if T not in docs:
        continue
    print(f"\n=== {title} (n_matches={docs[T]['pooled']['n_matches']}) ===")
    f = lambda d: f"{d['est']:.3f} ({d['lo']:.3f}-{d['hi']:.3f})"
    for k in ("match_averaged", "frame_weighted"):
        for r in ("r10", "r25"):
            for met in ("recall", "precision"):
                print(f"{k:15s} {r} {met:9s} " + "  ".join(f"{n} {f(d['pooled'][k][r][met])}" for n, d in docs.items()))
    g = lambda mm, k: (lambda e: "n/a" if e is None else f"{e:.2f}")(((mm.get("metrics") or {}).get("r10") or {}).get(k, {}).get("est") if mm.get("metrics") else None)
    print("per match r10 (recall / precision)")
    for m in docs[T]["matches"]:
        print(f"{m:48s} " + "  ".join(f"{n} {g(d['matches'][m],'recall')}/{g(d['matches'][m],'precision')}" for n, d in docs.items()))
PY
