# Offline ball harness: reference numbers

Snapshot `mancity@c9cc63ecbef0` (full hash `c9cc63ecbef0fc7f004258e03f1e6de3ddb671273c5c300837298801bbaa68d0`, frozen 2026-11-14): 18 clips with GT and candidates, 9 test clips with 170 clean GT ball frames + 2 no_ball frames, 9 dev clips with 199 (contaminated) GT ball frames. Detector candidates: `machine_labels/` (RT-DETR-L @1280, conf >= 0.1). Pixels at 1920x1080.

Reproduce:

```bash
python -m footy_track.ball_offline snapshot --eval-set mancity
for m in b0 viterbi; do python -m footy_track.ball_offline predict --snapshot c9cc63ecbef0 --method $m --out $m.jsonl; done
python -m footy_track.ball_offline score --snapshot c9cc63ecbef0 --pred viterbi.jsonl --baseline b0.jsonl --roles test,dev --per-clip
python -m footy_track.ball_offline ceiling --snapshot c9cc63ecbef0
```

Detector ceiling (any candidate >= 0.1 within 10 px of GT): test 141 / 170 = 0.829, dev 190 / 199 = 0.955. A tracker that only selects among candidates cannot exceed it by much.

Reference (not reproduced here): the experiment branch's B1 solver (`exp/offline-ball-tracking` fc05732) scores recall 0.759 / precision 0.985 (129/131) on the same clean test set. Main's Viterbi minus that B1, paired bootstrap over frames, is +0.012 recall [-0.012, +0.035] and -0.014 precision [-0.044, +0.014] (3 frames only Viterbi hits, 1 only B1): no difference these labels can resolve.

## Main's Viterbi vs B0 (top-1 per frame, tau 0.4)

### test (clean, detector-uncontaminated)

| method | GT ball frames | recall@10px [95% Wilson] | precision@10px (TP/emitted on scored) | F1@10 | recall@5 / @20 | coverage | median err px |
|---|---|---|---|---|---|---|---|
| viterbi | 170 | 0.771 [0.70-0.83] | 0.970 (131/135) | 0.859 | 0.747 / 0.782 | 0.896 | 1.5 |
| b0 (baseline) | 170 | 0.741 [0.67-0.80] | 0.947 (126/133) | 0.832 | 0.729 / 0.747 | 0.856 | 1.5 |


| clip | GT | recall@10 | precision@10 |
|---|---|---|---|
| arsenal_mancity_seg010 | 110 | 0.85 | 0.96 |
| arsenal_mancity_seg011 | 1 | 1.00 | 1.00 |
| arsenal_mancity_seg031 | 53 | 0.60 | 1.00 |
| arsenal_mancity_seg094 | 1 | 1.00 | 1.00 |
| arsenal_mancity_seg101 | 1 | 1.00 | 1.00 |
| arsenal_mancity_seg115 | 1 | 1.00 | 1.00 |
| arsenal_mancity_seg134 | 1 | 1.00 | 1.00 |
| arsenal_mancity_seg135 | 1 | 1.00 | 1.00 |
| arsenal_mancity_seg140 | 1 | 0.00 | n/a |

### dev (detector-contaminated; optimistic)

| method | GT ball frames | recall@10px [95% Wilson] | precision@10px (TP/emitted on scored) | F1@10 | recall@5 / @20 | coverage | median err px |
|---|---|---|---|---|---|---|---|
| viterbi | 199 | 0.925 [0.88-0.95] | 0.948 (184/194) | 0.936 | 0.905 / 0.940 | 0.873 | 1.5 |
| b0 (baseline) | 199 | 0.899 [0.85-0.93] | 0.904 (179/198) | 0.902 | 0.884 / 0.905 | 0.824 | 1.5 |


| clip | GT | recall@10 | precision@10 |
|---|---|---|---|
| arsenal_mancity_seg020 | 8 | 0.38 | 0.60 |
| arsenal_mancity_seg030 | 26 | 0.85 | 1.00 |
| arsenal_mancity_seg050 | 80 | 0.95 | 0.95 |
| arsenal_mancity_seg060 | 20 | 1.00 | 0.91 |
| arsenal_mancity_seg080 | 6 | 1.00 | 1.00 |
| arsenal_mancity_seg090 | 6 | 1.00 | 1.00 |
| arsenal_mancity_seg100 | 24 | 0.96 | 0.96 |
| arsenal_mancity_seg110 | 28 | 0.96 | 0.96 |
| arsenal_mancity_seg150 | 1 | 1.00 | 1.00 |

Paired bootstrap, tracker minus baseline (test, 10000 resamples of 172 frames):
- recall@10 diff +0.029 [95% CI +0.000, +0.059], P(>0)=0.96
- precision@10 diff +0.023 [95% CI -0.013, +0.062], P(>0)=0.92
- frames hit only by tracker: 6, only by baseline: 1

