> Review of 2026-11-08, describing `main` @ 26d2506. Part of the [2026-11 review](index.md); prioritised actions are in the [TODO list](../../todo.md).

# footy-track architecture (origin/main)

Drawn from `architecture-review.md`. Legend: **grey** = data on disk or S3, **blue** = code in the repo, **orange, dashed** = step that is not in git, **red** = output that nothing reads, **purple** = the new `/ball_check` path.

## 1. End-to-end data flow

```mermaid
flowchart TB
  classDef code fill:#dbeafe,stroke:#2563eb,color:#111
  classDef data fill:#e5e7eb,stroke:#6b7280,color:#111
  classDef missing fill:#ffedd5,stroke:#ea580c,stroke-dasharray:5 3,color:#111
  classDef dead fill:#fee2e2,stroke:#dc2626,color:#111
  classDef bc fill:#ede9fe,stroke:#7c3aed,color:#111

  VID[("Full match video")]:::data

  subgraph INGEST["Ingest"]
    SEG["split_broadcast_segments<br/>yolo11n-cls classifier"]:::code
    CLIPS[("Shot clips .mp4<br/>3 name aliases")]:::data
  end

  subgraph MACHINE["Machine labels"]
    RTD["RT-DETR-L batch inference<br/>imgsz 1280"]:::missing
    ML[("machine_labels/*.jsonl<br/>2.77M boxes, S3")]:::data
    FILT["ball row filter"]:::missing
    CAND[("ball_candidates/*.jsonl<br/>357k rows")]:::data
    EMB["embed_ball_candidates"]:::code
    TRK["track_detections<br/>LapTracker"]:::code
    TL[("tracklets/*.jsonl")]:::data
    LINK["score_links"]:::code
    LP[("link_proposals/")]:::dead
  end

  subgraph LABEL["Labeller server (FastAPI)"]
    LAB["/labeller<br/>VitTrack propagation"]:::code
    REV["/object_review"]:::code
    BC["/ball_check<br/>mobile verdict page"]:::bc
    IDR["/identity"]:::code
  end

  GT[("ball_gt_marks/&lt;clip&gt;.jsonl<br/>45 files, 22.6k rows")]:::data
  VERD[("ball_checks/&lt;clip&gt;.jsonl<br/>364 verdicts")]:::dead
  IDL[("identity_labels/")]:::dead

  subgraph STORE["Feature store (DuckDB)"]
    ING["ingest_gt<br/>top-level files only"]:::code
    RFI["importers: Roboflow"]:::code
    DB[("detection table<br/>26k rows, stale since Aug 4<br/>canonical view")]:::data
  end

  EXP["export_training_dataset<br/>split by clip name"]:::code
  YOLO[("YOLO dataset")]:::data
  TRAIN["train_object_detector<br/>defaults: imgsz 640, mps"]:::code
  W[("best.pt / W&amp;B")]:::data
  EVAL["ball_eval + run_bakeoff"]:::code
  SERVE["Serving / output<br/>scaffolds, not wired"]:::dead

  style INGEST fill:none,stroke:#9ca3af
  style MACHINE fill:none,stroke:#9ca3af
  style LABEL fill:none,stroke:#9ca3af
  style STORE fill:none,stroke:#9ca3af

  VID --> SEG --> CLIPS
  CLIPS --> RTD --> ML
  ML --> FILT --> CAND --> EMB
  ML --> TRK --> TL
  TL --> LINK --> LP
  TL --> IDR --> IDL

  CLIPS --> LAB
  LAB --> GT
  REV <--> GT
  CAND --> BC
  GT --> BC
  EMB --> BC
  BC -- "verdicts, append-only" --> VERD
  BC -- "dragged box, tag labeller" --> GT

  GT --> ING --> DB
  RF[("Roboflow YOLO sets")]:::data --> RFI --> DB
  DB -- "WHERE canonical" --> EXP --> YOLO --> TRAIN --> W
  W --> EVAL
  W -.-> SERVE
```

## 2. Where the pipeline silently goes wrong

```mermaid
flowchart TB
  classDef bad fill:#fee2e2,stroke:#dc2626,color:#111
  classDef ok fill:#e5e7eb,stroke:#6b7280,color:#111

  V["ball_check verdict:<br/>not_ball"]:::ok --> VF[("ball_checks/*.jsonl")]:::ok
  VF -. "ingest_gt never globs ball_checks/" .-> X1["verdict ignored"]:::bad
  SC[("sidecar still holds the box")]:::ok --> EXPQ["export filters on canonical only"]:::ok --> POS["exported as a POSITIVE ball"]:::bad

  S1["Labeller session edit"]:::ok --> STRIP["session.py keeps only label, model<br/>drops human_checked"]:::bad --> T3["every tier-2 box in the clip<br/>reverts to tier 3"]:::bad

  RI["Re-ingest after a correction"]:::ok --> GH["old vittrack row stays canonical<br/>upsert never deletes"]:::bad --> BOTH["old and corrected box<br/>both exported"]:::bad

  A1["mancity_seg050"]:::ok --- A2["arsenal_mancity_seg050"]:::ok --- A3["arsenal_mancity_20250925_seg050"]:::ok
  A3 --> LEAK["split and eval exclusion use exact names<br/>eval footage can leak into training"]:::bad
```

## 3. Findings behind the diagram

| # | Seam | Where |
|---|---|---|
| 1 | Labeller session strips `human_checked` | `session.py:156-161`, `309-317` |
| 2 | `not_ball` verdicts never reach the store; the box exports as a positive | `ingest_gt.py:279`, `export_training_dataset.py:91-128` |
| 3 | Clip identity is not canonical (3 aliases) | `ball_check.py:709-728`, `export_training_dataset.py:78-80, 362-368` |
| 4 | Re-ingest leaves ghost rows | `store.py:197-220`, `schema.py:40-48` |
| 5 | Undoing a corrected card leaves its box; a session flush can overwrite `ball_check` writes | `ball_check.html:351-365`, `session.py:252-301` |
| 6 | Trainer defaults are traps (`--imgsz 640`, `--device mps`) | `train_object_detector.py` |
| 7 | Producers not in git: RT-DETR batch inference, ball row filter, RT-DETR training, clip-alias symlinks | see review §1 |

Full text: [architecture-review.md](architecture-review.md)
