"""Train one arm of the temporal-detector ablation with stock Ultralytics (identical recipe, only the input differs).

    arm A  3 channels: frame t only (lossless PNG copy of the TIFF's B,G,R), COCO-pretrained, stock stem
    arm B  5 channels: [gray(t-2), B, G, R, gray(t+2)] TIFFs, COCO-pretrained, RGB stem filters copied to
           input channels 3/2/1, temporal channels zero-initialised (step 0 == arm A)

Identical in both arms: model, imgsz, batch, epochs, seed, optimiser (Ultralytics auto), mosaic/scale/translate/flip
defaults, HSV colour augmentation OFF (Ultralytics skips HSV for non-3-channel images, so it is disabled in both),
Albumentations OFF (patched), plots OFF (the 5-channel plot code path is not exercised), val every epoch.

Resumable: if weights/last.pt exists and the run is not finished, resumes it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import BALL_CLS, write_data_yaml
from multichannel import make_inflated_checkpoint, patch_ultralytics_multichannel


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--data-root", type=Path, required=True, help="dir containing k2/ and k2_rgb/")
    ap.add_argument("--weights-dir", type=Path, required=True, help="dir with <model>.pt")
    ap.add_argument("--pseudo", action="store_true", help="round 2: add <data-root>/pseudo (5ch) or pseudo_rgb (3ch) to the train set")
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--name", required=True)
    return ap.parse_args()


def main():
    a = build_args()
    patch_ultralytics_multichannel()
    from ultralytics import YOLO

    run = a.project / a.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "train_done.json").exists():
        print("already done:", run)
        return
    channels = 3 if a.arm == "A" else 5
    root = a.data_root / ("k2_rgb" if a.arm == "A" else "k2")
    extra = [str(a.data_root / ("pseudo_rgb" if a.arm == "A" else "pseudo") / "images" / "train")] if a.pseudo else None
    if extra:
        assert Path(extra[0]).is_dir(), extra
    data_yaml = write_data_yaml(root, run / "data_local.yaml", channels, "images", extra)
    stock = a.weights_dir / f"{a.model}.pt"
    if a.arm == "A":
        pretrained = stock
    else:
        pretrained = a.weights_dir / f"{a.model}_5ch.pt"
        if not pretrained.exists():
            print("inflating stem:", make_inflated_checkpoint(stock, pretrained))

    cfg = dict(
        data=str(data_yaml), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, seed=a.seed,
        device=0, amp=True, plots=False, val=True, cache=False, exist_ok=True, patience=10_000,
        deterministic=False, fraction=a.fraction, project=str(a.project), name=a.name,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    )
    (run / "train_config.json").write_text(json.dumps({"arm": a.arm, "model": a.model, "channels": channels,
                                                       "pretrained": str(pretrained), **cfg}, indent=1))
    t0 = time.time()
    curve = run / "ball_curve.jsonl"

    def on_epoch_end(trainer):
        box = trainer.validator.metrics.box
        idx = list(box.ap_class_index)
        row = {"epoch": trainer.epoch + 1, "elapsed_s": round(time.time() - t0, 1),
               "loss": [round(float(x), 4) for x in trainer.tloss],
               "map50_all": float(box.map50), "map_all": float(box.map)}
        if BALL_CLS in idx:
            i = idx.index(BALL_CLS)
            row.update(ball_ap50=float(box.ap50[i]), ball_ap=float(box.all_ap[i].mean()),
                       ball_p=float(box.p[i]), ball_r=float(box.r[i]))
        with open(curve, "a") as f:
            f.write(json.dumps(row) + "\n")

    last = run / "weights" / "last.pt"
    resume = last.exists()
    model = YOLO(str(last if resume else pretrained))
    model.add_callback("on_fit_epoch_end", on_epoch_end)
    if resume:
        print("RESUMING", last)
        model.train(resume=True)
    else:
        model.train(**cfg)
    (run / "train_done.json").write_text(json.dumps({"seconds_this_session": round(time.time() - t0, 1),
                                                      "resumed": resume}))


if __name__ == "__main__":
    main()
