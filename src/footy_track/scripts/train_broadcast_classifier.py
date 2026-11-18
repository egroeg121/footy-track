"""Train the broadcast-view classifier (yolo11n-cls, Yes/No) on a local folder dataset, headless (CUDA).

    python -m footy_track.scripts.train_broadcast_classifier --data <dir with train/ valid/> --out <run dir>

Same recipe as the model in ``docs/training/notable_runs.md`` (aakevy06): yolo11n-cls, 50 epochs,
``freeze=9``, ``imgsz=224``, ``batch=16``, ``seed=0``, ``deterministic``, ``cache``. Differences: device
(CUDA instead of MPS), no Roboflow download, no W&B.

Validation is whatever ``<data>/valid`` holds (for ``build_broadcast_classifier_dataset``: in-match eval
review frames only). Ultralytics' ``best.pt`` is the best epoch *on that validation set*, i.e. selected on
eval data, so the model to use for scoring is ``last.pt`` (fixed schedule, no selection); both are copied
to ``<out>/``. ``<out>/train_config.json`` records the arguments, the library versions, the data manifest
(copied from ``<data>/manifest.json``) and the sha256 of the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n-cls.pt", help="pretrained start (a path, or a name ultralytics downloads)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--freeze", type=int, default=9)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    import torch
    import ultralytics
    from ultralytics import YOLO

    for sub in ("train", "valid"):
        if not (args.data / sub).is_dir():
            print(f"{args.data / sub} missing", file=sys.stderr)
            return 2
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "last.pt"
    if final.is_file() and (args.out / "train_config.json").is_file():
        print(f"{args.out} already trained; nothing to do")  # idempotent for the crash re-run
        return 0
    run_dir = args.out / "ultralytics"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    t0 = datetime.now(UTC)
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, freeze=args.freeze, batch=args.batch,
        seed=args.seed, deterministic=True, cache=True, augment=True, plots=True, workers=args.workers,
        device=args.device, project=str(args.out), name="ultralytics", exist_ok=True,
    )
    wdir = Path(results.save_dir) / "weights"
    shutil.copy2(wdir / "last.pt", args.out / "last.pt")
    shutil.copy2(wdir / "best.pt", args.out / "best.pt")
    manifest = None
    if (args.data / "manifest.json").is_file():
        manifest = json.loads((args.data / "manifest.json").read_text())
    cfg = {
        "started": t0.isoformat(),
        "finished": datetime.now(UTC).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scoring_weights": "last.pt",
        "why_last": "best.pt is the best epoch on the eval_in_match validation set (selected on eval data)",
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
        "pretrained": args.model,
        "val_metrics": {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))},
        "sha256": {"last.pt": sha256_file(args.out / "last.pt"), "best.pt": sha256_file(args.out / "best.pt")},
        "data_manifest": manifest,
    }
    (args.out / "train_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sha256": cfg["sha256"], "val_metrics": cfg["val_metrics"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
