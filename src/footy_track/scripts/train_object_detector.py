"""
Script to train a YOLO object detection model.

This script downloads a dataset from Roboflow, then trains a YOLOv8 model on it.
It can be configured with command-line arguments for the model, dataset version, and number of layers to freeze.
"""

import argparse
import os
from datetime import datetime

import torch
from roboflow import Roboflow
from ultralytics import YOLO


def apply_mps_assigner_workaround() -> None:
    """Run ultralytics' TaskAlignedAssigner on CPU when training on MPS.

    torch-on-MPS has an advanced-indexing bug that crashes the assigner
    stochastically (RuntimeError: shape mismatch in get_box_metrics). The
    assigner is no_grad bookkeeping, so computing it on CPU is exact; the
    model forward/backward stays on MPS.
    """
    from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: PLC0415

    orig = TaskAlignedAssigner.forward

    def cpu_safe(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        dev = pd_scores.device
        if dev.type == "mps":
            out = orig(
                self,
                pd_scores.cpu(),
                pd_bboxes.cpu(),
                anc_points.cpu(),
                gt_labels.cpu(),
                gt_bboxes.cpu(),
                mask_gt.cpu(),
            )
            return tuple(t.to(dev) for t in out)
        return orig(
            self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

    TaskAlignedAssigner.forward = cpu_safe


def train_detector(
    model_name: str,
    dataset_version: int,
    freeze_layers: int,
    epochs: int,
    name_prepend: str = "",
    local_dataset: str | None = None,
    imgsz: int = 640,
    batch: int = -1,
    amp: bool = True,
    device: str = "mps",
):
    """
    Downloads a dataset from Roboflow (or uses a local path) and trains a YOLO detector.

    Args:
        model_name (str): The name of the YOLO model to use (e.g., 'yolo11n').
        dataset_version (int): The version of the dataset on Roboflow.
        freeze_layers (int): The number of layers to freeze during training.
        epochs (int): The number of epochs to train for.
        name_prepend (str): A string to prepend to the run name.
        local_dataset (str | None): If set, use this local YOLOv8-format dataset
            directory (containing ``data.yaml``) instead of downloading from Roboflow.
    """
    if local_dataset:
        dataset_location = local_dataset
        print(f"Using local dataset: {dataset_location}")
    else:
        # Roboflow and dataset setup
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
        project = rf.workspace("egroeg121").project("footy-track-detection")
        version = project.version(dataset_version)

        data_root = os.environ.get("DATA_ROOT", "data")
        dataset_location = os.path.join(
            data_root, f"detection_dataset/roboflow_dataset_{dataset_version}"
        )
        dataset = version.download(model_format="yolov8", location=dataset_location)
        dataset_location = dataset.location

    print(f"Dataset location: {dataset_location}")

    # Check for GPU availability
    print(
        f"MPS available: {torch.backends.mps.is_available()}, MPS built: {torch.backends.mps.is_built()}"
    )

    # Set up training run
    run_name = f"{datetime.now():%Y-%m-%d_%H-%M}_model_name={str(model_name)}_dataset_version={dataset_version}_epochs={epochs}_freeze_layers={freeze_layers}_imgsz={imgsz}"
    if name_prepend:
        run_name = f"{name_prepend}_{run_name}"
    model_path = f"{model_name}.pt"

    if device == "mps":
        apply_mps_assigner_workaround()

    print(f"Starting training for run: {run_name}")
    model = YOLO(model_path)

    # Train the model
    results = model.train(
        data=os.path.join(dataset_location, "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        freeze=freeze_layers,
        batch=batch,
        amp=amp,
        cache=True,
        augment=True,
        plots=True,
        device=device,
        project="footy_scan_detection",
        name=run_name,
    )
    print("Training complete.")
    print(f"Model and results saved to: {results.save_dir}")
    return results.results_dict["metrics/mAP50-95(B)"], os.path.join(
        results.save_dir, "weights", "best.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO object detection model.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="The YOLO model to use (e.g., yolo11n, yolo11m). '.pt' will be appended.",
    )
    parser.add_argument(
        "--dataset-version",
        type=int,
        default=3,
        help="The version of the dataset to download from Roboflow.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=9,
        help="Number of layers to freeze during training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train for.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (px). Use 1280 for small-object (ball) work.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 = auto). On MPS use an explicit small batch, e.g. 4-8.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (recommended on MPS).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Training device (mps, cpu, 0 for CUDA).",
    )
    parser.add_argument(
        "--local-dataset",
        type=str,
        default=None,
        help="Path to a local YOLOv8-format dataset directory (containing data.yaml). "
        "If set, skips the Roboflow download.",
    )

    args = parser.parse_args()

    train_detector(
        model_name=args.model,
        dataset_version=args.dataset_version,
        imgsz=args.imgsz,
        batch=args.batch,
        amp=not args.no_amp,
        device=args.device,
        freeze_layers=args.freeze,
        epochs=args.epochs,
        local_dataset=args.local_dataset,
    )
