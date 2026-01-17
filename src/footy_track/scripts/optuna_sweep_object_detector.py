"""
Script to optimize YOLO object detection model hyperparameters using Optuna.

This script uses the train_detector function and Optuna to find the best combination
of model, epochs, and frozen layers for object detection.
"""

import argparse

import optuna
from ultralytics import YOLO

from footy_track.scripts.train_object_detector import train_detector


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function to train a model with a given set of hyperparameters.

    Args:
        trial (optuna.Trial): An Optuna trial object.

    Returns:
        float: The metric to optimize (mAP50-95).
    """
    # Define the hyperparameter search space
    model_name = trial.suggest_categorical(
        "model_name", ["yolo11n", "yolo11s", "yolo11l"]
    )
    epochs = trial.suggest_int("epochs", 10, 5000, log=True)
    freeze_layers = trial.suggest_int("freeze_layers", 1, 20)

    # For the test run, we'll limit the epochs
    if trial.study.user_attrs.get("test_run", False):
        epochs = min(epochs, 3)

    print(f"Trial {trial.number}:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Freeze Layers: {freeze_layers}")

    # Run the training with the suggested hyperparameters
    # We prepend the trial number to the run name to keep track of Optuna runs
    try:
        mAP, best_model_path = train_detector(
            model_name=model_name,
            dataset_version=3,  # Keeping this constant for the study
            freeze_layers=freeze_layers,
            epochs=epochs,
            name_prepend=f"optuna_trial_{trial.number}",
        )

        # Manually run validation
        model = YOLO(best_model_path)
        metrics = model.val()
        mAP = metrics.box.map

        return mAP
    except RuntimeError as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a value that indicates failure, e.g., 0.0, so Optuna can continue.
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize YOLO object detection hyperparameters with Optuna."
    )
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of Optuna trials to run."
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a small test with a maximum of 3 epochs per trial.",
    )
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")

    if args.test_run:
        study.set_user_attr("test_run", True)

    study.optimize(objective, n_trials=args.n_trials)

    print("Optimization finished.")
    print(f"Best trial: {study.best_trial.value}")
    print("Best hyperparameters: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
