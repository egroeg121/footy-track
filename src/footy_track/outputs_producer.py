import logging
from abc import ABC
from pathlib import Path

import cv2
from ultralytics.engine.results import Results as UltralyticsResults

logger = logging.getLogger(__name__)


class OutputProducer(ABC):
    def __init__(self, save_path: Path | str):
        self.save_path = Path(save_path)

    def run(self):
        pass


class UltralyticsResultsToVideo(OutputProducer):
    def __init__(self, input_video_path, save_path: Path | str):
        super().__init__(save_path)
        self.input_video_path = Path(input_video_path)

    def run(self, results: list[UltralyticsResults], *args, **kwargs):
        return self.process_results(results, *args, **kwargs)

    def process_results(self, results: list[UltralyticsResults]):
        """Save a list of UltralyticsResults onto a video"""
        # Resolve and ensure the output directory exists
        save_path_abs = self.save_path.resolve()
        save_path_abs.parent.mkdir(parents=True, exist_ok=True)

        # Open source video to read metadata for writer
        cap = cv2.VideoCapture(str(self.input_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {self.input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(str(save_path_abs), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter for path: {save_path_abs}")

        for result in results:
            frame = result.plot()  # annotated BGR numpy array
            # Ensure frame size matches writer size
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)

        writer.release()
        cap.release()

        # Simple trace to help locate the output
        logger.info(f"Video written to: {save_path_abs}")
        return str(save_path_abs)


class UltralyticsResultsToJson(OutputProducer):
    def __init__(self, save_path: Path | str):
        super().__init__(save_path)

    def run(self, results: list[UltralyticsResults], *args, **kwargs):
        return self.process_results(results, *args, **kwargs)

    def process_results(self, results: list[UltralyticsResults]) -> list[dict]:
        """Save a list of UltralyticsResults onto a JSON file"""
        import json

        all_results = [result.dict() for result in results]

        save_path_abs = self.save_path.resolve()
        save_path_abs.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path_abs, "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"JSON written to: {save_path_abs}")
        return all_results
