"""Create some processors"""



from abc import ABC
import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults

logger = logging.getLogger(__name__)


class Processor(ABC):

    def __init__(self):
        logger.info("Processor initialized")

    def startup_hook(self):
        """Hook for on pipeline initial run"""
        pass

    def step_hook(self, data):
        """Hook for on item that is processed"""
        return data

    def end_hook(self):
        """Hook for on pipeline end"""
        pass

    def run():
        """Run the processor"""
        self.startup_hook()
        self.step_hook()
        self.end_hook()
        return


class UltralyticsBaseProcessor(Processor):
    """Create a base processor for Ultralytics models"""
    
    def __init__(self, model_name: str = "yolo11n.pt"):
        super().__init__()


        self.model_name = model_name
        self.model = YOLO(model_name)


    def run(self, video_path: Path, show: bool = False, **kwargs) -> UltralyticsResults:
        """Run the model on a frame"""
        results = self.model.track(video_path, show=show, tracker="botsort.yaml", **kwargs)
        return results