"""The main entrypoint for the footy_track package/pipeline."""

import json

from ultralytics import YOLO

from footy_track.pipeline import Pipeline
from footy_track.processors import UltralyticsBaseProcessor
from footy_track.outputs_producer import UltralyticsResultsToVideo, UltralyticsResultsToJson

if __name__ == "__main__":
    input_video_path = "data/arsenal_mancity_example_video.mp4"
    
    pipeline = Pipeline(
        processor=UltralyticsBaseProcessor(),
        output_producers=[
            UltralyticsResultsToVideo(
                input_video_path=input_video_path,
                save_path=".debug/arsenal_mancity_track_predictions/video.mp4"
            ),
            # UltralyticsResultsToJson(
            #     save_path=".debug/arsenal_mancity_track_predictions/results.json"
            # )
        ]
    )

    pipeline.run(video_path=input_video_path, show=False)

    
        
