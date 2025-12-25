from footy_track.outputs_producer import OutputProducer
from footy_track.processors import Processor


class Pipeline:
    def __init__(self, processor: Processor, output_producers: list[OutputProducer]):
        self.processor = processor
        self.output_producers = output_producers

    def run(self, video_path: str, show: bool = False) -> list:
        outputs = []

        results = self.processor.run(video_path=video_path, show=show)
        for output_producer in self.output_producers:
            outputs.append(output_producer.run(results))
        return outputs
