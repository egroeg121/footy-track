"""Compute embeddings for video frames using a pretrained model."""

# embed_folder.py
import pathlib
from abc import ABC
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision.models import ConvNeXt_Base_Weights, ResNet50_Weights, convnext_base, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class FeatureExtractor(ABC):
    pass


class TorchvisionFeatureExtractor(FeatureExtractor):
    def __init__(self, weights: str, model: Callable[[str], torch.nn.Module]):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).eval().to(self.device)
        self.preprocess = weights.transforms()
        self.feat = create_feature_extractor(self.model, return_nodes={"avgpool": "feat"})

    def extract(self, img: Image.Image) -> np.ndarray:
        """Extract a 2048-dim feature vector from a PIL Image."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.feat(input_tensor)
        vec = out["feat"].squeeze().cpu().numpy()
        vec = vec / np.linalg.norm(vec)  # normalize
        return vec

    def extract_from_path(self, image_path: pathlib.Path) -> np.ndarray:
        """Extract a 2048-dim feature vector from an image file path."""
        img = Image.open(image_path)
        return self.extract(img)


class ResNet50FeatureExtractor(TorchvisionFeatureExtractor):
    def __init__(self):
        super().__init__(weights=ResNet50_Weights.DEFAULT, model=resnet50)


class ConvNeXtBaseFeatureExtractor(TorchvisionFeatureExtractor):
    def __init__(self):
        super().__init__(weights=ConvNeXt_Base_Weights.DEFAULT, model=convnext_base)
