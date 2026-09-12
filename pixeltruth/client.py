import os
from dataclasses import dataclass
from typing import Union
from PIL import Image

@dataclass
class PredictionResult:
    label: str
    confidence: float

    @property
    def is_ai(self) -> bool:
        return self.label == 'AI-Generated'

    @property
    def is_real(self) -> bool:
        return self.label == 'Real-Image' or self.label == 'Real Image'

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'is_ai': self.is_ai,
            'is_real': self.is_real,
        }

class PixelTruthDetector:
    """Programmatic interface for PixelTruth dual-transformer ensemble detection."""

    def __init__(self):
        pass

    def predict(self, image_input: Union[str, Image.Image]) -> PredictionResult:
        """
        Classify an image given either a file path or a PIL Image instance.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError("image_input must be a file path string or a PIL.Image instance")

        from model.detector import predict
        label, confidence = predict(image)
        return PredictionResult(label=label, confidence=confidence)
