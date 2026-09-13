import os
import io
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
from PIL import Image

@dataclass
class PredictionResult:
    label: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    frequency_metrics: Optional[Dict[str, Any]] = None

    @property
    def is_ai(self) -> bool:
        return self.label == 'AI-Generated'

    @property
    def is_real(self) -> bool:
        return self.label == 'Real-Image' or self.label == 'Real Image'

    def to_dict(self) -> dict:
        data = {
            'label': self.label,
            'confidence': self.confidence,
            'is_ai': self.is_ai,
            'is_real': self.is_real,
        }
        if self.metadata is not None:
            data['metadata'] = self.metadata
        if self.frequency_metrics is not None:
            data['frequency_metrics'] = self.frequency_metrics
        return data

class PixelTruthDetector:
    """Programmatic interface for PixelTruth dual-transformer ensemble detection."""

    def __init__(self):
        pass

    def predict(
        self,
        image_input: Union[str, Image.Image],
        analyze_forensics: bool = False
    ) -> PredictionResult:
        """
        Classify an image given either a file path or a PIL Image instance.
        Optionally extracts EXIF/PNG metadata signatures and 2D FFT spectral metrics.
        """
        raw_bytes = None
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            with open(image_input, 'rb') as f:
                raw_bytes = f.read()
            image = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise TypeError("image_input must be a file path string or a PIL.Image instance")

        meta = None
        freq = None
        if analyze_forensics:
            from backend.frequency import analyze_frequency_domain
            from backend.exif_inspector import inspect_image_metadata
            freq = analyze_frequency_domain(image)
            if raw_bytes:
                meta = inspect_image_metadata(raw_bytes)

        from model.detector import predict
        label, confidence = predict(image)
        return PredictionResult(
            label=label,
            confidence=confidence,
            metadata=meta,
            frequency_metrics=freq
        )
