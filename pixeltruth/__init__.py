"""
PixelTruth Python SDK
Detect synthetic and AI-generated imagery using a dual-transformer ensemble.
"""

from .client import PixelTruthDetector, PredictionResult

__all__ = ["PixelTruthDetector", "PredictionResult"]
__version__ = "1.0.0"
