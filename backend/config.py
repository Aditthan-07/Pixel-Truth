import os

class Config:
    """PixelTruth Application Configuration"""
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 15 * 1024 * 1024))
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 50_000_000))
    ALLOWED_EXTENSIONS = set(
        ext.strip().lower() for ext in os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,webp,gif").split(",")
    )
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    GRADCAM_ALPHA = int(os.getenv("GRADCAM_ALPHA", 140))
