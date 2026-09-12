from PIL import Image
import io

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}

def is_allowed_extension(filename: str, allowed: set = None) -> bool:
    """Check if the provided filename has an allowed extension."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[-1].lower()
    target_allowed = allowed if allowed is not None else ALLOWED_EXTENSIONS
    return ext in target_allowed

def sanitize_and_prepare_image(image_bytes: bytes, max_dim: int = 4096) -> Image.Image:
    """
    Validates, strips unsafe metadata/EXIF, and normalizes image to RGB.
    Safely downsizes if dimensions exceed max_dim to avoid memory exhaustion.
    """
    stream = io.BytesIO(image_bytes)
    with Image.open(stream) as img:
        img.verify()
    
    stream.seek(0)
    image = Image.open(stream).convert('RGB')
    
    # Downscale if excessively large
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resample = getattr(Image, 'Resampling', Image).LANCZOS
        image = image.resize(new_size, resample)
        
    return image
