from PIL import Image, ExifTags
import io
from typing import Dict, Any

AI_SIGNATURE_KEYWORDS = [
    'stable diffusion', 'midjourney', 'dall-e', 'dalle', 'novelai',
    'comfyui', 'fooocus', 'civitai', 'prompt:', 'negative prompt:'
]

def inspect_image_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts and analyzes EXIF and PNG text chunks for camera or generative signatures.
    """
    result = {
        'has_exif': False,
        'has_camera_data': False,
        'camera_make': None,
        'camera_model': None,
        'ai_metadata_detected': False,
        'ai_software_tag': None,
        'raw_text_chunks': {}
    }

    try:
        stream = io.BytesIO(image_bytes)
        with Image.open(stream) as img:
            # 1. Check PNG text metadata chunks (tEXt / iTXt)
            if hasattr(img, 'text') and isinstance(img.text, dict):
                for k, v in img.text.items():
                    val_str = str(v)
                    result['raw_text_chunks'][k] = val_str[:200]
                    val_lower = val_str.lower()
                    if any(sig in val_lower for sig in AI_SIGNATURE_KEYWORDS):
                        result['ai_metadata_detected'] = True
                        result['ai_software_tag'] = k

            # 2. Check EXIF tags
            exif_data = img.getexif() if hasattr(img, 'getexif') else None
            if exif_data:
                result['has_exif'] = len(exif_data) > 0
                for tag_id, val in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    tag_str = str(val).strip()
                    tag_lower = tag_str.lower()

                    if tag_name == 'Make':
                        result['camera_make'] = tag_str
                        result['has_camera_data'] = True
                    elif tag_name == 'Model':
                        result['camera_model'] = tag_str
                        result['has_camera_data'] = True
                    elif tag_name == 'Software':
                        if any(sig in tag_lower for sig in AI_SIGNATURE_KEYWORDS):
                            result['ai_metadata_detected'] = True
                            result['ai_software_tag'] = tag_str

    except Exception:
        pass

    return result
