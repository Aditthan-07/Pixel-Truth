import numpy as np
from PIL import Image
import base64
import io

def apply_colormap(heatmap: np.ndarray, style: str = 'thermal') -> np.ndarray:
    """Applies a 3-channel RGB colormap to a 2D normalized heatmap [0, 255]."""
    h = heatmap.astype(np.float32) / 255.0
    cmap = np.zeros((*heatmap.shape, 3), dtype=np.uint8)

    if style == 'coolwarm':
        cmap[:, :, 0] = (h * 255).astype(np.uint8)
        cmap[:, :, 1] = ((1.0 - np.abs(h - 0.5) * 2) * 180).astype(np.uint8)
        cmap[:, :, 2] = ((1.0 - h) * 255).astype(np.uint8)
    elif style == 'fire':
        cmap[:, :, 0] = np.clip(h * 2.0 * 255, 0, 255).astype(np.uint8)
        cmap[:, :, 1] = np.clip((h - 0.4) * 2.5 * 255, 0, 255).astype(np.uint8)
        cmap[:, :, 2] = 20
    else:  # 'thermal' (default)
        cmap[:, :, 0] = heatmap
        cmap[:, :, 1] = (255 - heatmap)
        cmap[:, :, 2] = 80

    return cmap

def generate_gradcam(image: Image.Image, model_pipeline, alpha: int = 140, colormap: str = 'thermal') -> str:
    """
    Computes input-gradient saliency heatmap superimposed onto the original image.
    Uses percentile contrast normalization, bicubic upsampling, and configurable colormaps.
    """
    try:
        import torch
        import torch.nn.functional as F
        model = model_pipeline.model
        processor = model_pipeline.feature_extractor if hasattr(model_pipeline, 'feature_extractor') else model_pipeline.image_processor

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_values.requires_grad_(True)

        outputs = model(pixel_values)
        logits = outputs.logits
        top_class = logits.argmax(dim=1)
        logits[0, top_class].backward()

        gradients = pixel_values.grad[0]
        pooled = gradients.mean(dim=[1, 2])
        weighted = (pixel_values[0] * pooled[:, None, None]).sum(dim=0)
        heatmap = F.relu(weighted).detach().cpu().numpy()

        if heatmap.max() > 0:
            # Robust percentile clipping for enhanced contrast
            p99 = np.percentile(heatmap, 99)
            if p99 > 0:
                heatmap = np.clip(heatmap / p99, 0.0, 1.0)
            else:
                heatmap = heatmap / heatmap.max()

        # Bicubic resampling produces smoother forensic overlays
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        resample_filter = getattr(Image, 'Resampling', Image).BICUBIC
        heatmap_resized = np.array(heatmap_img.resize(image.size, resample_filter))

        colored = apply_colormap(heatmap_resized, style=colormap)
        overlay = Image.fromarray(colored).convert("RGBA")
        overlay.putalpha(max(0, min(255, alpha)))
        base = image.convert("RGBA")
        combined = Image.alpha_composite(base, overlay).convert("RGB")

        buffer = io.BytesIO()
        combined.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
