import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io

def generate_gradcam(image: Image.Image, model_pipeline, alpha: int = 140) -> str:
    """
    Computes input-gradient saliency heatmap superimposed onto the original image.
    Uses percentile contrast normalization and bicubic upsampling for clean visualization.
    """
    try:
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

        colormap = np.zeros((*heatmap_resized.shape, 3), dtype=np.uint8)
        colormap[:, :, 0] = heatmap_resized
        colormap[:, :, 1] = (255 - heatmap_resized)
        colormap[:, :, 2] = 80

        overlay = Image.fromarray(colormap).convert("RGBA")
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
