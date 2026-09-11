#!/usr/bin/env python3
"""
PixelTruth CLI Tool
Run AI image detection and saliency visualization directly from your terminal.
"""

import argparse
import sys
import os
import json
import base64
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_cli():
    parser = argparse.ArgumentParser(
        description="PixelTruth CLI - Dual-Transformer AI Image Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py path/to/image.jpg
  python cli.py path/to/image.png --output overlay.png
  python cli.py path/to/image.jpg --json
        """
    )
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--output", "-o", help="Optional path to save the saliency overlay image", default=None)
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed model confidence info")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        if args.json:
            print(json.dumps({"error": f"File not found: {args.image}"}), file=sys.stderr)
        else:
            print(f"[ERROR] File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        image = Image.open(args.image).convert('RGB')
    except Exception as e:
        if args.json:
            print(json.dumps({"error": f"Failed to load image: {e}"}), file=sys.stderr)
        else:
            print(f"[ERROR] Failed to load image: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"[*] Analyzing image: {os.path.basename(args.image)} ...")

    try:
        from model.detector import predict, _models
        label, confidence = predict(image)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": f"Inference failed: {e}"}), file=sys.stderr)
        else:
            print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    overlay_saved = None
    if args.output:
        try:
            from backend.gradcam import generate_gradcam
            model_pipeline = next(iter(_models.values())) if _models else None
            if model_pipeline:
                overlay_b64 = generate_gradcam(image, model_pipeline)
                if overlay_b64.startswith("data:image"):
                    overlay_b64 = overlay_b64.split(",", 1)[1]
                overlay_bytes = base64.b64decode(overlay_b64)
                with open(args.output, "wb") as f:
                    f.write(overlay_bytes)
                overlay_saved = args.output
                if not args.json:
                    print(f"[+] Saliency overlay saved to: {args.output}")
        except Exception as e:
            if not args.json:
                print(f"[WARNING] Failed to save overlay: {e}")

    if args.json:
        result = {
            "image": args.image,
            "verdict": label,
            "confidence": confidence,
            "overlay": overlay_saved
        }
        print(json.dumps(result, indent=2))
    else:
        badge = "[AI GENERATED]" if label == "AI-Generated" else "[REAL IMAGE]"
        print("=" * 45)
        print(f"  Verdict    : {badge} {label}")
        print(f"  Confidence : {confidence:.2f}%")
        print("=" * 45)

if __name__ == "__main__":
    run_cli()
