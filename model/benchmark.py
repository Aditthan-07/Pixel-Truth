"""
Model Benchmarking & Profiling Tool for PixelTruth
Measures inference latency, memory consumption, and transformer pass times.
"""

import time
import sys
import os
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def benchmark_inference(num_runs=5):
    print("=" * 60)
    print(" PixelTruth Dual-Transformer Inference Profiler")
    print("=" * 60)

    try:
        import torch
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        print(f"[*] PyTorch Version : {torch.__version__}")
        print(f"[*] Active Device   : {device}")
    except ImportError:
        print("[!] PyTorch not found in current environment")
        return

    # Create synthetic test image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    print(f"[*] Test Resolution : {dummy_img.size[0]}x{dummy_img.size[1]} RGB")

    print("\n[+] Loading models...")
    t0 = time.time()
    try:
        from model.detector import load_models, predict
        load_models()
        load_time = time.time() - t0
        print(f"[+] Model Loading Time: {load_time:.2f} seconds")
    except Exception as e:
        print(f"[!] Failed to load models: {e}")
        return

    latencies = []
    print(f"\n[*] Running {num_runs} inference cycles...")
    for i in range(num_runs):
        start = time.time()
        label, conf = predict(dummy_img)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} ms -> {label} ({conf:.1f}%)")

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    print("-" * 60)
    print("[*] Benchmark Results:")
    print(f"    Average Latency : {avg_latency:.2f} ms +/- {std_latency:.2f} ms")
    print(f"    Min Latency     : {np.min(latencies):.2f} ms")
    print(f"    Max Latency     : {np.max(latencies):.2f} ms")
    print(f"    Throughput      : {1000 / avg_latency:.2f} images/sec")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_inference()
