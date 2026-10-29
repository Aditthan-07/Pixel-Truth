# PixelTruth — AI Image Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![PixelTruth CI](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/ci.yml)
[![Lint & Code Quality](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/lint.yml/badge.svg)](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/lint.yml)
[![Code Coverage](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/coverage.yml/badge.svg)](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/coverage.yml)
[![OpenAPI 3.0](https://img.shields.io/badge/OpenAPI-3.0-6BA539?style=for-the-badge&logo=openapiinitiative&logoColor=white)](docs/openapi.yaml)

**A Deep Learning web app that detects whether an image is a real photograph or AI-generated — with visual explainability via Grad-CAM.**

</div>

---

## Overview

PixelTruth uses an ensemble of two pre-trained Vision Transformer (ViT) models from Hugging Face to classify uploaded images as **Real** or **AI Generated**, with a confidence score and a **Grad-CAM heatmap** overlay showing which regions of the image influenced the prediction.

---

## Features

- **Dual-Model Ensemble** — Combines `umm-maybe/AI-image-detector` (ViT) and `Organika/sdxl-detector` (Swin) for reliable cross-architecture predictions
- **Grad-CAM Explainability** — Visual heatmap overlay with percentile contrast normalization and configurable colormaps (`thermal`, `coolwarm`, `fire`)
- **Metadata & EXIF Forensics** — Inspects camera hardware tags (Make/Model) and detects generative prompt signatures in PNG/JPEG metadata
- **2D Fourier FFT Analysis** — Analyzes radial power spectrums to spot generative upsampling grid anomalies
- **Terminal CLI & Batch Mode** — Process individual images or entire directories with JSON exports
- **Python SDK Package** — Programmatic `PixelTruthDetector` class for direct script and notebook integration
- **Modern Web UI** — Drag-and-drop interface with clipboard paste (`Ctrl+V`) and 1-click JSON report export
- **REST API Backend** — Flask API with CORS, security headers, and OpenAPI 3.0 specification

---

## Demo

```
Upload an image → Get verdict (Real / AI Generated) → View Grad-CAM heatmap
```

---

## Project Structure

```
Pixel-Truth/
├── model/
│   ├── __init__.py
│   └── detector.py         # HuggingFace model loader & ensemble inference
├── backend/
│   ├── __init__.py
│   ├── app.py              # Flask REST API & routes
│   └── gradcam.py          # Grad-CAM heatmap generation
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Styling
│   └── script.js           # Upload & result handling
├── static/
│   └── uploads/            # Temporary image storage (git-ignored)
├── docs/
│   ├── GUIDE.md            # Usage guide
│   └── PAPER.md            # Research paper document
├── PAPER.md                # Research paper specification & methodology
├── requirements.txt        # Python dependencies
└── run.py                  # Entry point
```

---

## 📄 Research Paper
Read the complete research paper detailing the dual-transformer architecture, vote-and-average fusion algorithm, and saliency explainability:
👉 **[Read Paper (PAPER.md)](PAPER.md)**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| ML Models | Hugging Face Transformers (ViT) |
| Deep Learning | PyTorch, TorchVision |
| Explainability | Grad-CAM |
| Image Processing | Pillow, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## 🚀 Quick Start

1. **Clone the repository:**
   `ash
   git clone https://github.com/Aditthan-07/Pixel-Truth.git
   cd Pixel-Truth
   `

2. **Install dependencies:**
   `ash
   pip install -r requirements.txt
   `

3. **Start the application:**
   `ash
   python run.py
   `

   Open **http://localhost:5000** in your browser.

---

## 👤 Author

Created by [Aditthan](https://github.com/Aditthan-07).
