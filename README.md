# PixelTruth — AI Image Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![PixelTruth CI](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditthan-07/Pixel-Truth/actions/workflows/ci.yml)

**A Deep Learning web app that detects whether an image is a real photograph or AI-generated — with visual explainability via Grad-CAM.**

</div>

---

## Overview

PixelTruth uses an ensemble of two pre-trained Vision Transformer (ViT) models from Hugging Face to classify uploaded images as **Real** or **AI Generated**, with a confidence score and a **Grad-CAM heatmap** overlay showing which regions of the image influenced the prediction.

---

## Features

- **Dual-Model Ensemble** — Combines `umm-maybe/AI-image-detector` and `Organika/sdxl-detector` for more reliable predictions via majority voting
- **Grad-CAM Explainability** — Visual heatmap overlay showing which parts of the image triggered the AI detection
- **Drag & Drop UI** — Clean, modern frontend supporting PNG, JPG, WEBP, and GIF
- **Terminal CLI & JSON Mode** — Standalone CLI supporting formatted console output, overlay export, and machine-readable JSON
- **REST API Backend** — Flask API with CORS support, easily extendable
- **Zero Setup Models** — Models auto-download from Hugging Face on first run

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

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Aditthan-07/Pixel-Truth.git
cd Pixel-Truth
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the App

```bash
python run.py
```

Then open **http://localhost:5000** in your browser.

> **Note:** On first run, the two models (~300MB total) will download automatically from Hugging Face. This is a one-time process.

### Command-Line Interface (CLI)

PixelTruth also includes a standalone CLI for batch scripts and headless workflows:

```bash
# Analyze an image in the terminal
python cli.py path/to/image.jpg

# Save the visual Grad-CAM heatmap overlay to disk
python cli.py path/to/image.png --output overlay.png

# Output machine-readable JSON for integration into pipelines
python cli.py path/to/image.jpg --json
```

### Running with Docker

You can also run PixelTruth in an isolated container using Docker Compose:

```bash
docker compose up --build
```

Then visit **http://localhost:5000**.

---

## How It Works

### Ensemble Prediction

Two ViT-based models independently classify the image. Their confidence scores are averaged and a majority vote determines the final verdict:

```
Image → Model 1 (umm-maybe/AI-image-detector)  ─┐
                                                  ├─ Majority Vote → Label + Confidence
Image → Model 2 (Organika/sdxl-detector)        ─┘
```

### Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) computes gradients of the predicted class with respect to the input pixel values. These gradients are used to generate a heatmap overlay — **red regions** indicate areas that most influenced the AI detection.

---

## API Reference

### `POST /predict`

Accepts a multipart image upload and returns the prediction.

**Request**

```
Content-Type: multipart/form-data
Body: image=<file>
```

**Response**

```json
{
  "label": "AI Generated",
  "confidence": 94.73,
  "gradcam_image": "<base64-encoded PNG>",
  "original_image": "<base64-encoded PNG>"
}
```

**Supported formats:** PNG, JPG, JPEG, WEBP, GIF

---

## Deep Learning Concepts Demonstrated

- Transfer Learning with pre-trained Vision Transformers (ViT)
- Binary Image Classification (Real vs AI-Generated)
- Ensemble Learning via multi-model voting
- Grad-CAM for model interpretability
- REST API deployment of ML models

---

## Models Used

| Model | Source | Description |
|---|---|---|
| `umm-maybe/AI-image-detector` | Hugging Face | ViT fine-tuned on real vs AI-generated images |
| `Organika/sdxl-detector` | Hugging Face | Specialized detector for SDXL-generated images |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Built by [Aditthan](https://github.com/Aditthan-07) · Powered by Deep Learning & Hugging Face 🤗

</div>
