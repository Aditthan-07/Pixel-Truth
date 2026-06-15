# PixelTruth — AI Image Detector

A Deep Learning web application that classifies whether an uploaded image is a **real photograph** or **AI-generated**, using transfer learning and Grad-CAM explainability.

---

## Project Structure

```
Pixel Truth/
├── model/
│   └── detector.py        # Hugging Face model loader & inference
├── backend/
│   ├── app.py             # Flask REST API
│   └── gradcam.py         # Grad-CAM heatmap generation
├── frontend/
│   ├── index.html         # Main UI
│   ├── style.css          # Styling
│   └── script.js          # Upload & result handling
├── static/
│   └── uploads/           # Temporary image storage
├── docs/
│   └── GUIDE.md           # This file
├── requirements.txt       # Python dependencies
└── run.py                 # Entry point
```

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python run.py
```

Then open **http://localhost:5000** in your browser.

> On first run, the model (~300MB) downloads automatically from Hugging Face. This takes a few minutes once only.

---

## How to Use

1. Open the app in your browser
2. Drag and drop an image or click to browse
3. Wait for the analysis (a few seconds)
4. View the result:
   - **Verdict:** Real Image or AI Generated
   - **Confidence:** How certain the model is
   - **Grad-CAM:** Heatmap showing which regions influenced the prediction

---

## Model Details

- **Model:** `umm-maybe/AI-image-detector` (Hugging Face)
- **Architecture:** Vision Transformer (ViT) fine-tuned on real vs AI-generated images
- **Classes:** Real Image / AI Generated
- **Explainability:** Grad-CAM heatmap overlay

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Model | Hugging Face Transformers |
| Visualization | Grad-CAM |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

## Deep Learning Concepts Demonstrated

- Transfer Learning (pre-trained ViT model)
- Image Classification (binary)
- Grad-CAM explainability
- Image preprocessing pipeline
- REST API model deployment
