import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import base64
import io

from model.detector import predict, load_models, _models
from backend.gradcam import generate_gradcam

app = Flask(__name__, static_folder='../frontend')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('../frontend', filename)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Unsupported file type. Use PNG, JPG, WEBP, or GIF'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        label, confidence = predict(image)

        model_pipeline = next(iter(_models.values())) if _models else None
        gradcam_b64 = generate_gradcam(image, model_pipeline) if model_pipeline else ""

        buf = io.BytesIO()
        image.save(buf, format='PNG')
        original_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'label': label,
            'confidence': confidence,
            'gradcam_image': gradcam_b64,
            'original_image': original_b64
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
