import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app import app

if __name__ == '__main__':
    print("Starting PixelTruth - AI Image Detector...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000, use_reloader=False)
