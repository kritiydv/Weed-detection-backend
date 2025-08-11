import os
import json
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# Configuration
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Setup Flask
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# Load weed knowledge base
with open('weed_info.json', 'r', encoding='utf-8') as f:
    weed_info = json.load(f)

# Load YOLOv8 model
model = YOLO('best.pt')  # Replace with your model path

# Helper: check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: preprocess and infer
def run_detection(image_path):
    results = model(image_path)[0]
    return results

# Route: Weed detection with language toggle
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Get requested language from query param; default to Hindi
    lang = request.args.get('lang', 'hi')

    # Save uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Run YOLOv8 detection
    results = run_detection(image_path)

    # Save annotated image
    annotated_img = results.plot()
    result_path = os.path.join(ANNOTATED_FOLDER, f"result_{file.filename}")
    cv2.imwrite(result_path, annotated_img)

    detected_weeds = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = model.names[cls_id]

        if label in weed_info:
            info = weed_info[label]
            translations = info.get("language_translations", {})

            detected_weeds.append({
                "name": translations.get("name", {}).get(lang, label),
                "description": translations.get("description", {}).get(lang, info.get("details", "")),
                "eco_impact": translations.get("eco_impact", {}).get(lang, info.get("eco_impact", "")),
                "growth_season": translations.get("growth_season", {}).get(lang, info.get("growth_season", "")),
                "precautions": translations.get("precautions", {}).get(lang, info.get("precautions", "")),
                "organic_control": translations.get("organic_control", {}).get(lang, info.get("organic_control", "")),
                "herbicides": translations.get("herbicide_names", {}).get(lang, info.get("herbicide_names", [])),
                "affected_crops": translations.get("affected_crops", {}).get(lang, info.get("affected_crops", [])),
                "common_in_states": translations.get("common_in_states", {}).get(lang, info.get("common_in_states", [])),
                "image_urls": info.get("image_urls", [])
            })

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:5000")

    return jsonify({
    "status": "success",
    "lang": lang,
    "backend_url": backend_url,
    "annotated_image": f"{backend_url}/{result_path}",
    "detected_weeds": detected_weeds
})



# Start Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
