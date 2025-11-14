"""
Hurricane Harvey Building Damage Classification - Inference Server
Flask API for model inference
"""

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = 'best_model.h5'
IMG_SIZE = (128, 128)  # Must match training size

# Load model at startup
print("Loading model...")
model = None
model_info = {}

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    
    # Load model metadata
    if os.path.exists('best_model_info.json'):
        with open('best_model_info.json', 'r') as f:
            model_info = json.load(f)
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise


def preprocess_image(image_bytes):
    """
    Preprocess image for model inference.
    Accepts raw binary image data.
    """
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


@app.route('/summary', methods=['GET'])
def get_summary():
    """
    GET /summary endpoint
    Returns metadata about the model.
    """
    try:
        # Get model architecture summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        model_summary = "\n".join(summary_list)
        
        # Count parameters
        total_params = model.count_params()
        
        # Prepare response
        response = {
            "model_name": model_info.get('model_name', 'Unknown'),
            "model_architecture": model.name if hasattr(model, 'name') else 'Sequential',
            "input_shape": [None, IMG_SIZE[0], IMG_SIZE[1], 3],
            "output_shape": [None, 1],
            "total_parameters": int(total_params),
            "test_accuracy": model_info.get('test_accuracy', 'N/A'),
            "image_size": IMG_SIZE,
            "preprocessing": {
                "resize": IMG_SIZE,
                "normalization": "[0, 1]",
                "color_mode": "RGB"
            },
            "prediction_classes": ["no_damage", "damage"],
            "model_summary": model_summary
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/inference', methods=['POST'])
def predict():
    """
    POST /inference endpoint
    Accepts image data (multipart form data or binary) and returns prediction.
    
    Expected response format:
    {
        "prediction": "damage" or "no_damage"
    }
    """
    try:
        image_bytes = None
        
        # Try to get image from multipart form data (grader uses this)
        if 'image' in request.files:
            file = request.files['image']
            image_bytes = file.read()
        # Fall back to raw binary data
        elif request.data:
            image_bytes = request.data
        else:
            return jsonify({"error": "No image data provided"}), 400
        
        # Preprocess image
        try:
            preprocessed_image = preprocess_image(image_bytes)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Make prediction
        prediction_prob = model.predict(preprocessed_image, verbose=0)[0][0]
        
        # Convert to class label
        # Threshold at 0.5: >= 0.5 is "damage" (class 1), < 0.5 is "no_damage" (class 0)
        if prediction_prob >= 0.5:
            prediction_class = "damage"
        else:
            prediction_class = "no_damage"
        
        # Prepare response - MUST match exact specification
        response = {
            "prediction": prediction_class
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


@app.route('/', methods=['GET'])
def home():
    """
    Root endpoint with API information
    """
    return jsonify({
        "service": "Hurricane Harvey Building Damage Classification API",
        "version": "1.0",
        "endpoints": {
            "GET /": "API information",
            "GET /summary": "Model metadata and summary",
            "POST /inference": "Image classification endpoint (accepts binary image data)",
            "GET /health": "Health check"
        },
        "usage": {
            "inference": "Send POST request to /inference with binary image data in request body"
        }
    }), 200


if __name__ == '__main__':
    # Run Flask app
    # Use 0.0.0.0 to accept connections from outside the container
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
