#!/usr/bin/env python3
########################################################################
#       AN OPEN/GROQ Tomatoe leafs classifier INTEGRATION MY NWALI UGONNA EMMANUEL
#       GITHUB: https://github.com/Tigo-cmd/TigoAi
#       All contributions are welcome!!!
#       yea lets do some coding!!!!!!!!!!!!
#########################################################################

"""Source classes to handle all GroqAPI functionalities and trained models
                        BY Nwali Ugonna Emmanuel
            Backend API for Tomato Leaf Disease Detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from source import TigoGroq, load_dotenv
from flask_cors import CORS
import asyncio

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow all origins

# Initialize AI Assistant
load_dotenv()
client = TigoGroq()

# ... (rest of your app.py code)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests and returns AI response"""
    if 'message' not in request.json:
        return jsonify({"error": "No message provided"}), 400

    user_message = request.json['message']

    try:
        ai_response = asyncio.run(client.get_response_from_ai(user_message))
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load trained model
MODEL_PATH = 'model_inception.h5'
model = load_model(MODEL_PATH)

# Create an uploads directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease information dictionary
disease_info = {
    "Bacterial_spot": {
        "description": "Bacterial spot is a common tomato disease that causes small, water-soaked spots on leaves and fruit.",
        "symptoms": ["Small, water-soaked spots on leaves", "Spots may have a yellow halo", "Fruit lesions can be raised and scabby"],
        "treatment": ["Remove infected leaves", "Apply copper-based fungicides", "Practice crop rotation"]
    },
    "Early_blight": {
        "description": "Early blight is a fungal disease that causes dark, concentric rings on leaves and fruit.",
        "symptoms": ["Dark, circular spots with concentric rings", "Yellowing of leaves", "Fruit rot"],
        "treatment": ["Remove infected leaves", "Apply fungicides", "Improve air circulation"]
    },
    "Late_blight": {
        "description": "Late blight is a devastating fungal disease that can quickly destroy tomato plants.",
        "symptoms": ["Water-soaked lesions on leaves and stems", "White, fuzzy growth on the underside of leaves", "Rapid plant collapse"],
        "treatment": ["Remove and destroy infected plants", "Apply fungicides", "Improve drainage"]
    },
    "Leaf_Mold": {
        "description": "Leaf mold is a fungal disease that causes yellow spots on the upper surface of leaves and gray mold on the underside.",
        "symptoms": ["Yellow spots on the upper leaf surface", "Grayish-purple mold on the underside", "Leaf drop"],
        "treatment": ["Improve air circulation", "Remove infected leaves", "Apply fungicides"]
    },
    "Septoria_leaf_spot": {
        "description": "Septoria leaf spot is a fungal disease that causes small, circular spots with dark borders on leaves.",
        "symptoms": ["Small, circular spots with dark borders", "Yellowing and wilting of leaves", "Defoliation"],
        "treatment": ["Remove infected leaves", "Apply fungicides", "Practice crop rotation"]
    },
    "Spider_mites Two-spotted_spider_mite": {
        "description": "Spider mites are tiny pests that suck plant sap, causing stippling and webbing.",
        "symptoms": ["Fine webbing on leaves", "Yellow stippling on leaves", "Leaf drop"],
        "treatment": ["Spray with water", "Apply insecticidal soap or neem oil", "Introduce predatory mites"]
    },
    "Target_Spot": {
        "description": "Target spot is a fungal disease that causes circular spots with concentric rings, similar to a target.",
        "symptoms": ["Circular spots with concentric rings", "Yellowing of surrounding tissue", "Leaf drop"],
        "treatment": ["Remove infected leaves", "Apply fungicides", "Improve air circulation"]
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato yellow leaf curl virus is a viral disease that causes yellowing and curling of leaves.",
        "symptoms": ["Yellowing and curling of leaves", "Stunted growth", "Reduced fruit production"],
        "treatment": ["Remove infected plants", "Control whitefly populations", "Use resistant varieties"]
    },
    "Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus is a viral disease that causes mosaic patterns on leaves and stunted growth.",
        "symptoms": ["Mosaic patterns on leaves", "Stunted growth", "Reduced fruit production"],
        "treatment": ["Remove infected plants", "Control aphids", "Use resistant varieties"]
    },
    "Healthy": {
        "description": "The plant appears to be healthy and free of disease.",
        "symptoms": ["Vibrant green leaves", "Normal growth", "No signs of spots or discoloration"],
        "treatment": ["Continue regular care and monitoring"]
    },
    "Unknown": {
        "description": "The disease could not be identified.",
        "symptoms": ["Unknown"],
        "treatment": ["Unknown"]
    }
}


def model_predict(img_path, model):
    """Preprocess image and make predictions"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0  # Normalize
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    predicted_class_index = np.argmax(preds, axis=1)[0]  # Get highest probability class
    probability = np.max(preds) * 100 # Get the probability of the predicted class
    

    disease_mapping = {
        0: "Bacterial_spot",
        1: "Early_blight",
        2: "Late_blight",
        3: "Leaf_Mold",
        4: "Septoria_leaf_spot",
        5: "Spider_mites Two-spotted_spider_mite",
        6: "Target_Spot",
        7: "Tomato_Yellow_Leaf_Curl_Virus",
        8: "Tomato_mosaic_virus",
        9: "Healthy"
    }

    predicted_disease = disease_mapping.get(predicted_class_index, "Unknown")

    return predicted_disease, probability, predicted_class_index


@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload and returns JSON prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Securely save file
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    # Get prediction
    disease, probability, predicted_class_index = model_predict(file_path, model)
    print(disease)

    # Get detailed disease information
    disease_details = disease_info.get(disease, disease_info["Unknown"])

    # Create the response dictionary
    response = {
        "disease": disease,
        "probability": probability,
        "description": disease_details["description"],
        "symptoms": disease_details["symptoms"],
        "treatment": disease_details["treatment"],
        "message": f"Prediction completed. Disease detected: {disease}",
        "predicted_class_index": int(predicted_class_index)
    }

    return jsonify(response)


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Tomato Leaf Disease Detection API"})


if __name__ == '__main__':
    app.run(port=5001, debug=True)
