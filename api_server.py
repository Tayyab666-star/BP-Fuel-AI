from flask import Flask, request, jsonify
from flask_cors import CORS
import bp_prediction_service
import numpy as np
import cv2
import base64
import io

app = Flask(__name__)
CORS(app) # Enable CORS for your frontend to communicate with this API

# --- Initialize ML models when the API server starts ---
# This function will now be called directly when the script runs.
# No @app.before_first_request needed here for this simple setup.
def initialize_ml_service():
    if not bp_prediction_service.load_ml_assets():
        print("CRITICAL: Failed to load ML assets. API may not function correctly.")
        # In a production setup, you might want to raise an exception or log a critical error.
        # For development, you might just let it proceed with errors logged.

# --- API Endpoint for Health Prediction ---
@app.route('/predict_health', methods=['POST'])
def predict_health():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # 1. Get User Questionnaire Data
    user_data = {
        'age': data.get('age'),
        'gender': data.get('gender'),
        'diet': data.get('diet'),
        'salt_intake': data.get('salt_intake'),
        'exercise': data.get('exercise'),
        'smoker': data.get('smoker'),
        'alcohol': data.get('alcohol'),
        'prev_conditions': data.get('prev_conditions', []),
        'height': data.get('height', 170), # Placeholder if not sent from frontend
        'weight': data.get('weight', 70),  # Placeholder if not sent from frontend
        'cholesterol': data.get('cholesterol', 1), # Placeholder
        'gluc': data.get('gluc', 1) # Placeholder
    }

    # 2. Handle Image/Video Data for (Simulated) CV Features
    image_b64 = data.get('image_data') 
    frame = None
    cv_features = {}

    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # --- SIMULATE CV FEATURE EXTRACTION FROM FRAME ---
            if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                np.random.seed(frame.shape[0] * frame.shape[1])
            else:
                np.random.seed(42)

            cv_features = {
                'FacialRednessIndex': np.random.uniform(0.4, 0.7),
                'EyeAreaRatio': np.random.uniform(0.02, 0.04),
                'SkinToneVariability': np.random.uniform(0.003, 0.007),
                'EstimatedHeartRate_CV': np.random.randint(65, 95),
                'PPG_SignalNoiseRatio': np.random.uniform(15, 25.0)
            }
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({"error": f"Invalid image data: {e}"}), 400
    else:
        cv_features = {
            'FacialRednessIndex': 0.5,
            'EyeAreaRatio': 0.03,
            'SkinToneVariability': 0.005,
            'EstimatedHeartRate_CV': 75,
            'PPG_SignalNoiseRatio': 20.0
        }


    # 3. Make Prediction
    systolic_bp, diastolic_bp = bp_prediction_service.predict_blood_pressure(user_data, cv_features)

    if systolic_bp is not None and diastolic_bp is not None:
        # 4. Generate Tips
        tips = bp_prediction_service.generate_tips(
            user_data.get('age', 0),
            user_data.get('diet', 'Average'),
            user_data.get('salt_intake', 'Moderate'),
            user_data.get('exercise', 'Rarely'),
            user_data.get('smoker', 'No'),
            user_data.get('alcohol', 'No'),
            user_data.get('prev_conditions', []),
            systolic_bp,
            diastolic_bp
        )
        return jsonify({
            'systolic_bp': int(systolic_bp),
            'diastolic_bp': int(diastolic_bp),
            'tips': tips
        })
    else:
        return jsonify({"error": "Failed to predict blood pressure. Internal server error."}), 500

if __name__ == '__main__':
    # Call the initialization function directly before starting the app
    initialize_ml_service() 
    app.run(debug=True, port=5000)