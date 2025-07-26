
# BP Fuel AI - Blood Pressure Estimation App

This project demonstrates a web application built with a React frontend and a Python Flask backend that estimates blood pressure. It combines traditional patient health information (collected via a questionnaire) with (simulated) computer vision features derived from a webcam or uploaded image/video. The application then provides personalized health tips based on the estimated blood pressure and questionnaire responses.

**Disclaimer:** This app is for educational and demonstration purposes only and and does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.

## Features

* **Interactive Questionnaire:** Collects user's age, gender, diet, exercise habits, smoking/alcohol status, and previous health conditions.
* **(Simulated) Computer Vision Integration:** Allows users to capture an image via webcam or upload an image/video, from which hypothetical computer vision features (e.g., facial redness, estimated heart rate from video) are simulated and used in the prediction model.
* **Machine Learning Model:** Utilizes a Random Forest Regressor trained on a cardiovascular disease dataset to estimate systolic and diastolic blood pressure.
* **Personalized Health Tips:** Provides suggestions and warnings based on the estimated blood pressure category (Normal, Elevated, Hypertension Stage 1 & 2, Hypotension) and questionnaire responses.
* **Modern Web Stack:** React for the frontend, Flask for the Python backend API, managed with Vite and styled with Tailwind CSS.

## Project Structure

```

.
├── public/                 \# React public assets
├── src/                    \# React source code (components, pages, utilities)
├── cardio\_train.csv        \# The dataset used for training (download separately)
├── model\_training.py       \# Python script to train ML models and save them
├── bp\_prediction\_service.py \# Python module for ML prediction logic
├── api\_server.py           \# Python Flask API server
├── requirements.txt        \# Python dependencies list for backend
├── Procfile                \# Render deployment configuration for backend
├── package.json            \# React project dependencies and scripts
├── package-lock.json
├── tailwind.config.js
├── vite.config.js
├── tsconfig.json           \# TypeScript configuration
├── tsconfig.app.json
├── tsconfig.node.json
├── postcss.config.js
└── README.md               \# This README file

````

**Note:** The generated `.pkl` model files (`systolic_bp_model.pkl`, `diastolic_bp_model.pkl`, `data_preprocessor.pkl`) are **not directly committed** to this GitHub repository. Instead, they are hosted on Hugging Face Hub and downloaded by the backend during deployment due to their size.

## Setup and Running the Application

Follow these steps to set up and run the application locally, and then to deploy it.

### **Phase 1: Backend Setup (Python ML Service)**

This phase prepares your machine learning models and makes them ready to be served by an API.

#### **1. Acquire the Dataset**
* Download the `cardio_train.csv` dataset from Kaggle: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
* Unzip it and place the `cardio_train.csv` file directly into your project's root directory.

#### **2. Install Python Backend Dependencies**
* Open your terminal in VS Code.
* Navigate to your project's root directory.
* Create a `requirements.txt` file (if not already present) with the following content:
    ```
    Flask
    Flask-Cors
    numpy
    pandas
    scikit-learn
    opencv-python
    joblib
    gunicorn
    requests
    ```
* Install these dependencies:
    ```bash
    pip install -r requirements.txt
    ```

#### **3. Generate and Upload ML Models to Hugging Face Hub**

##### **a. Generate `.pkl` Files Locally**
* Ensure `model_training.py` is in your project root.
* Run the script to generate your `.pkl` model files locally:
    ```bash
    python model_training.py
    ```
    This will create `systolic_bp_model.pkl`, `diastolic_bp_model.pkl`, and `data_preprocessor.pkl` in your root directory.

##### **b. Authenticate with Hugging Face Hub**
* Install `huggingface_hub`: `pip install huggingface_hub`
* Generate a **new token with `write` role** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
* Log in from your terminal:
    ```bash
    python -c "from huggingface_hub import login; login()"
    # Paste your token when prompted.
    ```

##### **c. Create Hugging Face Model Repository**
* Go to [https://huggingface.co/new](https://huggingface.co/new).
* Set **Owner:** Your username, **Model name:** `BloodPressureDetectionSystem`, **Visibility:** `Public`, **Model type:** `Other`. Click "Create model".

##### **d. Upload `.pkl` Files to Hugging Face**
* Create an `upload_models_to_hf.py` file in your root with this content. **Replace `Tayyablegend` with your actual Hugging Face username.**
    ```python
    # upload_models_to_hf.py
    from huggingface_hub import HfApi
    import os

    HF_USERNAME = "Tayyablegend" # REPLACE with your actual HF username
    HF_MODEL_REPO_NAME = "BloodPressureDetectionSystem"

    repo_id = f"{HF_USERNAME}/{HF_MODEL_REPO_NAME}"
    api = HfApi()

    local_model_files = [
        "systolic_bp_model.pkl",
        "diastolic_bp_model.pkl",
        "data_preprocessor.pkl"
    ]

    print(f"Attempting to upload files to {repo_id}...")
    for filename in local_model_files:
        if os.path.exists(filename):
            try:
                api.upload_file(path_or_fileobj=filename, path_in_repo=filename, repo_id=repo_id, repo_type="model")
                print(f"✅ Successfully uploaded {filename} to {repo_id}")
            except Exception as e:
                print(f"❌ Error uploading {filename}: {e}")
        else:
            print(f"⚠️ Warning: {filename} not found locally. Please run 'model_training.py' first to generate them.")
    print("\nUpload process finished. Verify files on Hugging Face Hub.")
    ```
* Run this script:
    ```bash
    python upload_models_to_hf.py
    ```
* **Verify on Hugging Face:** Confirm `systolic_bp_model.pkl`, `diastolic_bp_model.pkl`, `data_preprocessor.pkl` are listed under the "Files and versions" tab of your repository (`https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/tree/main`).

#### **4. Configure and Run Backend Locally (for testing)**

##### **a. Update `bp_prediction_service.py` with HF URLs**
* Get the direct download URLs for your `.pkl` files from Hugging Face (Right-click "Download" button on each file's page, copy link address).
    * `https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/systolic_bp_model.pkl?download=true`
    * `https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/diastolic_bp_model.pkl?download=true`
    * `https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/data_preprocessor.pkl?download=true`
* Open `bp_prediction_service.py` and update the `MODEL_URLS` dictionary with these exact URLs.

    ```python
    # bp_prediction_service.py (excerpt)
    MODEL_URLS = {
        'systolic_bp_model.pkl': '[https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/systolic_bp_model.pkl?download=true](https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/systolic_bp_model.pkl?download=true)',
        'diastolic_bp_model.pkl': '[https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/diastolic_bp_model.pkl?download=true](https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/diastolic_bp_model.pkl?download=true)',
        'data_preprocessor.pkl': '[https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/data_preprocessor.pkl?download=true](https://huggingface.co/Tayyablegend/BloodPressureDetectionSystem/resolve/main/data_preprocessor.pkl?download=true)'
    }
    # ... (rest of the file)
    ```

##### **b. Create `api_server.py`**
* Create `api_server.py` in your root with this content:
    ```python
    # api_server.py
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import bp_prediction_service
    import numpy as np
    import cv2
    import base64
    import os

    app = Flask(__name__)
    CORS(app)

    def initialize_ml_service():
        if not bp_prediction_service.load_ml_assets():
            print("CRITICAL: Failed to load ML assets. API may not function correctly.")

    @app.route('/predict_health', methods=['POST'])
    def predict_health():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        user_data = {
            'age': data.get('age'), 'gender': data.get('gender'), 'diet': data.get('diet'), 'salt_intake': data.get('salt_intake'),
            'exercise': data.get('exercise'), 'smoker': data.get('smoker'), 'alcohol': data.get('alcohol'), 'prev_conditions': data.get('prev_conditions', []),
            'height': data.get('height', 170), 'weight': data.get('weight', 70), 'cholesterol': data.get('cholesterol', 1), 'gluc': data.get('gluc', 1)
        }
        image_b64 = data.get('image_data') 
        frame = None ; cv_features = {}
        if image_b64:
            try:
                img_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    np.random.seed(frame.shape[0] * frame.shape[1])
                else: np.random.seed(42)
                cv_features = {
                    'FacialRednessIndex': np.random.uniform(0.4, 0.7), 'EyeAreaRatio': np.random.uniform(0.02, 0.04),
                    'SkinToneVariability': np.random.uniform(0.003, 0.007), 'EstimatedHeartRate_CV': np.random.randint(65, 95),
                    'PPG_SignalNoiseRatio': np.random.uniform(15, 25.0)
                }
            except Exception as e:
                print(f"Error processing image: {e}")
                return jsonify({"error": f"Invalid image data: {e}"}), 400
        else: cv_features = {'FacialRednessIndex': 0.5, 'EyeAreaRatio': 0.03, 'SkinToneVariability': 0.005, 'EstimatedHeartRate_CV': 75, 'PPG_SignalNoiseRatio': 20.0 }
        systolic_bp, diastolic_bp = bp_prediction_service.predict_blood_pressure(user_data, cv_features)
        if systolic_bp is not None and diastolic_bp is not None:
            tips = bp_prediction_service.generate_tips(
                user_data.get('age', 0), user_data.get('diet', 'Average'), user_data.get('salt_intake', 'Moderate'),
                user_data.get('exercise', 'Rarely'), user_data.get('smoker', 'No'), user_data.get('alcohol', 'No'),
                user_data.get('prev_conditions', []), systolic_bp, diastolic_bp
            )
            return jsonify({'systolic_bp': int(systolic_bp), 'diastolic_bp': int(diastolic_bp), 'tips': tips})
        else: return jsonify({"error": "Failed to predict blood pressure. Internal server error."}), 500

    if __name__ == '__main__':
        initialize_ml_service()
        app.run(debug=True, port=5000)
    ```

##### **c. Run Flask Backend Locally**
* Open a **NEW terminal** in VS Code.
* Navigate to your project root and run:
    ```bash
    python api_server.py
    ```
    Keep this terminal open.

---

### **Phase 2: Frontend Setup (React Application)**

This phase prepares your React application to connect to the backend.

#### **5. Install Node.js Frontend Dependencies**
* Open your **original terminal** in VS Code.
* Navigate to your project root.
* Run:
    ```bash
    npm install
    ```

#### **6. Create `src/utils/api.js`**
* Create `src/utils/api.js` with this content:
    ```javascript
    // src/utils/api.js
    const API_BASE_URL = '[http://127.0.0.1:5000](http://127.0.0.1:5000)'; // For LOCAL testing

    export const predictHealth = async (formData, imageData) => {
      try {
        const response = await fetch(`${API_BASE_URL}/predict_health`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_data: formData, image_data: imageData }),
        });
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Something went wrong with the API request.');
        }
        return response.json();
      } catch (error) {
        console.error('API call error:', error);
        throw error;
      }
    };
    ```

#### **7. Modify `src/pages/BPCheck.jsx`**
* Replace the content of your `src/pages/BPCheck.jsx` with the code I provided previously (the one with the `use_container_width` fix and enhanced tips). This includes the UI, webcam/upload logic, and calls to `predictHealth`.
* Ensure the dummy UI components at the bottom of `BPCheck.jsx` are present or replaced by your project's actual UI components.

#### **8. Run React Frontend Locally**
* In the same terminal where you ran `npm install`.
* Run:
    ```bash
    npm run dev
    ```
    This will open your React app in a browser (e.g., `http://localhost:5173`).

---

### **Phase 3: Deploy to Render.com & Final Integration**

This phase makes your backend API live online and connects your frontend to it.

#### **9. Commit & Push Backend to GitHub**
* Ensure `api_server.py`, `bp_prediction_service.py`, `requirements.txt`, `Procfile`, `cardio_train.csv`, and `model_training.py` are all committed and pushed to your **backend's GitHub repository** (e.g., `Tayyablegend/BloodPressureDetectionSystem`).
    * Use `git add .` (if you have a `.gitignore` to exclude `.pkl`s) or `git add api_server.py bp_prediction_service.py cardio_train.csv model_training.py Procfile requirements.txt`.
    * `git commit -m "Final backend for Render"`
    * `git push origin main`

#### **10. Deploy Backend on Render.com**
* Go to [https://render.com/](https://render.com/) and create a **Web Service**.
* Connect to your **backend GitHub repository**.
* **Configure:** Name, Region, Branch (`main`), Root Directory (blank), Runtime (`Python 3`), Build Command (`pip install -r requirements.txt`), Start Command (`gunicorn api_server:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120`).
* Create Service.
* **Monitor Deploy Logs:** Watch closely for `Successfully downloaded` messages from Hugging Face and `ML models and preprocessor loaded successfully.`.
* **Copy Public URL:** Once "Live", copy the Render URL (e.g., `https://your-backend-name.onrender.com`).

#### **11. Update Frontend API URL for Live Deployment**
* In your React project, open `src/utils/api.js`.
* **Change `API_BASE_URL`** from `http://127.0.0.1:5000` to your **actual Render backend URL**.

    ```javascript
    // src/utils/api.js (for Live Deployment)
    const API_BASE_URL = '[https://your-backend-name.onrender.com](https://your-backend-name.onrender.com)'; // REPLACE THIS with your Render URL
    ```
* **Save the file.**

#### **12. Commit & Deploy React Frontend**
* Commit the updated `src/utils/api.js` to your **frontend's GitHub repository**.
* Push to GitHub.
* Redeploy your React frontend (e.g., if hosted on Netlify/Vercel/Streamlit Community Cloud).

#### **13. Final Test**
* Access your live deployed React frontend in your browser.
* Navigate to the BP check page, fill out the form, and try the image capture/upload.
* Confirm that it successfully fetches blood pressure predictions and tips from your live Render backend.

This is the complete, integrated process. You're very close!
````
