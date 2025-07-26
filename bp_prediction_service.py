import numpy as np
import pandas as pd
import joblib
import os # For checking if files exist

# --- Global Variables for Loaded Models and Preprocessor ---
# These will be loaded once when this module is imported
systolic_model = None
diastolic_model = None
data_preprocessor = None
ml_feature_columns = None

# --- Load Models and Preprocessor ---
def load_ml_assets():
    """
    Loads the pre-trained ML models and data preprocessor.
    This function should be called once at the start of your application.
    """
    global systolic_model, diastolic_model, data_preprocessor, ml_feature_columns

    model_files = ['systolic_bp_model.pkl', 'diastolic_bp_model.pkl', 'data_preprocessor.pkl']
    for f in model_files:
        if not os.path.exists(f):
            print(f"Error: Model file '{f}' not found. Please ensure it's in the same directory.")
            print("You need to run 'model_training.py' first to generate these files.")
            return False # Indicate failure to load

    try:
        systolic_model = joblib.load('systolic_bp_model.pkl')
        diastolic_model = joblib.load('diastolic_bp_model.pkl')
        data_preprocessor = joblib.load('data_preprocessor.pkl')
        
        # This list must exactly match the feature columns used during training.
        # It's crucial for preparing new input data correctly.
        ml_feature_columns = [
            'gender', 'height', 'weight', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'age_years', 'FacialRednessIndex', 'EyeAreaRatio', 'SkinToneVariability',
            'EstimatedHeartRate_CV', 'PPG_SignalNoiseRatio'
        ]
        print("ML models and preprocessor loaded successfully.")
        return True # Indicate successful load
    except Exception as e:
        print(f"Error loading ML assets: {e}")
        return False # Indicate failure to load

# --- Helper function to prepare input data for the ML models ---
def prepare_features_for_prediction(user_data: dict, cv_features: dict) -> np.ndarray:
    """
    Prepares input data for the ML models by combining user questionnaire
    data and computer vision features, then preprocessing them.

    Args:
        user_data (dict): Dictionary of user's questionnaire responses.
                          Expected keys: 'age', 'gender', 'diet', 'salt_intake',
                          'exercise', 'smoker', 'alcohol', 'prev_conditions'.
        cv_features (dict): Dictionary of simulated or extracted computer vision features.
                            Expected keys: 'FacialRednessIndex', 'EyeAreaRatio',
                            'SkinToneVariability', 'EstimatedHeartRate_CV',
                            'PPG_SignalNoiseRatio'.

    Returns:
        np.ndarray: Preprocessed input array ready for model prediction.
    """
    if data_preprocessor is None or ml_feature_columns is None:
        raise RuntimeError("ML assets not loaded. Call load_ml_assets() first.")

    # Map general user data to the specific features expected by the model
    # Fill in placeholder/default values for features not collected directly from a simple UI
    data_for_prediction = {
        'age_years': user_data.get('age', 30), # Default age
        'gender': 1 if user_data.get('gender') == 'Male' else (2 if user_data.get('gender') == 'Female' else 0), # 1: Male, 2: Female, 0: Other/Unknown
        'height': user_data.get('height', 170), # Default height (cm)
        'weight': user_data.get('weight', 70),  # Default weight (kg)
        
        # Cholesterol and Gluc are typically 1=normal, 2=above normal, 3=well above normal
        # Assuming normal (1) if not explicitly provided by the user in your app
        'cholesterol': user_data.get('cholesterol', 1),
        'gluc': user_data.get('gluc', 1), 
        
        'smoke': 1 if user_data.get('smoker') == 'Yes' else 0, # 1: Yes, 0: No
        'alco': 1 if user_data.get('alcohol') == 'Yes' else 0, # 1: Yes, 0: No
        # Simplified 'active' based on exercise frequency from questionnaire
        'active': 1 if user_data.get('exercise') in ["Daily", "Few times a week"] else 0, # 1: Active, 0: Not Active
        
        # CV Features (either truly extracted or simulated by your calling app)
        'FacialRednessIndex': cv_features.get('FacialRednessIndex', 0.5),
        'EyeAreaRatio': cv_features.get('EyeAreaRatio', 0.03),
        'SkinToneVariability': cv_features.get('SkinToneVariability', 0.005),
        'EstimatedHeartRate_CV': cv_features.get('EstimatedHeartRate_CV', 75),
        'PPG_SignalNoiseRatio': cv_features.get('PPG_SignalNoiseRatio', 20.0)
    }

    # Create DataFrame in the exact order expected by the preprocessor
    # This is crucial for correct feature mapping
    input_df = pd.DataFrame([data_for_prediction], columns=ml_feature_columns)

    # Transform the input data using the loaded preprocessor
    processed_input = data_preprocessor.transform(input_df)
    return processed_input

# --- Main Prediction Function ---
def predict_blood_pressure(user_data: dict, cv_features: dict) -> tuple[float, float] | tuple[None, None]:
    """
    Predicts systolic and diastolic blood pressure using the loaded ML models.

    Args:
        user_data (dict): Dictionary of user's questionnaire responses.
        cv_features (dict): Dictionary of simulated or extracted computer vision features.

    Returns:
        tuple[float, float] | tuple[None, None]: A tuple (systolic_bp, diastolic_bp) if successful,
                                                  otherwise (None, None).
    """
    if systolic_model is None or diastolic_model is None or data_preprocessor is None:
        print("Error: Models or preprocessor not loaded. Call load_ml_assets() first.")
        return None, None

    try:
        processed_input = prepare_features_for_prediction(user_data, cv_features)
        
        predicted_systolic = systolic_model.predict(processed_input)[0]
        predicted_diastolic = diastolic_model.predict(processed_input)[0]

        return float(predicted_systolic), float(predicted_diastolic)
    except Exception as e:
        print(f"Error during blood pressure prediction: {e}")
        return None, None

# --- Health Tips Generation Function ---
def generate_tips(age: int, diet: str, salt_intake: str, exercise: str, smoker: str, alcohol: str, prev_conditions: list, systolic: float | None = None, diastolic: float | None = None) -> list[str]:
    """
    Generates personalized health tips based on user data and blood pressure.

    Args:
        age (int): User's age.
        diet (str): User's diet description.
        salt_intake (str): User's salt intake level.
        exercise (str): User's exercise frequency.
        smoker (str): User's smoking status.
        alcohol (str): User's alcohol consumption status.
        prev_conditions (list): List of user's previous health conditions.
        systolic (float | None): Estimated systolic blood pressure.
        diastolic (float | None): Estimated diastolic blood pressure.

    Returns:
        list[str]: A list of personalized health tips.
    """
    tips = []
    
    # Blood Pressure Category Based Suggestions
    if systolic is not None and diastolic is not None:
        if systolic < 90 or diastolic < 60:
            tips.append("### ⚠️ Low Blood Pressure (Hypotension) Suggestions:")
            tips.append("-   Consult a Doctor: If you frequently experience low BP symptoms (dizziness, fainting), see a healthcare professional for diagnosis and treatment.")
            tips.append("-   Stay Hydrated: Drink plenty of fluids throughout the day to prevent dehydration, which can lower blood pressure.")
            tips.append("-   Increase Salt (with caution): Discuss with your doctor if increasing sodium intake slightly is appropriate for you.")
            tips.append("-   Small, Frequent Meals: Eating smaller, low-carb meals can help prevent sudden drops in BP after eating.")
            tips.append("-   Avoid Sudden Movements: Rise slowly from sitting or lying positions to prevent orthostatic hypotension.")
            tips.append("---")
        elif (systolic >= 90 and systolic <= 120) and (diastolic >= 60 and diastolic <= 80):
            tips.append("### ✅ Normal Blood Pressure Suggestions:")
            tips.append("-   Maintain Healthy Habits: Keep up your balanced diet, regular exercise, and stress management practices.")
            tips.append("-   Regular Check-ups: Even with normal BP, routine medical check-ups are important for overall health monitoring.")
            tips.append("-   Monitor Trends: Be aware of any changes in your readings over time.")
            tips.append("---")
        elif (systolic > 120 and systolic <= 129) and (diastolic >= 60 and diastolic <= 80):
            tips.append("### 💛 Elevated Blood Pressure Suggestions (Pre-Hypertension):")
            tips.append("-   Lifestyle Changes are Key: This is a crucial stage to prevent hypertension. Focus on lifestyle modifications.")
            tips.append("-   DASH Diet Emphasis: Strictly follow the DASH diet principles.")
            tips.append("-   Increase Physical Activity: Aim for at least 150 minutes of moderate-intensity exercise per week.")
            tips.append("-   Limit Sodium & Alcohol: Reduce salt intake and moderate alcohol consumption.")
            tips.append("-   Stress Management: Implement stress-reduction techniques like meditation or yoga.")
            tips.append("---")
        elif (systolic >= 130 and systolic <= 139) or (diastolic >= 80 and diastolic <= 89):
            tips.append("### 🧡 High Blood Pressure (Hypertension Stage 1) Suggestions:")
            tips.append("-   Consult a Doctor: Your doctor may recommend lifestyle changes and possibly medication.")
            tips.append("-   Consistent Monitoring: Monitor your blood pressure at home regularly and keep a log for your doctor.")
            tips.append("-   Dietary Adjustments: Seriously commit to low-sodium, heart-healthy foods (DASH diet).")
            tips.append("-   Regular Exercise: Make physical activity a consistent part of your routine.")
            tips.append("---")
        elif systolic >= 140 or diastolic >= 90:
            tips.append("### ❤️‍🔥 High Blood Pressure (Hypertension Stage 2 or Hypertensive Crisis) Suggestions:")
            tips.append("-   URGENT Medical Attention: **Immediately consult a doctor.** This level of blood pressure often requires medication and significant lifestyle changes.")
            tips.append("-   Do NOT Self-Treat: Do not ignore these readings. Follow your doctor's instructions meticulously.")
            tips.append("-   Emergency if Symptoms: If these readings are accompanied by symptoms like chest pain, severe headache, shortness of breath, or numbness/weakness, **seek emergency medical care immediately.**")
            tips.append("---")

    # General Questionnaire-based tips (these remain valuable irrespective of BP category)
    tips.append("### General Health Tips based on Questionnaire:")

    if age >= 50:
        tips.append("-   Age Factor: Regular blood pressure monitoring becomes even more critical after age 50. Discuss screening frequency with your doctor.")
    if diet == "Unhealthy":
        tips.append("-   Diet Improvement: Adopting a healthier diet rich in fruits, vegetables, and lean proteins is vital. Limit processed foods, unhealthy fats, and added sugars.")
    elif diet == "Average":
        tips.append("-   Diet Tweaks: Small improvements like reducing sugary drinks and increasing fiber can significantly impact your health.")
    if salt_intake == "High":
        tips.append("-   Sodium Reduction: Be mindful of hidden salts in processed foods. Cooking at home allows for better control of sodium intake.")
    if exercise in ["Rarely", "Never"]:
        tips.append("-   Increase Activity: Start with short walks and gradually increase duration and intensity. Aim for at least 30 minutes of moderate activity most days.")
    elif exercise == "Few times a week":
        tips.append("-   Boost Exercise: Consider increasing the duration or intensity of your workouts, or try new activities to stay motivated.")
    if smoker == "Yes":
        tips.append("-   Quit Smoking: Smoking damages blood vessels and significantly increases heart disease risk. Quitting is the best gift you can give your health.")
    if alcohol == "Yes":
        tips.append("-   Moderate Alcohol: If you drink, do so in moderation (up to one drink per day for women, two for men). Excessive alcohol raises blood pressure.")
    
    if "Hypertension" in prev_conditions:
        tips.append("-   Manage Existing Hypertension: Continue to diligently follow your doctor's treatment plan for hypertension, including medication and lifestyle changes.")
    if "Diabetes" in prev_conditions:
        tips.append("-   Diabetes Management: Tightly control your blood sugar levels, as diabetes is a major risk factor for cardiovascular complications.")
    if "Heart Disease" in prev_conditions:
        tips.append("-   Cardiologist Consults: Maintain regular follow-ups with your cardiologist and adhere strictly to your prescribed medications and lifestyle regimen.")
    
    # Final general wellness tips
    tips.append("---")
    tips.append("### Overall Wellness Reminders:")
    tips.append("-   Stress Management: Practice relaxation techniques such as deep breathing, meditation, or yoga to help manage stress levels, which can impact BP.")
    tips.append("-   Quality Sleep: Aim for 7-9 hours of consistent, good-quality sleep each night. Poor sleep can contribute to high blood pressure.")
    tips.append("-   Regular Check-ups: Always prioritize regular visits to your healthcare provider for comprehensive health assessments and personalized advice.")
    tips.append("-   Seek Professional Advice: Remember, this tool is for informational purposes. For any health concerns or before making changes to your treatment plan, consult a qualified medical professional.")

    return tips

# Example of how to use this service module:
if __name__ == "__main__":
    # --- Step 1: Load the ML assets (do this once when your app starts) ---
    if load_ml_assets():
        print("\n--- Example Prediction ---")
        # --- Step 2: Prepare user data (from your web app's forms/inputs) ---
        sample_user_data = {
            'age': 40,
            'gender': 'Female',
            'diet': 'Healthy',
            'salt_intake': 'Moderate',
            'exercise': 'Daily',
            'smoker': 'No',
            'alcohol': 'No',
            'prev_conditions': ['None'],
            # If your app collects height/weight/cholesterol/gluc directly, add them here
            'height': 165,
            'weight': 68,
            'cholesterol': 1,
            'gluc': 1
        }

        # --- Step 3: Simulate/Extract Computer Vision Features ---
        # In a real web app, this would come from image/video analysis.
        # For demonstration, we'll simulate them here.
        simulated_cv_features = {
            'FacialRednessIndex': 0.55,
            'EyeAreaRatio': 0.035,
            'SkinToneVariability': 0.004,
            'EstimatedHeartRate_CV': 70,
            'PPG_SignalNoiseRatio': 22.0
        }

        # --- Step 4: Make a prediction ---
        systolic_bp, diastolic_bp = predict_blood_pressure(sample_user_data, simulated_cv_features)

        if systolic_bp is not None and diastolic_bp is not None:
            print(f"Predicted Blood Pressure: {int(systolic_bp)}/{int(diastolic_bp)} mmHg")

            # --- Step 5: Generate tips ---
            tips = generate_tips(
                sample_user_data['age'],
                sample_user_data['diet'],
                sample_user_data['salt_intake'],
                sample_user_data['exercise'],
                sample_user_data['smoker'],
                sample_user_data['alcohol'],
                sample_user_data['prev_conditions'],
                systolic_bp,
                diastolic_bp
            )
            print("\n--- Personalized Health Tips ---")
            for tip in tips:
                print(tip)
        else:
            print("Failed to predict blood pressure.")
    else:
        print("Failed to load ML assets. Prediction service not available.")