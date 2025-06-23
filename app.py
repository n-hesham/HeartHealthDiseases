from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import logging
import sklearn

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Model Paths ---
MODEL_PATH = r"model.pkl"
SCALER_PATH = r"scaler.pkl"

# --- Expected Features in Correct Order ---
EXPECTED_FEATURE_NAMES = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

# --- Load Model and Scaler ---
try:
    app.logger.info("Flask app starting...")
    app.logger.info(f"Scikit-learn version: {sklearn.__version__}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    app.logger.info("Model and scaler loaded successfully.")
    # If version warnings reappear here, the .pkl files are still old!
except FileNotFoundError:
    app.logger.error(f"Model or scaler not found. Paths: {MODEL_PATH}, {SCALER_PATH}")
    model = None
    scaler = None
except Exception as e:
    app.logger.error(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# --- Routes ---

@app.route('/')
def index():
    """Render the input form (and optionally, the result)."""
    # Pass show_result=False explicitly for the initial GET request
    return render_template('index.html', show_result=False)


@app.route('/predict', methods=['POST'])
def predict():
    # Default context for rendering template on error or success
    render_context = {'show_result': False}

    if not model or not scaler:
        app.logger.error("Model or scaler not loaded.")
        render_context['error'] = "Model or Scaler failed to load."
        return render_template('index.html', **render_context)

    form_data = request.form
    patient_name = form_data.get('name', 'N/A')
    render_context['name'] = patient_name # Add name to context early

    try:
        app.logger.info(f"Prediction request for: {patient_name}")
        input_values = {}
        # Preserve entered form data for re-rendering if needed
        for feature in EXPECTED_FEATURE_NAMES + ['name']: # Include name
             render_context[f'form_{feature}'] = form_data.get(feature, '')

        for feature in EXPECTED_FEATURE_NAMES:
            value = form_data.get(feature)
            if value is None or value.strip() == '':
                raise ValueError(f"Missing value for '{feature}'.")
            try:
                # Use float for Oldpeak, int for others based on your form/data
                input_values[feature] = float(value) if feature == 'Oldpeak' else int(value)
            except ValueError:
                # Raise specific error for bad numeric conversion
                raise ValueError(f"Invalid non-numeric value entered for '{feature}': '{value}'")

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_values], columns=EXPECTED_FEATURE_NAMES)

        # Scale features
        scaled_features = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Calculate prediction probability (optional)
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            try:
                # predict_proba usually returns [[prob_class_0, prob_class_1]]
                prediction_proba = model.predict_proba(scaled_features)[0]
            except Exception as e:
                app.logger.warning(f"Could not calculate prediction probability: {e}")

        # --- CORRECT INDENTATION STARTS HERE ---
        # These lines MUST be inside the main 'try' block
        result_code = prediction[0] # Get the single prediction result (0 or 1)
        result_text = "Heart Disease Detected" if result_code == 1 else "No Heart Disease Detected"
        status_class = "positive" if result_code == 1 else "negative"

        # Update context for successful prediction
        render_context.update({
            'result': result_text,
            'status_class': status_class,
            'probability': prediction_proba,
            'show_result': True # Flag to show the result section
        })
        app.logger.info(f"Prediction for {patient_name}: {result_text}")
        return render_template('index.html', **render_context)
        # --- CORRECT INDENTATION ENDS HERE ---

    except ValueError as ve:
        # Handle specific input validation errors
        app.logger.error(f"Input Error for {patient_name}: {ve}")
        render_context['error'] = str(ve) # Pass error message to template
        # Re-render form with error and sticky values
        return render_template('index.html', **render_context)

    except Exception as e:
        # Handle unexpected errors during scaling or prediction
        app.logger.error(f"Prediction failed for {patient_name}: {e}", exc_info=True) # Log traceback
        render_context['error'] = "An unexpected error occurred during prediction." # User-friendly message
        # Re-render form with error and sticky values
        return render_template('index.html', **render_context)


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)