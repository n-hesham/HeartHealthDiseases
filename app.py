from flask import Flask, render_template, request
import joblib
import numpy as np
import logging



logging.basicConfig(level=logging.ERROR)

# تحميل النموذج والمقياس
model_path = r"E:\Project\HeartHealthDiseases\model.pkl"
scaler_path = r"E:\Project\HeartHealthDiseases\scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    raise Exception("Error loading model or scaler. Please check the file paths.")

app = Flask(__name__)

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html')

# صفحة form (لإدخال بيانات التحاليل)
@app.route('/form', methods=['POST'])
def form():
    try:
        # استلام البيانات من صفحة index
        name = request.form['name']
        age = int(request.form['Age'])
        gender = int(request.form['Sex'])

        return render_template('form.html', name=name, age=age, gender=gender)
    except Exception as e:
        logging.error(f"Error in form submission: {e}")
        return render_template('error.html', error_message="Error in form submission. Please check the input.")

# صفحة التنبؤ بناءً على البيانات المدخلة من صفحة form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        age = request.form.get('Age')
        if not age:
            raise ValueError("Age is missing or empty.")
        age = int(age)

        gender = request.form.get('Sex')
        if not gender:
            raise ValueError("Gender is missing or empty.")
        gender = int(gender)

        chest_pain_type = request.form.get('ChestPainType')
        if not chest_pain_type:
            raise ValueError("Chest Pain Type is missing or empty.")
        chest_pain_type = int(chest_pain_type)

        resting_bp = request.form.get('RestingBP')
        if not resting_bp:
            raise ValueError("Resting BP is missing or empty.")
        resting_bp = int(resting_bp)

        cholesterol = request.form.get('Cholesterol')
        if not cholesterol:
            raise ValueError("Cholesterol is missing or empty.")
        cholesterol = int(cholesterol)

        fasting_bs = request.form.get('FastingBS')
        if not fasting_bs:
            raise ValueError("Fasting BS is missing or empty.")
        fasting_bs = int(fasting_bs)

        resting_ecg = request.form.get('RestingECG')
        if not resting_ecg:
            raise ValueError("Resting ECG is missing or empty.")
        resting_ecg = int(resting_ecg)

        max_hr = request.form.get('MaxHR')
        if not max_hr:
            raise ValueError("Max HR is missing or empty.")
        max_hr = int(max_hr)

        exercise_angina = request.form.get('ExerciseAngina')
        if not exercise_angina:
            raise ValueError("Exercise Angina is missing or empty.")
        exercise_angina = int(exercise_angina)

        oldpeak = request.form.get('Oldpeak')
        if not oldpeak:
            raise ValueError("Oldpeak is missing or empty.")
        oldpeak = float(oldpeak)

        st_slope = request.form.get('ST_Slope')
        if not st_slope:
            raise ValueError("ST_Slope is missing or empty.")
        st_slope = int(st_slope)

        # Combine inputs into a NumPy array for prediction
        features = np.array([[age, gender, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                               resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Interpret prediction
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
        status = "positive" if prediction[0] == 1 else "negative"

        # Render the result page
        return render_template('result.html',status=status, result=result)

    except ValueError as ve:
        logging.error(f"Input validation error: {ve}")
        return render_template('error.html', error_message=f"Input Error: {ve}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('error.html', error_message=f"An error occurred: {e}")




if __name__ == '__main__':
    app.run(debug=True)
