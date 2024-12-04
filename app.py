from flask import Flask, render_template, request
import joblib
import numpy as np

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
        age = int(request.form['age'])
        gender = request.form['gender']

        return render_template('form.html', name=name, age=age, gender=gender)
    except Exception as e:
        print(f"Error in form submission: {e}")
        return render_template('error.html', error_message="Error in form submission. Please check the input.")

# صفحة التنبؤ بناءً على البيانات المدخلة من صفحة form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات من صفحة form
        name = request.form['name']
        age = int(request.form['age'])
        gender = int(request.form['gender'])

        # طباعة البيانات المستلمة
        print("Received Data:", request.form)

        # الحصول على قيم المدخلات من نموذج form
        input_data = []
        fields = [
            'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
                # إضافة age و gender إلى بيانات الإدخال
        input_data.append(age)   # إضافة العمر
        input_data.append(gender)  # إضافة الجنس

        print("Processed Input Data:", input_data)

        # التحقق من صحة المدخلات
        for field in fields:
            value = request.form.get(field)
            if value:
                try:
                    if field == 'oldpeak':  # النوع العشري
                        input_data.append(float(value))
                    else:  # الأنواع الصحيحة
                        input_data.append(int(value))
                except ValueError:
                    raise ValueError(f"Invalid value for {field}: {value}. Please ensure the value is numeric.")
            else:
                raise ValueError(f"Missing value for {field}. Please complete all fields.")



        # تحويل البيانات باستخدام المقياس
        input_data = np.array([input_data])
        input_data_scaled = scaler.transform(input_data)  # التحجيم باستخدام المقياس الذي تم تحميله

        # طباعة البيانات بعد التحجيم
        print("Scaled Data:", input_data_scaled)

        # التنبؤ
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0]

        # طباعة نتائج التنبؤ
        print("Prediction:", prediction)
        print("Probability:", probability)

        # تحديد النتيجة
        if prediction == 1:
            result = f"The patient has a {probability[1] * 100:.2f}% probability of having heart disease."
            status = "positive"
        else:
            result = f"The patient has a {probability[0] * 100:.2f}% probability of not having heart disease."
            status = "negative"

        return render_template('result.html', name=name, result=result, status=status)

    except ValueError as ve:
        print(f"Input validation error: {ve}")
        return render_template('error.html', error_message=f"Input Error: {ve}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('error.html', error_message=f"An error occurred: {e}")



if __name__ == '__main__':
    app.run(debug=True)
