from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Modeli yükle
with open("model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan gelen veriler
        features = [float(request.form[feature]) for feature in ['concave_points_worst', 'perimeter_worst',
                                                                 'concave_points_mean', 'radius_worst',
                                                                 'perimeter_mean']]
        # Veriyi normalize et ve tahmin yap
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        result = "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


