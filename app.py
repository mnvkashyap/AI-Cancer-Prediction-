from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask App
app = Flask(__name__)

# Load Dataset
cancer = pd.read_csv(
    'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv'
)

# Feature Names
features = list(
    cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns
)

# Load Model & Scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# PREDICT PAGE
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None

    if request.method == 'POST':

        values = []

        for feature in features:
            values.append(float(request.form[feature]))

        input_data = pd.DataFrame(
            [values],
            columns=features
        )

        # Scale Input
        input_scaled = scaler.transform(input_data)

        # Predict
        result = model.predict(input_scaled)[0]

        if result == 'M':
            prediction = 'Malignant Cancer Detected'
        else:
            prediction = 'Benign Tumor Detected'

    return render_template(
        'predict.html',
        features=features,
        prediction=prediction
    )

# CSV UPLOAD PAGE
@app.route('/upload', methods=['GET', 'POST'])
def upload():

    table = None

    if request.method == 'POST':

        file = request.files['file']

        if file:

            df = pd.read_csv(file)

            scaled_data = scaler.transform(df)

            predictions = model.predict(scaled_data)

            df['Prediction'] = predictions

            table = df.head().to_html(
                classes='table table-dark'
            )

    return render_template(
        'upload.html',
        table=table
    )

# BREAST HEALTH ASSESSMENT
@app.route('/assessment', methods=['GET', 'POST'])
def assessment():

    if request.method == 'POST':

        score = 0

        lump = request.form.get('lump')
        pain = request.form.get('pain')
        discharge = request.form.get('discharge')
        skin = request.form.get('skin')
        swelling = request.form.get('swelling')
        family = request.form.get('family')
        age = int(request.form.get('age'))

        if lump == 'yes':
            score += 4

        if pain == 'yes':
            score += 2

        if discharge == 'yes':
            score += 3

        if skin == 'yes':
            score += 2

        if swelling == 'yes':
            score += 2

        if family == 'yes':
            score += 2

        if age > 45:
            score += 2

        if score >= 10:
            risk = 'High Risk'
            message = 'Please consult a healthcare professional.'

        elif score >= 5:
            risk = 'Moderate Risk'
            message = 'Some symptoms may require medical attention.'

        else:
            risk = 'Low Risk'
            message = 'Continue regular self-checks.'

        return render_template(
            'result.html',
            risk=risk,
            score=score,
            message=message
        )

    return render_template('assessment.html')

# RUN APP
if __name__ == '__main__':
    app.run(debug=True)
