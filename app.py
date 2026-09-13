"""Run with: python app.py"""
from pathlib import Path
import math
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType
from preprocessing import CATEGORIES, FEATURES, NUMERIC

app = Flask(__name__)
pipeline = joblib.load(Path(__file__).resolve().parent / 'model' / 'churn_model.pkl')


def validate_customer(payload):
    if not isinstance(payload, dict):
        raise ValueError('Provide one customer as a JSON object.')
    missing = sorted(set(FEATURES) - payload.keys())
    extra = sorted(payload.keys() - set(FEATURES) - {'customerID'})
    if missing:
        raise ValueError('Missing fields: ' + ', '.join(missing))
    if extra:
        raise ValueError('Unknown fields: ' + ', '.join(extra))
    result = {key: payload[key] for key in FEATURES}
    for name, allowed in CATEGORIES.items():
        value = result[name]
        if isinstance(value, bool) or value not in allowed:
            raise ValueError(f'{name} must be one of {allowed}.')
    result['SeniorCitizen'] = int(result['SeniorCitizen'])
    for name in NUMERIC:
        value = result[name]
        if name == 'TotalCharges' and (value is None or value == ''):
            result[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite, nonnegative number.')
        if name == 'tenure' and value != int(value):
            raise ValueError('tenure must be a whole number of months.')
    if (result['PhoneService'] == 'No') != (result['MultipleLines'] == 'No phone service'):
        raise ValueError('PhoneService and MultipleLines are inconsistent.')
    for name in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if (result['InternetService'] == 'No') != (result[name] == 'No internet service'):
            raise ValueError(f'InternetService and {name} are inconsistent.')
    return result


@app.post('/predict')
def predict():
    try:
        customer = validate_customer(request.get_json())
    except (ValueError, BadRequest, UnsupportedMediaType) as exc:
        return jsonify(error=str(exc)), 400
    frame = pd.DataFrame([customer])
    prediction = int(pipeline.predict(frame)[0])
    positive_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(frame)[0, positive_index])
    return jsonify(prediction='Yes' if prediction else 'No', churn_probability=round(probability, 6))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
