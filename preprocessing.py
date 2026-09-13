"""Reusable feature engineering shared by training and inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORIES = {
    'gender': ['Female', 'Male'], 'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'], 'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    **{name: ['Yes', 'No', 'No internet service'] for name in
       ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']},
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
}
FEATURES = list(CATEGORIES) + NUMERIC
ENGINEERED = ['SupportServicesCount', 'IsNewCustomer']


class CustomerFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.loc[:, FEATURES].copy()
        for column in NUMERIC:
            result[column] = pd.to_numeric(result[column], errors='coerce')
        for column in CATEGORIES:
            result[column] = result[column].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
        support = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        result['SupportServicesCount'] = result[support].eq('Yes').sum(axis=1)
        result['IsNewCustomer'] = np.where(result.tenure.isna(), np.nan, (result.tenure <= 12).astype(int))
        return result
