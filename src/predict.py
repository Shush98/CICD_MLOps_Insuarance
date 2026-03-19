import pickle
import numpy as np
import pandas as pd

MODEL_PATH = 'lgbm_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'

CATEGORICAL_FEATURES = [
    'Gender', 'Marital Status', 'Number of Dependents', 'Education Level',
    'Occupation', 'Location', 'Policy Type', 'Previous Claims',
    'Insurance Duration', 'Customer Feedback', 'Smoking Status',
    'Exercise Frequency', 'Property Type', 'Policy Age'
]

ALL_FEATURES = [
    'Gender', 'Marital Status', 'Number of Dependents', 'Education Level',
    'Occupation', 'Location', 'Policy Type', 'Previous Claims',
    'Vehicle Age', 'Insurance Duration', 'Customer Feedback',
    'Smoking Status', 'Exercise Frequency', 'Property Type', 'Policy Age',
    'Age', 'Annual Income', 'Health Score', 'Credit Score'
]


def _load_artifacts():
    """Load model and label encoder from disk."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder


def predict(features: dict) -> float:
    """
    Predict insurance premium given a dictionary of input features.

    Args:
        features: dict with keys matching ALL_FEATURES

    Returns:
        Predicted premium amount (float, original scale)
    """
    model, label_encoder = _load_artifacts()

    df = pd.DataFrame([features])[ALL_FEATURES]

    # Apply label encoding to categorical columns
    for col in CATEGORICAL_FEATURES:
        le = label_encoder[col]
        df[col] = le.transform(df[col])

    # Apply log1p transformation (same as training)
    for col in df.columns:
        df[col] = np.log1p(df[col])

    # Predict (model output is log-transformed, so reverse with expm1)
    log_prediction = model.predict(df)[0]
    prediction = np.expm1(log_prediction)

    return round(float(prediction), 2)
