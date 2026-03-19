import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder

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

LEARNING_RATE = 0.09
MAX_DEPTH = 4
N_ESTIMATORS = 120


if __name__ == '__main__':
    train_data = pd.read_csv('data/train_processed.csv')
    train_data.fillna('', inplace=True)

    X_train = train_data[ALL_FEATURES]
    y_train = train_data['Premium Amount']

    label_encoder = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        label_encoder[col] = le

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    for col in X_train.columns:
        X_train[col] = np.log1p(X_train[col])

    y_train = np.log1p(y_train)

    model = LGBMRegressor(
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        n_estimators=N_ESTIMATORS
    )
    model.fit(X_train, y_train)

    with open('lgbm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model training complete. Artifacts saved: lgbm_model.pkl, label_encoder.pkl")
