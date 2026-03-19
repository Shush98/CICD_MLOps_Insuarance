import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.model_selection import train_test_split

DATA_URL = 'https://raw.githubusercontent.com/Shush98/MLOps_Insurance_premium/refs/heads/main/data/raw.csv'


def load_data():
    df = pd.read_csv(DATA_URL)
    return df


def handle_missing_values(df):
    df['Annual Income'].fillna(df['Annual Income'].mean(), inplace=True)
    df['Marital Status'].fillna('Single', inplace=True)
    df['Number of Dependents'].fillna(0.0, inplace=True)
    df['Occupation'].fillna('Unknown', inplace=True)
    df['Health Score'].fillna(df['Health Score'].mean(), inplace=True)
    df['Previous Claims'].fillna(0.0, inplace=True)
    df['Credit Score'].fillna(df['Credit Score'].mean(), inplace=True)
    df['Customer Feedback'].fillna('Average', inplace=True)
    df.dropna(how='any', inplace=True)
    return df


def calculate_policy_age(df):
    def age_from_date(born):
        born = str(born)
        born = datetime.strptime(born, "%Y-%m-%d %H:%M:%S.%f").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    df['Policy Age'] = df['Policy Start Date'].apply(age_from_date)
    df.drop(columns=['Policy Start Date'], inplace=True)
    return df


def remove_outliers(df):
    Q1 = df['Premium Amount'].quantile(0.25)
    Q3 = df['Premium Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    upper_array = df.index[df['Premium Amount'] >= upper]
    lower_array = df.index[df['Premium Amount'] <= lower]
    df.drop(index=upper_array.union(lower_array), inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':
    df = load_data()
    df = handle_missing_values(df)
    df = calculate_policy_age(df)
    df = remove_outliers(df)

    train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)

    train_data.to_csv('data/train_processed.csv', index=False)
    test_data.to_csv('data/test_processed.csv', index=False)

    print("Preprocessing complete. Files saved to data/")
