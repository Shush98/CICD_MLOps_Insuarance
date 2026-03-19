from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from predict import predict

app = FastAPI(title="Insurance Premium Prediction API")


class InsuranceInput(BaseModel):
    Gender: str
    Marital_Status: str
    Number_of_Dependents: float
    Education_Level: str
    Occupation: str
    Location: str
    Policy_Type: str
    Previous_Claims: float
    Vehicle_Age: float
    Insurance_Duration: float
    Customer_Feedback: str
    Smoking_Status: str
    Exercise_Frequency: str
    Property_Type: str
    Policy_Age: int
    Age: float
    Annual_Income: float
    Health_Score: float
    Credit_Score: float


@app.get("/")
def home():
    return {"message": "Insurance Premium Prediction API is running"}


@app.post("/predict")
def get_prediction(data: InsuranceInput):
    try:
        features = {
            "Gender": data.Gender,
            "Marital Status": data.Marital_Status,
            "Number of Dependents": data.Number_of_Dependents,
            "Education Level": data.Education_Level,
            "Occupation": data.Occupation,
            "Location": data.Location,
            "Policy Type": data.Policy_Type,
            "Previous Claims": data.Previous_Claims,
            "Vehicle Age": data.Vehicle_Age,
            "Insurance Duration": data.Insurance_Duration,
            "Customer Feedback": data.Customer_Feedback,
            "Smoking Status": data.Smoking_Status,
            "Exercise Frequency": data.Exercise_Frequency,
            "Property Type": data.Property_Type,
            "Policy Age": data.Policy_Age,
            "Age": data.Age,
            "Annual Income": data.Annual_Income,
            "Health Score": data.Health_Score,
            "Credit Score": data.Credit_Score,
        }
        prediction = predict(features)
        return {"predicted_premium": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
