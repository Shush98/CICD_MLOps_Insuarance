from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

SAMPLE_INPUT = {
    "Gender": "Male",
    "Marital_Status": "Single",
    "Number_of_Dependents": 0.0,
    "Education_Level": "Bachelor's",
    "Occupation": "Employed",
    "Location": "Urban",
    "Policy_Type": "Basic",
    "Previous_Claims": 0.0,
    "Vehicle_Age": 5.0,
    "Insurance_Duration": 2.0,
    "Customer_Feedback": "Good",
    "Smoking_Status": "No",
    "Exercise_Frequency": "Monthly",
    "Property_Type": "House",
    "Policy_Age": 3,
    "Age": 30.0,
    "Annual_Income": 60000.0,
    "Health_Score": 25.0,
    "Credit_Score": 600.0
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_returns_200():
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200


def test_predict_endpoint_returns_premium():
    response = client.post("/predict", json=SAMPLE_INPUT)
    data = response.json()
    assert "predicted_premium" in data
    assert isinstance(data["predicted_premium"], float)
    assert data["predicted_premium"] > 0


def test_predict_missing_field_returns_422():
    incomplete_input = {k: v for k, v in SAMPLE_INPUT.items() if k != "Age"}
    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422
