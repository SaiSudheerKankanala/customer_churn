import pytest
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture
def sample_data():
    data = {
        "customerID": ["C001", "C002", "C003", "C004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", "Yes"],
        "tenure": [12, 24, 36, 48],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "OnlineSecurity": ["Yes", "No", "No", "Yes"],
        "OnlineBackup": ["No", "Yes", "No", "Yes"],
        "DeviceProtection": ["Yes", "No", "Yes", "No"],
        "TechSupport": ["No", "Yes", "No", "Yes"],
        "StreamingTV": ["Yes", "No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No", "Yes"],
        "Contract": ["One year", "Month-to-month", "Two year", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        "MonthlyCharges": [50.5, 70.0, 85.5, 95.0],
        "TotalCharges": [606.0, 1680.0, 3078.0, 4560.0],
        "Churn": ["No", "Yes", "No", "Yes"]
    }

    return pd.DataFrame(data)


@pytest.fixture
def training_sample_data():
    data = {
        "customerID": [f"C{i:03d}" for i in range(1, 41)],
        "gender": ["Male", "Female"] * 20,
        "SeniorCitizen": [0, 1] * 20,
        "Partner": ["Yes", "No"] * 20,
        "Dependents": ["No", "Yes"] * 20,
        "tenure": list(range(1, 41)),
        "PhoneService": ["Yes", "No"] * 20,
        "MultipleLines": ["No", "Yes"] * 20,
        "InternetService": ["DSL", "Fiber optic"] * 20,
        "OnlineSecurity": ["Yes", "No"] * 20,
        "OnlineBackup": ["No", "Yes"] * 20,
        "DeviceProtection": ["Yes", "No"] * 20,
        "TechSupport": ["No", "Yes"] * 20,
        "StreamingTV": ["Yes", "No"] * 20,
        "StreamingMovies": ["No", "Yes"] * 20,
        "Contract": ["Month-to-month", "One year"] * 20,
        "PaperlessBilling": ["Yes", "No"] * 20,
        "PaymentMethod": ["Electronic check", "Mailed check"] * 20,
        "MonthlyCharges": [50.0, 80.0] * 20,
        "TotalCharges": [100.0, 200.0] * 20,
        "Churn": ["Yes", "No"] * 20
    }

    return pd.DataFrame(data)



@pytest.fixture
def sample_features(sample_data):
    return sample_data.drop(columns = ['Churn'])


@pytest.fixture
def valid_customer_payload():
    return {"gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.5,
        "TotalCharges": 606.0}

@pytest.fixture
def fake_model():
    model = MagicMock()
    model.predict.return_value = [1]
    model.predict_proba.return_value = [[0.2, 0.8]]

    return model

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path