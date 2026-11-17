import pytest
from pydantic import ValidationError
from src.app import Customers

@pytest.mark.parametrize("field, value", [
    ("gender", "Other"),
    ("SeniorCitizen", 2),
    ("Partner", "Maybe"),
    ("Dependents", "Unknown"),
    ("PhoneService", "Maybe"),
    ("InternetService", "Satellite"),
    ("Contract", "Monthly"),
    ("PaperlessBilling", "Unknown"),
    ("PaymentMethod", "Cash"),
])

def test_invalid_literal_values(valid_customer_payload,field,value):
    payload = valid_customer_payload.copy()
    payload[field] = value
    with pytest.raises(ValidationError):
        Customers(**payload) 

@pytest.mark.parametrize("field", [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
])
def test_missing_required_field(valid_customer_payload, field):
    payload = valid_customer_payload.copy()
    del payload[field]

    with pytest.raises(ValidationError):
        Customers(**payload)

def test_invalid_tenure_type(valid_customer_payload):
    payload = valid_customer_payload.copy()
    payload["tenure"] = "twelve"

    with pytest.raises(ValidationError):
        Customers(**payload)