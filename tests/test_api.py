from fastapi.testclient import TestClient
import src.app as app_module

client = TestClient(app_module.app)
def test_predict_api_sucess(monkeypatch,valid_customer_payload):
    monkeypatch.setattr(app_module.predict_model,"predict",lambda customer_data:{"prediction":1})
    response = client.post("/predict",json = valid_customer_payload)
    assert response.status_code == 200
    assert response.json() == {"prediction":1}

def test_predict_invalid_input(valid_customer_payload):
    payload = valid_customer_payload.copy()
    payload['gender'] = "Other"
    response = client.post("/predict",json = payload)
    assert response.status_code == 422