
import joblib
from unittest.mock import MagicMock 
from src.model_loader import load_model_artifact

def test_model_loading_successfully(monkeypatch,tmp_path):
    fake_model = MagicMock()
    background_data = MagicMock()

    artifact = {"model":fake_model,
                 "background_data":background_data}
    model_path = tmp_path/"model.pkl"
    model_path.touch()
    monkeypatch.setattr(joblib,'load', lambda path:artifact)
    model,background = load_model_artifact(model_path)
    assert model is fake_model
    assert background is background_data