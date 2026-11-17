
import numpy as np
import pandas as pd

from src.predict import prediction


def create_predictor(fake_model, sample_features, monkeypatch):

    background_data = sample_features
    monkeypatch.setattr(
        "src.predict.load_model_artifact",
        lambda path: (fake_model, background_data)
    )
    class FakeExplainer:

        def __init__(self, model, background_data):
            self.model = model

        def explain(self, data):
            return {
                "churn_probability": 0.8,
                "explanations": []
            }

    monkeypatch.setattr(
        "src.predict.ChurnExplainer",
        FakeExplainer
    )

    return prediction()


def test_prediction_returns_output(fake_model, sample_features, monkeypatch):

    predictor = create_predictor(
        fake_model,
        sample_features,
        monkeypatch
    )

    customer_data = sample_features.iloc[0].to_dict()

    result = predictor.predict(customer_data)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "churn_probability" in result
    assert "explanation" in result
    assert result["prediction"] == 1


def test_prediction_probability(fake_model, sample_features, monkeypatch):

    predictor = create_predictor(
        fake_model,
        sample_features,
        monkeypatch
    )

    customer_data = sample_features.iloc[0].to_dict()

    result = predictor.predict(customer_data)

    probability = result["churn_probability"]

    assert probability is not None
    assert 0 <= probability <= 1


def test_prediction_calls_model(fake_model, sample_features, monkeypatch):

    predictor = create_predictor(
        fake_model,
        sample_features,
        monkeypatch
    )

    customer_data = sample_features.iloc[0].to_dict()

    predictor.predict(customer_data)

    fake_model.predict.assert_called_once()

    called_data = fake_model.predict.call_args[0][0]

    assert isinstance(called_data, pd.DataFrame)
    assert called_data.shape == (1, 20)
