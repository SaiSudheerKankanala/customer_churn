import pandas as pd
import pytest
from src.data_injection import data_injection

def test_load_data_sucessfully(monkeypatch,sample_data):
    monkeypatch.setattr("src.data_injection.pd.read_csv",lambda path:sample_data.copy())
    loader = data_injection()
    result = loader.load_data()
    assert result.equals(sample_data)
    assert not result.empty
    assert "Churn" in result.columns

def test_empty_dataset(monkeypatch):
    monkeypatch.setattr("src.data_injection.pd.read_csv", lambda path: pd.DataFrame())
    loader = data_injection()
    with pytest.raises(ValueError,match = 'Data is empty'):
        loader.load_data()

def test_missing_target_columns(sample_data,monkeypatch):
    data = sample_data.drop(columns = ['Churn'])
    monkeypatch.setattr("src.data_injection.pd.read_csv", lambda path:data)
    loader = data_injection()
    with pytest.raises(ValueError,match = "dependent variable is not available in data"):
        loader.load_data()