from src.train import model_building

def test_model_building_initialization(training_sample_data):
    x_train = training_sample_data.drop(columns = ['Churn'])
    y_train = training_sample_data['Churn'].map({'Yes':1,
                                        'No':0})
    trainer = model_building(x_train,y_train)
    assert trainer.x_train.equals(x_train)
    assert trainer.y_train.equals(y_train)

def test_training_returns_models(training_sample_data,monkeypatch):
    x_train = training_sample_data.drop(columns = ['Churn'])
    y_train = training_sample_data['Churn'].map({'Yes':1,
                                        'No':0})
    trainer = model_building(x_train,y_train)
    monkeypatch.setattr("src.train.mlflow.set_tracking_uri", lambda *args, **kwargs:None)
    monkeypatch.setattr("src.train.mlflow.set_experiment", lambda *args, **kwargs:None)

    class FakeRun:
        def __enter__(self):
            self.info = type("Info",(),{"run_id":"test_run_id"})()
            return self
        def __exit__(self,exc_type,exc_value,traceback):
            return False
    monkeypatch.setattr("src.train.mlflow.start_run", lambda *args,**kwargs:FakeRun())
    monkeypatch.setattr("src.train.mlflow.log_params", lambda *args,**kwargs:None)
    monkeypatch.setattr("src.train.mlflow.log_metric", lambda *args,**kwargs:None)

    models,run_ids = trainer.training()
    assert isinstance(models,dict)
    assert len(models) == 3
    assert set(models.keys()) == {"logistic", "rf", "xgb"}
    assert set(run_ids.keys()) == {"logistic", "rf", "xgb"}
    assert all(run_id == "test_run_id" for run_id in run_ids.values())
