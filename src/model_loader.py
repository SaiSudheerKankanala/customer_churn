import joblib
from pathlib import Path


def load_model_artifact(artifact_path):
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise TypeError("Loaded artifact must be a dictionary")

    required_keys = {"model", "background_data"}
    if not required_keys.issubset(artifact.keys()):
        raise ValueError("Artifact must contain model and background_data")

    model = artifact["model"]
    if not hasattr(model, "predict"):
        raise TypeError("Loaded model is not a valid prediction model")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba")

    background_data = artifact["background_data"]

    return model, background_data
