
import pandas as pd

from src.config import artifacts_dir
from src.model_loader import load_model_artifact
from src.explainability import ChurnExplainer


class prediction:

    def __init__(self):

        self.model, self.background_data = load_model_artifact(
            artifacts_dir / "best_model.pkl"
        )

        self.explainer = ChurnExplainer(
            self.model,
            self.background_data
        )

    def predict(self, data):

        input_data = pd.DataFrame(
            [data],
            columns=self.background_data.columns
        )

        prediction_result = int(
            self.model.predict(input_data)[0]
        )

        explanation = self.explainer.explain(data)

        return {

            "prediction": prediction_result,

            "churn_probability": explanation["churn_probability"],

            "explanation": explanation["explanations"][:5]

        }
