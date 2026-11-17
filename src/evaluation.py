
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

import mlflow

from src.config import (
    thresholds,
    min_recall,
    max_recall,
    logger
)


class Evaluation:

    def __init__(
        self,
        x_val,
        y_val,
        x_test,
        y_test,
        trained_models,
        run_ids
    ):
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.trained_models = trained_models
        self.run_ids = run_ids

    def evaluating_steps(self, x, y, model, threshold=0.5):

        probabilities = model.predict_proba(x)[:, 1]

        y_pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)

        return {
            'acc': accuracy_score(y, y_pred),
            'recall': recall_score(y, y_pred, zero_division=0),
            'precision': precision_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'specificity':specificity
        }

    def evaluating_models(self):

        validated_models = {}
        for name, model in self.trained_models.items():
            validated_models[name] = {}

            for threshold in thresholds:
                metrics = self.evaluating_steps( self.x_val, self.y_val, model, threshold)
                validated_models[name][threshold] = metrics

                # Resume the respective model's MLflow run
                with mlflow.start_run(
                    run_id=self.run_ids[name]
                ):

                    mlflow.log_metrics({
                        f"val_{key}_threshold_{str(threshold).replace('.', '_')}": float(value)
                        for key, value in metrics.items()
                    })

                logger.info(
                    f"Model: {name}, "
                    f"Threshold: {threshold}, "
                    f"Metrics: {metrics}"
                )
        eligible_model = []

        for name, threshold_result in validated_models.items():

            for threshold, metrics in threshold_result.items():

                recall = metrics['recall']

                if recall >= min_recall and recall < max_recall:

                    eligible_model.append({
                        'name': name,
                        'threshold': threshold,
                        'metrics': metrics
                    })

                    logger.info(
                        f"Eligible Model: {name}, "
                        f"Threshold: {threshold}, "
                        f"Recall: {recall}"
                    )

        logger.info(
            f"Total eligible models found: {len(eligible_model)}"
        )

        if not eligible_model:

            logger.warning(
                f"No eligible models found above "
                f"{min_recall * 100}% recall"
            )

            raise ValueError(
                "No model satisfies the minimum recall requirement"
            )

        # Best model selection
        best_candidate = max(
            eligible_model,
            key=lambda candidate: (
                candidate['metrics']['specificity'],
                candidate['metrics']['precision'],
                candidate['metrics']['acc'],
                candidate['metrics']['f1']
            )
        )

        best_model_name = best_candidate['name']

        best_model_threshold = best_candidate['threshold']

        best_model = self.trained_models[best_model_name]

        logger.info(
            f"Best model selected: {best_model_name}"
        )

        logger.info(
            f"Best threshold selected: {best_model_threshold}"
        )

        logger.info(
            f"Best validation metrics: {best_candidate['metrics']}"
        )

        # Final test evaluation
        logger.info(
            "Starting final evaluation on test data"
        )

        test_metrics = self.evaluating_steps(
            self.x_test,
            self.y_test,
            best_model,
            best_model_threshold
        )

        # Log final test metrics in the selected model's run
        with mlflow.start_run(
            run_id=self.run_ids[best_model_name]
        ):

            mlflow.log_param(
                'best_model_name',
                best_model_name
            )

            mlflow.log_param(
                'best_threshold',
                float(best_model_threshold)
            )

            mlflow.log_metrics({
                f"test_{key}": float(value)
                for key, value in test_metrics.items()
            })

        logger.info(
            f"Final test metrics: {test_metrics}"
        )

        return best_model_name, test_metrics
