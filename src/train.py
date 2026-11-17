
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    randomstate,
    Target_column,
    test_size,
    rf_param,
    xgb_param,
    logger,
    MLFLOW_TRACKING_URL,
    MLFLOW_EXPERIMENT_NAME
)

from src.preprocessing import Preprocessing

import mlflow


class data_split:

    def __init__(self, data):
        self.data = data

    def split(self):

        x = self.data.drop(columns=Target_column)

        y = self.data[Target_column].map({
            'Yes': 1,
            'No': 0
        })

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=randomstate,
            stratify=y
        )

        logger.info(
            f"The shape of the x_train is {x_train.shape}, "
            f"The shape of the y_train is {y_train.shape}"
        )

        logger.info(
            f"The shape of the x_test is {x_test.shape}, "
            f"The shape of the y_test is {y_test.shape}"
        )

        x_train_val, x_val, y_train_val, y_val = train_test_split(
            x_train,
            y_train,
            test_size=test_size,
            random_state=randomstate,
            stratify=y_train
        )

        logger.info(
            f"The shape of the x_train_val is {x_train_val.shape}, "
            f"The shape of the y_train_val is {y_train_val.shape}"
        )

        logger.info(
            f"The shape of the x_val is {x_val.shape}, "
            f"The shape of the y_val is {y_val.shape}"
        )

        return (
            x_train_val,
            y_train_val,
            x_val,
            y_val,
            x_test,
            y_test
        )


class model_building:

    def __init__(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train

    def training(self):

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        logger.info("Training the model")

        models = {
            'logistic': LogisticRegression(max_iter=1000),

            'rf': RandomForestClassifier(
                random_state=randomstate
            ),

            'xgb': XGBClassifier(
                random_state=randomstate,
                eval_metric='logloss'
            )
        }

        grid_param = {
            'rf': rf_param,
            'xgb': xgb_param
        }

        trained_models = {}
        run_ids = {}

        for name, model in models.items():

            with mlflow.start_run(run_name=name) as run:

                run_ids[name] = run.info.run_id

                logger.info(f"Training started for {name}")

                # Preprocessing
                preprocessed = Preprocessing(
                    self.x_train
                ).preprocessing()

                steps = [
                    ('preprocessed', preprocessed)
                ]

                # Logistic Regression feature selection
                if name == 'logistic':

                    steps.append((
                        'scaler',
                        RobustScaler()
                    ))

                    steps.append((
                        'feature_selection',
                        RFE(
                            estimator=LogisticRegression(),
                            n_features_to_select=5
                        )
                    ))

                # SMOTE
                steps.append((
                    'smote',
                    SMOTE(random_state=randomstate)
                ))

                # Model
                steps.append((
                    'model',
                    model
                ))

                pipeline = Pipeline(steps)

                # Hyperparameter tuning for RF and XGBoost
                if name in grid_param:

                    cv = StratifiedKFold(
                        n_splits=5,
                        shuffle=True,
                        random_state=randomstate
                    )

                    search = RandomizedSearchCV(
                        estimator=pipeline,

                        param_distributions={
                            f"model__{key}": value
                            for key, value in grid_param[name].items()
                        },

                        scoring='recall',
                        cv=cv,
                        refit=True,
                        n_iter=20,
                        n_jobs=-1,
                        verbose=2
                    )

                    search.fit(
                        self.x_train,
                        self.y_train
                    )

                    trained_models[name] = search.best_estimator_

                    best_params = {
                        key.replace("model__", ""): value
                        for key, value in search.best_params_.items()
                    }

                    mlflow.log_params(best_params)

                    mlflow.log_metric(
                        "best_cv_recall",
                        float(search.best_score_)
                    )

                    logger.info(
                        f"{name} best parameters: {best_params}"
                    )

                    logger.info(
                        f"{name} best CV recall: {search.best_score_}"
                    )

                # Direct training for Logistic Regression
                else:

                    pipeline.fit(
                        self.x_train,
                        self.y_train
                    )

                    trained_models[name] = pipeline

                    mlflow.log_params(
                        model.get_params(deep=False)
                    )

                logger.info(
                    f"Training completed for {name}"
                )

        logger.info("All models training completed")

        return trained_models, run_ids
