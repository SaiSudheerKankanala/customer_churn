import numpy as np
from sklearn.compose import ColumnTransformer
from src.preprocessing import Preprocessing
from sklearn.preprocessing import OneHotEncoder

def test_preprocessing_object(sample_features):
    result = Preprocessing(sample_features).preprocessing()    
    assert isinstance(result,ColumnTransformer)

def test_remove_unnecessary_columns(sample_features):
    processor = Preprocessing(sample_features)
    processor.preprocessing()
    assert 'customerID' not in processor.data.columns

def test_identify_numerical_and_categorical_columns(sample_features):
    processor = Preprocessing(sample_features)
    processor.preprocessing()

    numerical_columns = processor.data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = processor.data.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    assert "tenure" in numerical_columns
    assert "gender" in categorical_columns

def test_numerical_imputer_strategy(sample_features):
    processor = Preprocessing(sample_features).preprocessing()
    numerical_pipeline = processor.transformers[0][1]
    imputer = numerical_pipeline.named_steps["impute"]
    assert imputer.strategy == "mean"

def test_categorical_imputation(sample_features):
    processor = Preprocessing(sample_features).preprocessing()
    categorical_pipeline = processor.transformers[1][1]
    imputer = categorical_pipeline.named_steps["impute"]
    assert imputer.strategy == 'most_frequent'


def test_categorical_missing_values_imputation(sample_features):
    processor = Preprocessing(sample_features)
    preprocessor = processor.preprocessing()
    categorical_columns = processor.data.select_dtypes(exclude=["number"]).columns.tolist()
    categorical_data = processor.data[categorical_columns].copy()
    categorical_data.loc[categorical_data.index[0], "gender"] = np.nan
    expected_value = categorical_data["gender"].mode()[0]
    categorical_pipeline = preprocessor.transformers[1][1]
    imputer = categorical_pipeline.named_steps["impute"]
    imputed_data = imputer.fit_transform(categorical_data)
    gender_index = categorical_data.columns.get_loc("gender")
    assert imputed_data[0, gender_index] == expected_value

def test_onehot_encoding(sample_features):
    processor = Preprocessing(sample_features).preprocessing()
    categorical_pipeline = processor.transformers[1][1]
    encoder = categorical_pipeline.named_steps['encoder']
    assert isinstance(encoder,OneHotEncoder)
    assert encoder.drop == 'first'
    assert encoder.handle_unknown == 'ignore'


def test_encoder_handles_unknown_categories(sample_features):
    new_data = sample_features.iloc[[0]].copy()
    new_data['gender'] = 'Unknown_Gender'
    processor = Preprocessing(sample_features).preprocessing()
    processor.fit(sample_features)
    transformed_data = processor.transform(new_data)
    assert transformed_data.shape[0] == 1

def test_preprocessing_output_shape(sample_features):
    processor = Preprocessing(sample_features).preprocessing()
    transformed_data = processor.fit_transform(sample_features)
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    assert transformed_data.shape[0] == len(sample_features)
    assert transformed_data.ndim == 2
    assert transformed_data.shape[1] > 0