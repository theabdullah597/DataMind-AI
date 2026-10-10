import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.tools.ml_tools import validate_target_column, prepare_ml_data, train_regression_models, train_classification_models

def test_validate_target_column():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "target": [0, 1, 0]
    })
    
    # Valid classification
    res = validate_target_column(df, "target", task_type="classification")
    assert res["valid"] is True
    
    # Invalid regression (target has < 2 unique values)
    df_invalid = pd.DataFrame({"A": [1, 2, 3], "target": [1, 1, 1]})
    with pytest.raises(ValueError):
        validate_target_column(df_invalid, "target", task_type="classification")
        
def test_prepare_ml_data():
    df = pd.DataFrame({
        "num1": [1.5, 2.5, 3.5, 4.5, 5.5],
        "cat1": ["a", "b", "a", "b", "a"],
        "constant": [1, 1, 1, 1, 1], # Should be removed
        "target": [0, 1, 0, 1, 0]
    })
    
    res = prepare_ml_data(df, "target", task_type="classification", test_size=0.4)
    assert "constant" not in res["numerical_columns"]
    assert "constant" not in res["categorical_columns"]
    assert len(res["X_train"]) == 3
    assert len(res["X_test"]) == 2
    
def test_train_classification_models():
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.rand(100),
        "num2": np.random.rand(100),
        "target": np.random.randint(0, 2, 100)
    })
    
    prepared = prepare_ml_data(df, "target", task_type="classification", test_size=0.2)
    models = train_classification_models(prepared)
    
    assert "Logistic Regression" in models["results"]
    assert "accuracy" in models["results"]["Logistic Regression"]
    assert models["results"]["Logistic Regression"]["accuracy"] >= 0

def test_train_regression_models():
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.rand(100),
        "num2": np.random.rand(100),
        "target": np.random.rand(100) * 100
    })
    
    prepared = prepare_ml_data(df, "target", task_type="regression", test_size=0.2)
    models = train_regression_models(prepared)
    
    assert "Linear Regression" in models["results"]
    assert "mse" in models["results"]["Linear Regression"]
    assert models["results"]["Linear Regression"]["mse"] >= 0
