import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.tools.clean_tools import (
    remove_duplicates,
    handle_missing_values,
    correct_data_types,
    extract_datetime_features
)

def test_remove_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2

def test_handle_missing_values():
    df = pd.DataFrame({"A": [1, np.nan, 3], "B": ["x", np.nan, "x"]})
    cleaned = handle_missing_values(df, {"A": "mean", "B": "mode"})
    assert cleaned["A"].isnull().sum() == 0
    assert cleaned["B"].isnull().sum() == 0
    assert cleaned["A"].iloc[1] == 2.0
    assert cleaned["B"].iloc[1] == "x"

def test_correct_data_types():
    df = pd.DataFrame({"A": ["1", "2"], "B": [1.1, 2.2]})
    cleaned = correct_data_types(df, {"A": "int", "B": "str"})
    assert pd.api.types.is_integer_dtype(cleaned["A"])
    assert pd.api.types.is_string_dtype(cleaned["B"]) or cleaned["B"].dtype == 'object'

def test_extract_datetime_features():
    df = pd.DataFrame({"dt": ["2020-01-01", "2021-12-31"]})
    cleaned = extract_datetime_features(df, ["dt"])
    assert "dt_year" in cleaned.columns
    assert cleaned["dt_year"].iloc[0] == 2020
