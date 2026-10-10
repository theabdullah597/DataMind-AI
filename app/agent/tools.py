import os
import json
import pandas as pd
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# Import our backend functions
from app.tools.tool_script import inspect_dataset
from app.tools.clean_tools import remove_duplicates, handle_missing_values, correct_data_types, extract_datetime_features
from app.tools.eda_tools import analyze_numerical_columns, analyze_categorical_columns, analyze_target_column, analyze_correlations, detect_outliers, generate_eda_insights
from app.tools.ml_tools import prepare_ml_data, train_regression_models, train_classification_models
import joblib
import uuid

UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_df(dataset_id: str) -> pd.DataFrame:
    file_path = os.path.join(UPLOAD_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {dataset_id} not found")
    if dataset_id.endswith(".csv"):
        return pd.read_csv(file_path, sep=None, engine='python', encoding_errors='replace')
    return pd.read_excel(file_path)

@tool
def get_dataset_profile(dataset_id: str) -> str:
    """Get the profile of a dataset including row count, columns, missing values, etc."""
    try:
        file_path = os.path.join(UPLOAD_DIR, dataset_id)
        profile = inspect_dataset(file_path)
        return json.dumps(profile)
    except Exception as e:
        return f"Error profiling dataset: {str(e)}"

@tool
def clean_dataset(dataset_id: str, remove_dups: bool = False, missing_strategy: str = "{}", type_corrections: str = "{}") -> str:
    """
    Clean the dataset.
    missing_strategy is a JSON string of {"col": "mean"|"median"|"mode"|"drop"}.
    type_corrections is a JSON string of {"col": "int"|"float"|"str"|"datetime"}.
    """
    try:
        df = get_df(dataset_id)
        initial_rows = len(df)
        
        if remove_dups:
            df = remove_duplicates(df)
            
        missing_dict = json.loads(missing_strategy)
        if missing_dict:
            df = handle_missing_values(df, missing_dict)
            
        type_dict = json.loads(type_corrections)
        if type_dict:
            df = correct_data_types(df, type_dict)
            
        file_path = os.path.join(UPLOAD_DIR, dataset_id)
        if dataset_id.endswith(".csv"):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)
            
        return f"Dataset cleaned. Rows went from {initial_rows} to {len(df)}."
    except Exception as e:
        return f"Error cleaning dataset: {str(e)}"

@tool
def perform_eda(dataset_id: str, target_column: str = None) -> str:
    """Perform exploratory data analysis and generate insights. Optionally specify a target column."""
    try:
        df = get_df(dataset_id)
        
        num_res = analyze_numerical_columns(df)
        cat_res = analyze_categorical_columns(df)
        corr_res = analyze_correlations(df)
        out_res = detect_outliers(df)
        
        tgt_res = {}
        if target_column and target_column in df.columns:
            tgt_res = analyze_target_column(df, target_column)
            
        insights = generate_eda_insights(num_res, cat_res, tgt_res, corr_res, out_res)
        
        return json.dumps({
            "insights": insights,
            "target_analysis": tgt_res,
            "correlations": corr_res.get("strong_correlations", [])
        })
    except Exception as e:
        return f"Error performing EDA: {str(e)}"

@tool
def train_model(dataset_id: str, target_column: str, task_type: str = "auto") -> str:
    """Train machine learning models and return the recommended model_id and its metrics. task_type can be 'classification', 'regression' or 'auto'"""
    try:
        df = get_df(dataset_id)
        prepared = prepare_ml_data(df, target_column, task_type=task_type)
        
        y_train = prepared["y_train"]
        is_classification = False
        if task_type == "classification":
            is_classification = True
        elif task_type == "regression":
            is_classification = False
        else:
            is_classification = (not pd.api.types.is_numeric_dtype(y_train) or y_train.nunique() <= 10)
            
        if is_classification:
            res = train_classification_models(prepared)
        else:
            res = train_regression_models(prepared)
            
        best_name = res["recommendation"]["model"]
        metrics = res["results"][best_name]
        
        model_id = str(uuid.uuid4())
        joblib.dump({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "task_type": "classification" if is_classification else "regression",
            "model_name": best_name,
            "model": res["trained_models"][best_name],
            "preprocessor": prepared["preprocessor"],
            "metrics": metrics,
            "features": prepared["numerical_columns"] + prepared["categorical_columns"]
        }, os.path.join(MODELS_DIR, f"{model_id}.joblib"))
        
        return f"Successfully trained models. Recommended model: {best_name} (ID: {model_id}). Metrics: {json.dumps(metrics)}"
    except Exception as e:
        return f"Error training model: {str(e)}"
