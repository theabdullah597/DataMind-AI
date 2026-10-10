import os
import uuid
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from app.tools.ml_tools import prepare_ml_data, train_regression_models, train_classification_models

router = APIRouter(
    prefix="/datasets/{dataset_id}/ml",
    tags=["Machine Learning"]
)

UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class TrainRequest(BaseModel):
    target_column: str
    task_type: Optional[str] = "auto"
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
    primary_metric: Optional[str] = None
    
@router.post("/train")
async def train_models(dataset_id: str, request: TrainRequest):
    file_path = os.path.join(UPLOAD_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        if dataset_id.endswith(".csv"):
            df = pd.read_csv(file_path, sep=None, engine='python', encoding_errors='replace')
        elif dataset_id.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        prepared = prepare_ml_data(
            df=df,
            target_column=request.target_column,
            task_type=request.task_type,
            test_size=request.test_size,
            random_state=request.random_state
        )
        
        # Determine actual task type from y_train if auto
        y_train = prepared["y_train"]
        is_classification = False
        if request.task_type == "classification":
            is_classification = True
        elif request.task_type == "regression":
            is_classification = False
        else:
            unique_count = y_train.nunique()
            is_classification = (not pd.api.types.is_numeric_dtype(y_train) or unique_count <= 10)
            
        if is_classification:
            primary_metric = request.primary_metric or "accuracy"
            models_output = train_classification_models(prepared, primary_metric=primary_metric)
        else:
            primary_metric = request.primary_metric or "rmse"
            models_output = train_regression_models(prepared, primary_metric=primary_metric)
            
        best_model_name = models_output["recommendation"]["model"]
        best_model = models_output["trained_models"][best_model_name]
        
        # Save model and preprocessor
        model_id = str(uuid.uuid4())
        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        
        artifact = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "target_column": request.target_column,
            "task_type": "classification" if is_classification else "regression",
            "model_name": best_model_name,
            "model": best_model,
            "preprocessor": prepared["preprocessor"],
            "metrics": models_output["results"][best_model_name],
            "features": prepared["numerical_columns"] + prepared["categorical_columns"]
        }
        
        joblib.dump(artifact, model_path)
        
        return {
            "model_id": model_id,
            "task_type": artifact["task_type"],
            "model_name": best_model_name,
            "recommendation_reason": models_output["recommendation"]["reason"],
            "metrics": artifact["metrics"],
            "all_results": models_output["results"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")
