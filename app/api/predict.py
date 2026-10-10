import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from typing import List, Dict, Any

router = APIRouter(
    prefix="/models/{model_id}",
    tags=["Models", "Predictions"]
)

MODELS_DIR = "models"

@router.get("")
async def get_model_metadata(model_id: str):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        artifact = joblib.load(model_path)
        return {
            "model_id": artifact.get("model_id"),
            "dataset_id": artifact.get("dataset_id"),
            "target_column": artifact.get("target_column"),
            "task_type": artifact.get("task_type"),
            "model_name": artifact.get("model_name"),
            "features": artifact.get("features"),
            "metrics": artifact.get("metrics")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading model: {str(e)}")

@router.post("/predict")
async def predict_single(model_id: str, features: Dict[str, Any]):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        artifact = joblib.load(model_path)
        model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        expected_features = artifact["features"]
        
        # Validate schema
        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
            
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Only keep expected features (handles unexpected features)
        df = df[expected_features]
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)
        
        result = {"prediction": prediction[0].item() if hasattr(prediction[0], "item") else prediction[0]}
        
        if artifact["task_type"] == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_processed)
            result["probabilities"] = proba[0].tolist()
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/batch")
async def predict_batch(model_id: str, file: UploadFile = File(...)):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file, sep=None, engine='python', encoding_errors='replace')
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        artifact = joblib.load(model_path)
        model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        expected_features = artifact["features"]
        
        # Validate schema
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
            
        df_input = df[expected_features]
        X_processed = preprocessor.transform(df_input)
        
        predictions = model.predict(X_processed)
        df["prediction"] = predictions
        
        if artifact["task_type"] == "classification" and hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_processed)
            for i in range(probas.shape[1]):
                df[f"probability_class_{i}"] = probas[:, i]
                
        df = df.fillna("NaN")
        return {
            "model_id": model_id,
            "predictions": df.to_dict(orient="records")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.get("/download")
async def download_model(model_id: str):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=f"model_{model_id}.joblib", media_type="application/octet-stream")
