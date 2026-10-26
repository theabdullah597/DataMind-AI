import io
import os
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

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
@router.post("/predict/batch")
async def predict_batch(model_id: str, file: UploadFile = File(...)):
    """
    Execute batch prediction on an uploaded CSV or Excel file of new records.
    Appends predictions and probabilities to the dataset and returns results.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file, sep=None, engine='python', encoding_errors='replace')
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or XLSX.")

        artifact = joblib.load(model_path)
        model = artifact["model"]
        preprocessor = artifact["preprocessor"]
        expected_features = artifact["features"]

        # Validate schema
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is missing required feature columns: {missing_features}. Download the template CSV to see required headers."
            )

        df_input = df[expected_features]
        X_processed = preprocessor.transform(df_input)

        predictions = model.predict(X_processed)
        df["prediction"] = predictions

        if artifact["task_type"] == "classification" and hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_processed)
            for i in range(probas.shape[1]):
                df[f"probability_class_{i}"] = probas[:, i].round(4)

        df = df.fillna("NaN")
        return {
            "model_id": model_id,
            "total_predictions": len(df),
            "features_used": expected_features,
            "predictions": df.to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@router.get("/template")
@router.get("/predict/template")
async def download_prediction_template(model_id: str):
    """
    Download a pre-formatted blank CSV template containing the exact required feature column headers.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        artifact = joblib.load(model_path)
        expected_features = artifact.get("features", [])

        df_template = pd.DataFrame(columns=expected_features)
        csv_buffer = io.StringIO()
        df_template.to_csv(csv_buffer, index=False)

        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=batch_prediction_template_{model_id[:8]}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating template: {str(e)}")


@router.get("/download")
async def download_model(model_id: str):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=f"model_{model_id}.joblib", media_type="application/octet-stream")
