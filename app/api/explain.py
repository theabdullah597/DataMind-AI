import os
import joblib
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(
    prefix="/models/{model_id}/explain",
    tags=["Explainability"]
)

MODELS_DIR = "models"

@router.get("")
async def explain_model(model_id: str):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        artifact = joblib.load(model_path)
        model = artifact["model"]
        features = artifact["features"]
        
        explanations = {}
        
        # 1. Linear model coefficients
        if hasattr(model, "coef_"):
            coefs = model.coef_
            # For classification, coef_ might be 2D
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            explanations["coefficients"] = dict(zip(features, coefs.tolist()))
            
        # 2. Tree-based feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            explanations["feature_importances"] = dict(zip(features, importances.tolist()))
            
        # 3. Limitations/Notice
        explanations["notice"] = (
            "Feature importance and coefficients indicate correlation and contribution "
            "to the model's predictions, NOT necessarily real-world causation."
        )
            
        return {
            "model_id": model_id,
            "model_name": artifact["model_name"],
            "explanations": explanations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")
