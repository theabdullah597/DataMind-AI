import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from app.tools.eda_tools import (
    analyze_numerical_columns,
    analyze_categorical_columns,
    analyze_target_column,
    analyze_correlations,
    detect_outliers,
    generate_eda_insights
)

router = APIRouter(
    prefix="/datasets/{dataset_id}/eda",
    tags=["EDA"]
)

UPLOAD_DIR = "uploads"

@router.post("")
@router.post("/")
async def generate_eda(dataset_id: str, target_column: str = Query(None)):
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
            
        numerical_results = analyze_numerical_columns(df)
        categorical_results = analyze_categorical_columns(df)
        correlation_results = analyze_correlations(df)
        outlier_results = detect_outliers(df)
        
        target_results = {}
        if target_column:
            try:
                target_results = analyze_target_column(df, target_column)
            except Exception as e:
                # Target column might be invalid, but we still return other EDA
                target_results = {"error": str(e)}
                
        insights = generate_eda_insights(
            numerical_results,
            categorical_results,
            target_results if "error" not in target_results else {},
            correlation_results,
            outlier_results
        )
        
        return {
            "dataset_id": dataset_id,
            "numerical_analysis": numerical_results,
            "categorical_analysis": categorical_results,
            "correlation_analysis": correlation_results,
            "outlier_analysis": outlier_results,
            "target_analysis": target_results,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing EDA: {str(e)}")
