import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from app.tools.clean_tools import (
    remove_duplicates,
    handle_missing_values,
    correct_data_types,
    extract_datetime_features
)

router = APIRouter(
    prefix="/datasets/{dataset_id}/clean",
    tags=["Clean"]
)

UPLOAD_DIR = "uploads"

class CleanRequest(BaseModel):
    remove_duplicates: Optional[bool] = False
    keep_duplicates: Optional[str] = "first" # "first", "last", "False"
    missing_strategy: Optional[Dict[str, str]] = None
    type_corrections: Optional[Dict[str, str]] = None
    datetime_columns: Optional[List[str]] = None
    
@router.post("")
async def clean_dataset(dataset_id: str, request: CleanRequest):
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
            
        # 1. Remove duplicates
        if request.remove_duplicates:
            keep = False if request.keep_duplicates == "False" else request.keep_duplicates
            df = remove_duplicates(df, keep=keep)
            
        # 2. Handle missing values
        if request.missing_strategy:
            df = handle_missing_values(df, request.missing_strategy)
            
        # 3. Correct data types
        if request.type_corrections:
            df = correct_data_types(df, request.type_corrections)
            
        # 4. Extract datetime features
        if request.datetime_columns:
            df = extract_datetime_features(df, request.datetime_columns)
            
        # Save back to file (this overwrites the existing file)
        # Note: in a real production system, we'd probably save this as a new version
        if dataset_id.endswith(".csv"):
            df.to_csv(file_path, index=False)
        elif dataset_id.endswith(".xlsx"):
            df.to_excel(file_path, index=False)
            
        return {
            "dataset_id": dataset_id,
            "message": "Dataset cleaned successfully.",
            "rows_remaining": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {str(e)}")
