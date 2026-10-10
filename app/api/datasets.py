import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from app.tools.tool_script import inspect_dataset
import pandas as pd

router=APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)
UPLOAD_DIR="uploads"
os.makedirs(UPLOAD_DIR,exist_ok=True)
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed_extensions=[".csv",".xlsx"]
    file_extension=os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400,detail="File type not allowed")

    unique_filename=f"{uuid.uuid4()}{file_extension}"
    file_path=os.path.join(UPLOAD_DIR,unique_filename)
    try:
        file_size = 0
        with open(file_path,"wb") as buffer:
            while chunk:=await file.read(1024*1024):
                file_size += len(chunk)
                if file_size > 50 * 1024 * 1024:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(status_code=413, detail="File too large (max 50MB)")
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400,detail=str(e))
    try:

        dataset_profile = inspect_dataset(file_path)

    except Exception as e:

        # Delete invalid file
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze dataset: {str(e)}"
        )

        # ----------------------------------------
        # 5. Return response
        # ----------------------------------------

    return {
        "message": "Dataset uploaded and inspected successfully.",
        "filename": file.filename,
        "saved_filename": unique_filename,
        "profile": dataset_profile
    }

@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
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
            
        total_rows = len(df)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        
        subset = df.iloc[start_idx:end_idx]
        subset = subset.fillna("NaN") # JSON serialization safety
        records = subset.to_dict(orient="records")
        
        return {
            "dataset_id": dataset_id,
            "page": page,
            "size": size,
            "total_rows": total_rows,
            "total_pages": (total_rows + size - 1) // size,
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

@router.get("/{dataset_id}/profile")
async def profile_dataset(dataset_id: str):
    file_path = os.path.join(UPLOAD_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        dataset_profile = inspect_dataset(file_path)
        return {
            "dataset_id": dataset_id,
            "profile": dataset_profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error profiling dataset: {str(e)}")

@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    file_path = os.path.join(UPLOAD_DIR, dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    media_type = "text/csv" if dataset_id.endswith(".csv") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(file_path, filename=f"DataMind_{dataset_id}", media_type=media_type)
