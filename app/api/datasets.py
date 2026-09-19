import os
import uuid
from fastapi import APIRouter,UploadFile,File,HTTPException
from app.tools.tool_script import inspect_dataset

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
        with open(file_path,"wb") as buffer:
            while chunk:=await file.read(1024*1024):
                buffer.write(chunk)
    except Exception as e:
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
