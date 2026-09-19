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
