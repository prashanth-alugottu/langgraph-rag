from fastapi import APIRouter, UploadFile, File
from services.uploading_service import upload_file
from typing import List

router = APIRouter(prefix="/api/v1")


@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        file_bytes = await file.read()
        result = upload_file(file_bytes, file.filename)
        results.append({
            "filename": file.filename,
            "details": result
        })

    return {
        "message": "Files uploaded successfully",
        "files": results
    }
