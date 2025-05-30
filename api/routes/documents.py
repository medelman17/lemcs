from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List
import uuid
from datetime import datetime

router = APIRouter()


@router.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    uploaded_files = []
    
    for file in files:
        if not any(file.filename.endswith(ext) for ext in [".docx", ".pdf", ".txt"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} has unsupported format"
            )
        
        document_id = str(uuid.uuid4())
        uploaded_files.append({
            "id": document_id,
            "filename": file.filename,
            "size": file.size,
            "uploaded_at": datetime.utcnow().isoformat()
        })
    
    return {
        "message": f"Successfully uploaded {len(uploaded_files)} documents",
        "documents": uploaded_files
    }


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    return {
        "id": document_id,
        "filename": "sample_memorandum.docx",
        "status": "processed",
        "metadata": {
            "pages": 15,
            "citations_extracted": 42,
            "legal_issues": ["breach of contract", "warranty claims"]
        }
    }