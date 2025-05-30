from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

router = APIRouter()


class ConsolidationRequest(BaseModel):
    document_ids: List[str]
    consolidation_method: str = "CRRACC"
    output_format: str = "docx"
    options: Optional[dict] = {}


class ConsolidationResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    estimated_completion: Optional[str] = None


@router.post("/consolidation/jobs", response_model=ConsolidationResponse)
async def create_consolidation_job(request: ConsolidationRequest):
    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 documents are required for consolidation",
        )

    job_id = str(uuid.uuid4())

    return ConsolidationResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        estimated_completion=datetime.utcnow().isoformat(),
    )


@router.get("/consolidation/jobs/{job_id}")
async def get_consolidation_status(job_id: str):
    return {
        "job_id": job_id,
        "status": "in_progress",
        "progress": 65,
        "current_stage": "citation_extraction",
        "stages_completed": ["document_parsing", "legal_nlp_analysis"],
        "stages_remaining": ["synthesis", "quality_check", "final_formatting"],
    }
