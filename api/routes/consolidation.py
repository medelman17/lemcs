from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import logging

from agents.consolidation_agent import ConsolidationAgent
from nlp.document_grouping_service import DocumentGroupingService, GroupingStrategy
from nlp.citation_deduplication_service import CitationDeduplicationService, CitationFormat
from nlp.legal_theory_synthesis_service import LegalTheorySynthesisService, SynthesisMode
from db.database import get_session
from db.models import Document

router = APIRouter()
logger = logging.getLogger(__name__)

# Background task storage (in production, use Redis or similar)
consolidation_jobs = {}


class ConsolidationMethodEnum(str, Enum):
    CRRACC = "CRRACC"
    CHRONOLOGICAL = "chronological"
    THEORY_BASED = "theory_based"


class OutputFormatEnum(str, Enum):
    DOCX = "docx"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"


class ConsolidationRequest(BaseModel):
    document_ids: List[str] = Field(..., min_items=2, description="List of document IDs to consolidate")
    consolidation_method: ConsolidationMethodEnum = Field(default=ConsolidationMethodEnum.CRRACC)
    grouping_strategy: str = Field(default="legal_theory", description="Document grouping strategy")
    synthesis_mode: str = Field(default="comprehensive", description="Content synthesis approach")
    citation_format: str = Field(default="new_jersey", description="Target citation format")
    output_format: OutputFormatEnum = Field(default=OutputFormatEnum.DOCX)
    title: Optional[str] = Field(default=None, description="Custom title for consolidated document")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")


class ConsolidationResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    estimated_completion: Optional[str] = None
    document_count: int
    consolidation_method: str


class ConsolidationStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_stage: str
    stages_completed: List[str]
    stages_remaining: List[str]
    error_message: Optional[str] = None
    result_preview: Optional[Dict[str, Any]] = None


class ConsolidationResult(BaseModel):
    job_id: str
    status: str
    title: str
    consolidated_content: str
    total_citations: int
    legal_theories_count: int
    quality_metrics: Dict[str, float]
    processing_time_ms: int
    created_at: str
    download_urls: Optional[Dict[str, str]] = None


class DocumentGroupingRequest(BaseModel):
    document_ids: List[str]
    strategy: str = "legal_theory"
    min_cluster_size: int = 2
    similarity_threshold: float = 0.7


class DocumentGroupingResult(BaseModel):
    strategy: str
    clusters: List[Dict[str, Any]]
    ungrouped_documents: List[str]
    quality_metrics: Dict[str, float]


class CitationAnalysisRequest(BaseModel):
    document_ids: List[str]
    target_format: str = "new_jersey"
    preserve_context: bool = True


class CitationAnalysisResult(BaseModel):
    original_citation_count: int
    deduplicated_citation_count: int
    citation_clusters: List[Dict[str, Any]]
    format_consistency_score: float
    preservation_rate: float


# Dependency to get consolidation agent
async def get_consolidation_agent() -> ConsolidationAgent:
    return ConsolidationAgent()


async def get_grouping_service() -> DocumentGroupingService:
    return DocumentGroupingService()


async def get_citation_service() -> CitationDeduplicationService:
    return CitationDeduplicationService()


async def get_synthesis_service() -> LegalTheorySynthesisService:
    return LegalTheorySynthesisService()


@router.post("/consolidation/jobs", response_model=ConsolidationResponse)
async def create_consolidation_job(
    request: ConsolidationRequest,
    background_tasks: BackgroundTasks,
    agent: ConsolidationAgent = Depends(get_consolidation_agent)
):
    """
    Create a new document consolidation job.
    
    This endpoint initiates the CRRACC consolidation process for multiple legal memoranda.
    The process runs asynchronously in the background.
    """
    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 documents are required for consolidation",
        )

    job_id = str(uuid.uuid4())
    
    # Initialize job status
    consolidation_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_stage": "initialization",
        "stages_completed": [],
        "stages_remaining": [
            "document_loading", "legal_theory_extraction", "document_grouping",
            "content_synthesis", "citation_deduplication", "quality_assessment", "final_assembly"
        ],
        "created_at": datetime.utcnow(),
        "request": request.dict(),
        "error_message": None,
        "result": None
    }
    
    # Start background consolidation task
    background_tasks.add_task(
        run_consolidation_job,
        job_id,
        request,
        agent
    )
    
    logger.info(f"Created consolidation job {job_id} with {len(request.document_ids)} documents")
    
    return ConsolidationResponse(
        job_id=job_id,
        status="queued",
        created_at=consolidation_jobs[job_id]["created_at"].isoformat(),
        estimated_completion=(datetime.utcnow().isoformat()),  # Would calculate based on doc count
        document_count=len(request.document_ids),
        consolidation_method=request.consolidation_method.value
    )


@router.get("/consolidation/jobs/{job_id}", response_model=ConsolidationStatus)
async def get_consolidation_status(job_id: str):
    """Get the current status of a consolidation job."""
    
    if job_id not in consolidation_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Consolidation job {job_id} not found"
        )
    
    job = consolidation_jobs[job_id]
    
    # Create preview if job is completed
    result_preview = None
    if job["status"] == "completed" and job["result"]:
        result_preview = {
            "title": job["result"].title,
            "theories_count": job["result"].legal_theories_count,
            "citations_count": job["result"].total_citations,
            "quality_score": job["result"].quality_metrics.get("overall_quality", 0.0)
        }
    
    return ConsolidationStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_stage=job["current_stage"],
        stages_completed=job["stages_completed"],
        stages_remaining=job["stages_remaining"],
        error_message=job["error_message"],
        result_preview=result_preview
    )


@router.get("/consolidation/jobs/{job_id}/result", response_model=ConsolidationResult)
async def get_consolidation_result(job_id: str):
    """Get the final result of a completed consolidation job."""
    
    if job_id not in consolidation_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Consolidation job {job_id} not found"
        )
    
    job = consolidation_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Consolidation job {job_id} is not yet completed (status: {job['status']})"
        )
    
    if not job["result"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consolidation result not available"
        )
    
    result = job["result"]
    
    # Format consolidated content
    consolidated_content = _format_consolidation_result(result)
    
    # Calculate processing time
    processing_time = (datetime.utcnow() - job["created_at"]).total_seconds() * 1000
    
    return ConsolidationResult(
        job_id=job_id,
        status=job["status"],
        title=result.title,
        consolidated_content=consolidated_content,
        total_citations=result.total_citations,
        legal_theories_count=len(result.legal_theories),
        quality_metrics=result.quality_metrics,
        processing_time_ms=int(processing_time),
        created_at=job["created_at"].isoformat(),
        download_urls=None  # Would generate actual download URLs
    )


@router.post("/consolidation/document-grouping", response_model=DocumentGroupingResult)
async def analyze_document_grouping(
    request: DocumentGroupingRequest,
    grouping_service: DocumentGroupingService = Depends(get_grouping_service)
):
    """
    Analyze how documents would be grouped for consolidation.
    
    This endpoint helps users understand document relationships before starting consolidation.
    """
    try:
        # Load documents from database
        documents = await _load_documents(request.document_ids)
        
        # Parse strategy
        strategy = GroupingStrategy(request.strategy)
        
        # Perform grouping analysis
        grouping_result = await grouping_service.group_documents(
            documents=documents,
            strategy=strategy,
            min_cluster_size=request.min_cluster_size,
            similarity_threshold=request.similarity_threshold
        )
        
        # Format clusters for response
        formatted_clusters = []
        for cluster in grouping_result.clusters:
            formatted_clusters.append({
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.cluster_name,
                "primary_theory": cluster.primary_theory,
                "document_count": len(cluster.documents),
                "similarity_score": cluster.similarity_score,
                "strength_score": cluster.strength_score,
                "key_provisions": cluster.key_provisions,
                "document_ids": [doc.get("id", "") for doc in cluster.documents]
            })
        
        return DocumentGroupingResult(
            strategy=grouping_result.strategy.value,
            clusters=formatted_clusters,
            ungrouped_documents=[doc.get("id", "") for doc in grouping_result.ungrouped_documents],
            quality_metrics=grouping_result.quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Document grouping analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document grouping analysis failed: {str(e)}"
        )


@router.post("/consolidation/citation-analysis", response_model=CitationAnalysisResult)
async def analyze_citations(
    request: CitationAnalysisRequest,
    citation_service: CitationDeduplicationService = Depends(get_citation_service)
):
    """
    Analyze citations across documents for deduplication and formatting.
    
    This endpoint provides citation analysis before consolidation to preview
    the citation management process.
    """
    try:
        # Load documents from database
        documents = await _load_documents(request.document_ids)
        
        # Parse citation format
        citation_format = CitationFormat(request.target_format)
        
        # Perform citation analysis
        dedup_result = await citation_service.deduplicate_citations(
            source_documents=documents,
            preserve_context=request.preserve_context,
            normalize_format=citation_format
        )
        
        # Format clusters for response
        formatted_clusters = []
        for cluster in dedup_result.citation_clusters:
            formatted_clusters.append({
                "cluster_id": cluster.cluster_id,
                "primary_citation": cluster.consolidated_format,
                "alternative_count": len(cluster.alternative_citations),
                "citation_type": cluster.citation_type.value,
                "authority_ranking": cluster.authority_ranking,
                "usage_frequency": cluster.usage_frequency,
                "contexts": cluster.contexts[:3]  # Limit for response size
            })
        
        return CitationAnalysisResult(
            original_citation_count=dedup_result.original_citation_count,
            deduplicated_citation_count=dedup_result.deduplicated_citation_count,
            citation_clusters=formatted_clusters,
            format_consistency_score=dedup_result.format_consistency_score,
            preservation_rate=dedup_result.preservation_rate
        )
        
    except Exception as e:
        logger.error(f"Citation analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation analysis failed: {str(e)}"
        )


@router.delete("/consolidation/jobs/{job_id}")
async def cancel_consolidation_job(job_id: str):
    """Cancel a running consolidation job."""
    
    if job_id not in consolidation_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Consolidation job {job_id} not found"
        )
    
    job = consolidation_jobs[job_id]
    
    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Mark job as cancelled
    job["status"] = "cancelled"
    job["current_stage"] = "cancelled"
    
    logger.info(f"Cancelled consolidation job {job_id}")
    
    return {"message": f"Consolidation job {job_id} cancelled successfully"}


@router.get("/consolidation/jobs")
async def list_consolidation_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List consolidation jobs with optional filtering."""
    
    # Filter jobs
    filtered_jobs = []
    for job in consolidation_jobs.values():
        if status_filter and job["status"] != status_filter:
            continue
        filtered_jobs.append({
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "current_stage": job["current_stage"],
            "created_at": job["created_at"].isoformat(),
            "document_count": len(job["request"]["document_ids"])
        })
    
    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Apply pagination
    paginated_jobs = filtered_jobs[offset:offset + limit]
    
    return {
        "jobs": paginated_jobs,
        "total": len(filtered_jobs),
        "limit": limit,
        "offset": offset
    }


# Background task functions

async def run_consolidation_job(job_id: str, request: ConsolidationRequest, agent: ConsolidationAgent):
    """Run the consolidation job in the background."""
    
    job = consolidation_jobs[job_id]
    
    try:
        # Update job status
        job["status"] = "running"
        job["current_stage"] = "document_loading"
        job["progress"] = 10.0
        
        # Load documents
        documents = await _load_documents(request.document_ids)
        _update_job_progress(job, "legal_theory_extraction", 20.0)
        
        # Parse grouping strategy and synthesis mode
        grouping_strategy = GroupingStrategy(request.grouping_strategy)
        synthesis_mode = SynthesisMode(request.synthesis_mode)
        
        # Run consolidation
        result = await agent.consolidate_memoranda(
            memoranda_data=documents,
            consolidation_strategy=grouping_strategy.value
        )
        
        _update_job_progress(job, "completed", 100.0)
        
        # Store result
        job["result"] = result
        job["status"] = "completed"
        
        logger.info(f"Consolidation job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Consolidation job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["current_stage"] = "failed"


async def _load_documents(document_ids: List[str]) -> List[Dict[str, Any]]:
    """Load documents from database by IDs."""
    
    documents = []
    
    try:
        async with get_session() as session:
            for doc_id in document_ids:
                # In a real implementation, load from database
                # For now, create mock documents
                mock_doc = {
                    "id": doc_id,
                    "content": f"Mock content for document {doc_id}",
                    "title": f"Document {doc_id}",
                    "citations": [],
                    "metadata": {}
                }
                documents.append(mock_doc)
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load documents: {str(e)}"
        )


def _update_job_progress(job: Dict[str, Any], stage: str, progress: float):
    """Update job progress and stage."""
    
    job["progress"] = progress
    
    if stage != job["current_stage"]:
        # Move current stage to completed
        if job["current_stage"] not in job["stages_completed"]:
            job["stages_completed"].append(job["current_stage"])
        
        # Update current stage
        job["current_stage"] = stage
        
        # Remove from remaining stages
        if stage in job["stages_remaining"]:
            job["stages_remaining"].remove(stage)


def _format_consolidation_result(result) -> str:
    """Format the consolidation result for API response."""
    
    formatted_content = f"""
# {result.title}

## I. CONCLUSION

{result.conclusion.content}

## II. RULE STATEMENT

{result.rule_statement.content}

## III. RULE EXPLANATION

{result.rule_explanation.content}

## IV. APPLICATION

{result.application.content}

## V. COUNTERARGUMENT

{result.counterargument.content}

## VI. CONCLUSION

{result.final_conclusion.content}

---

**Quality Metrics:**
- Overall Quality: {result.quality_metrics.get('overall_quality', 0.0):.2%}
- Legal Theories: {len(result.legal_theories)}
- Total Citations: {result.total_citations}
- Consolidated Documents: {result.consolidated_memoranda_count}
"""
    
    return formatted_content.strip()
