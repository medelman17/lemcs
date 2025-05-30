"""
Citation extraction API endpoints.
Provides REST API for extracting and analyzing legal citations from documents.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import uuid

from db.database import get_db
from db.models import Document, Citation, AgentWorkflow, DocumentStatus
from nlp.citation_service import citation_service
from agents.citation_extractor import citation_extractor_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/citations", tags=["citations"])


# Request/Response Models
class CitationExtractionRequest(BaseModel):
    """Request model for citation extraction"""
    document_id: str = Field(..., description="UUID of the document to process")
    resolve_references: bool = Field(True, description="Whether to resolve supra/id citations")
    create_embeddings: bool = Field(False, description="Whether to create vector embeddings")
    analyze_authority: bool = Field(True, description="Whether to analyze citation authority")
    create_relationships: bool = Field(True, description="Whether to create citation relationships")


class CitationResponse(BaseModel):
    """Response model for a single citation"""
    id: str
    text: str
    type: str
    reporter: Optional[str] = None
    volume: Optional[str] = None
    page: Optional[str] = None
    position_start: Optional[int] = None
    position_end: Optional[int] = None
    confidence_score: Optional[float] = None
    court_level: Optional[str] = None
    jurisdiction: Optional[str] = None
    precedential_strength: Optional[str] = None
    authority_score: Optional[float] = None


class CitationExtractionResponse(BaseModel):
    """Response model for citation extraction results"""
    document_id: str
    citations: List[CitationResponse]
    extraction_stats: Dict[str, Any]
    workflow_id: str
    processing_time_ms: int
    errors: List[str]


class CitationSearchRequest(BaseModel):
    """Request model for citation search"""
    query: str = Field(..., description="Search query for citations")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")


# API Endpoints

@router.post("/extract", response_model=CitationExtractionResponse)
async def extract_citations(
    request: CitationExtractionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract citations from a document using eyecite.
    
    This endpoint processes a document to extract all legal citations,
    optionally resolving reference citations and analyzing authority.
    """
    try:
        # Get the document
        document = await db.get(Document, request.document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {request.document_id} not found"
            )
        
        # Check if document has text content
        if not document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no extracted text content"
            )
        
        # Process options
        options = {
            "resolve_references": request.resolve_references,
            "create_embeddings": request.create_embeddings,
            "analyze_authority": request.analyze_authority,
            "create_relationships": request.create_relationships
        }
        
        # Run citation extraction workflow
        result = await citation_extractor_agent.process_document(
            document=document,
            db_session=db,
            options=options
        )
        
        # Format response
        citations = []
        authority_analysis = {a.citation.id: a for a in result.get("authority_analysis", [])}
        
        for citation in result.get("citations", []):
            analysis = authority_analysis.get(citation.id)
            citation_response = CitationResponse(
                id=str(citation.id),
                text=citation.citation_text,
                type=citation.citation_type,
                reporter=citation.reporter,
                volume=citation.volume,
                page=citation.page,
                position_start=citation.position_start,
                position_end=citation.position_end,
                confidence_score=citation.confidence_score,
                court_level=analysis.court_level if analysis else None,
                jurisdiction=analysis.jurisdiction if analysis else None,
                precedential_strength=analysis.precedential_strength if analysis else None,
                authority_score=analysis.authority_score if analysis else None
            )
            citations.append(citation_response)
        
        return CitationExtractionResponse(
            document_id=request.document_id,
            citations=citations,
            extraction_stats=result.get("extraction_result", {}).extraction_stats or {},
            workflow_id=str(result.get("workflow_id", "")),
            processing_time_ms=result.get("metrics", {}).get("extraction_time_ms", 0),
            errors=result.get("errors", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation extraction failed: {str(e)}"
        )


@router.get("/document/{document_id}", response_model=List[CitationResponse])
async def get_document_citations(
    document_id: str,
    include_authority: bool = Query(True, description="Include authority analysis"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all citations for a specific document.
    
    Returns previously extracted citations for a document,
    optionally including authority analysis.
    """
    try:
        # Validate document exists
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Get citations from database
        from sqlalchemy import select
        query = select(Citation).where(Citation.document_id == document_id)
        result = await db.execute(query)
        citations = result.scalars().all()
        
        # Format response
        citation_responses = []
        for citation in citations:
            citation_response = CitationResponse(
                id=str(citation.id),
                text=citation.citation_text,
                type=citation.citation_type,
                reporter=citation.reporter,
                volume=citation.volume,
                page=citation.page,
                position_start=citation.position_start,
                position_end=citation.position_end,
                confidence_score=citation.confidence_score
            )
            
            # Add authority analysis if requested
            if include_authority:
                try:
                    analysis = await citation_service.analyze_citation_authority(citation)
                    citation_response.court_level = analysis.court_level
                    citation_response.jurisdiction = analysis.jurisdiction
                    citation_response.precedential_strength = analysis.precedential_strength
                    citation_response.authority_score = analysis.authority_score
                except Exception as e:
                    logger.warning(f"Authority analysis failed for citation {citation.id}: {e}")
            
            citation_responses.append(citation_response)
        
        return citation_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get citations for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve citations: {str(e)}"
        )


@router.post("/search", response_model=List[CitationResponse])
async def search_citations(
    request: CitationSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search citations by text content.
    
    Performs text-based search across extracted citations,
    optionally limited to specific documents.
    """
    try:
        from sqlalchemy import select, or_, and_
        
        # Build search query
        query = select(Citation)
        
        # Add text search filter
        search_filters = [
            Citation.citation_text.ilike(f"%{request.query}%"),
            Citation.reporter.ilike(f"%{request.query}%")
        ]
        query = query.where(or_(*search_filters))
        
        # Add document filter if specified
        if request.document_ids:
            query = query.where(Citation.document_id.in_(request.document_ids))
        
        # Add limit
        query = query.limit(request.limit)
        
        # Execute query
        result = await db.execute(query)
        citations = result.scalars().all()
        
        # Format response
        citation_responses = []
        for citation in citations:
            citation_response = CitationResponse(
                id=str(citation.id),
                text=citation.citation_text,
                type=citation.citation_type,
                reporter=citation.reporter,
                volume=citation.volume,
                page=citation.page,
                position_start=citation.position_start,
                position_end=citation.position_end,
                confidence_score=citation.confidence_score
            )
            citation_responses.append(citation_response)
        
        return citation_responses
        
    except Exception as e:
        logger.error(f"Citation search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation search failed: {str(e)}"
        )


@router.get("/statistics")
async def get_citation_statistics(
    document_id: Optional[str] = Query(None, description="Get stats for specific document"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get citation extraction statistics.
    
    Returns overall statistics about citation extraction performance,
    optionally for a specific document.
    """
    try:
        from sqlalchemy import select, func
        
        # Get service statistics
        service_stats = citation_service.get_extraction_statistics()
        
        # Get database statistics
        if document_id:
            # Document-specific stats
            query = select(func.count(Citation.id)).where(Citation.document_id == document_id)
            result = await db.execute(query)
            citation_count = result.scalar() or 0
            
            stats = {
                "document_id": document_id,
                "total_citations": citation_count,
                **service_stats
            }
        else:
            # Overall stats
            query = select(func.count(Citation.id))
            result = await db.execute(query)
            total_citations = result.scalar() or 0
            
            query = select(func.count(Document.id.distinct())).select_from(
                Citation.__table__.join(Document.__table__)
            )
            result = await db.execute(query)
            documents_with_citations = result.scalar() or 0
            
            stats = {
                "total_citations": total_citations,
                "documents_with_citations": documents_with_citations,
                **service_stats
            }
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Failed to get citation statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/workflows")
async def get_citation_workflows(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of workflows"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent citation extraction workflows.
    
    Returns information about recent citation extraction workflows
    for monitoring and debugging purposes.
    """
    try:
        from sqlalchemy import select, desc
        from db.models import AgentType
        
        # Get recent citation extraction workflows
        query = (
            select(AgentWorkflow)
            .where(AgentWorkflow.agent_type == AgentType.CITATION_EXTRACTOR)
            .order_by(desc(AgentWorkflow.created_at))
            .limit(limit)
        )
        
        result = await db.execute(query)
        workflows = result.scalars().all()
        
        # Format response
        workflow_data = []
        for workflow in workflows:
            workflow_info = {
                "id": str(workflow.id),
                "status": workflow.status.value,
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "execution_time_ms": workflow.execution_time_ms,
                "quality_score": workflow.quality_score,
                "retry_count": workflow.retry_count,
                "input_data": workflow.input_data,
                "error_message": workflow.error_message
            }
            workflow_data.append(workflow_info)
        
        return JSONResponse(content={"workflows": workflow_data})
        
    except Exception as e:
        logger.error(f"Failed to get citation workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve workflows: {str(e)}"
        )