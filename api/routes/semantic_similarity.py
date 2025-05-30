"""
Semantic Similarity API endpoints.
Provides REST API for semantic analysis and matching of legal citations.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import uuid

from db.database import get_db
from db.models import Document, Citation
from nlp.semantic_similarity import semantic_similarity_service, SemanticMatch, CitationContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/semantic", tags=["semantic_similarity"])


# Request/Response Models
class CitationContextRequest(BaseModel):
    """Request model for extracting citation context"""
    citation_id: str = Field(..., description="UUID of the citation to analyze")
    context_window: int = Field(500, ge=100, le=2000, description="Number of characters around citation to extract")


class CitationContextResponse(BaseModel):
    """Response model for citation context"""
    citation_id: str
    citation_text: str
    surrounding_text: str
    extracted_entities: List[str]
    legal_concepts: List[str]
    case_names: List[str]
    statutory_references: List[str]
    position_in_document: float


class EmbeddingRequest(BaseModel):
    """Request model for generating citation embeddings"""
    citation_id: str = Field(..., description="UUID of the citation to embed")
    include_surrounding_context: bool = Field(True, description="Whether to include surrounding text in embedding")


class EmbeddingResponse(BaseModel):
    """Response model for citation embedding"""
    citation_id: str
    embedding_dimension: int
    cache_hit: bool
    embedding_text_preview: str  # First 200 chars of text used for embedding


class SimilarityRequest(BaseModel):
    """Request model for calculating semantic similarity"""
    source_citation_id: str = Field(..., description="UUID of source citation")
    target_citation_id: str = Field(..., description="UUID of target citation")
    include_context: bool = Field(True, description="Whether to include surrounding context in similarity calculation")


class SimilarityResponse(BaseModel):
    """Response model for semantic similarity result"""
    source_citation_id: str
    target_citation_id: str
    similarity_score: float
    context_overlap: float
    combined_confidence: float
    match_reason: str
    semantic_features: Dict[str, Any]


class SemanticMatchRequest(BaseModel):
    """Request model for finding semantic matches"""
    source_citation_id: str = Field(..., description="UUID of citation to find matches for")
    document_id: Optional[str] = Field(None, description="Limit search to specific document")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    max_matches: int = Field(5, ge=1, le=20, description="Maximum number of matches to return")


class SemanticMatchResponse(BaseModel):
    """Response model for semantic match result"""
    source_citation_id: str
    target_citation_id: str
    target_citation_text: str
    similarity_score: float
    context_overlap: float
    combined_confidence: float
    match_reason: str
    semantic_features: Dict[str, Any]


# API Endpoints

@router.post("/context", response_model=CitationContextResponse)
async def extract_citation_context(
    request: CitationContextRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Extract semantic context for a citation.
    
    Analyzes the surrounding text of a citation to extract legal entities,
    concepts, case names, and statutory references.
    """
    try:
        # Get the citation with its document
        citation = await db.get(Citation, request.citation_id)
        if not citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Citation {request.citation_id} not found"
            )
        
        # Get the document for full text
        document = await db.get(Document, citation.document_id)
        if not document or not document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document text not available for context extraction"
            )
        
        # Extract context
        context = await semantic_similarity_service.extract_citation_context(
            citation=citation,
            document_text=document.extracted_text,
            context_window=request.context_window
        )
        
        return CitationContextResponse(
            citation_id=context.citation_id,
            citation_text=context.citation_text,
            surrounding_text=context.surrounding_text,
            extracted_entities=context.extracted_entities,
            legal_concepts=context.legal_concepts,
            case_names=context.case_names,
            statutory_references=context.statutory_references,
            position_in_document=context.position_in_document
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context extraction failed: {str(e)}"
        )


@router.post("/embedding", response_model=EmbeddingResponse)
async def generate_citation_embedding(
    request: EmbeddingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate semantic embedding for a citation.
    
    Creates a vector embedding using OpenAI that captures the semantic
    meaning of the citation and optionally its surrounding context.
    """
    try:
        # Get the citation with its document
        citation = await db.get(Citation, request.citation_id)
        if not citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Citation {request.citation_id} not found"
            )
        
        # Get the document for full text
        document = await db.get(Document, citation.document_id)
        if not document or not document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document text not available for embedding generation"
            )
        
        # Extract context first
        context = await semantic_similarity_service.extract_citation_context(
            citation=citation,
            document_text=document.extracted_text
        )
        
        # Check if embedding is cached
        cache_key = f"{context.citation_id}_{request.include_surrounding_context}"
        cache_hit = cache_key in semantic_similarity_service.embedding_cache
        
        # Generate embedding
        embedding = await semantic_similarity_service.generate_citation_embedding(
            citation_context=context,
            include_surrounding_context=request.include_surrounding_context
        )
        
        # Prepare text preview
        if request.include_surrounding_context:
            preview_text = semantic_similarity_service._prepare_embedding_text(context)[:200]
        else:
            preview_text = context.citation_text[:200]
        
        return EmbeddingResponse(
            citation_id=request.citation_id,
            embedding_dimension=len(embedding),
            cache_hit=cache_hit,
            embedding_text_preview=preview_text + "..." if len(preview_text) == 200 else preview_text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_semantic_similarity(
    request: SimilarityRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate semantic similarity between two citations.
    
    Computes multiple similarity metrics including semantic embedding similarity,
    context overlap, and combined confidence scores.
    """
    try:
        # Get source citation
        source_citation = await db.get(Citation, request.source_citation_id)
        if not source_citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source citation {request.source_citation_id} not found"
            )
        
        # Get target citation
        target_citation = await db.get(Citation, request.target_citation_id)
        if not target_citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target citation {request.target_citation_id} not found"
            )
        
        # Get documents for both citations
        source_document = await db.get(Document, source_citation.document_id)
        target_document = await db.get(Document, target_citation.document_id)
        
        if not source_document or not source_document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source document text not available"
            )
        
        if not target_document or not target_document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Target document text not available"
            )
        
        # Extract contexts
        source_context = await semantic_similarity_service.extract_citation_context(
            citation=source_citation,
            document_text=source_document.extracted_text
        )
        
        target_context = await semantic_similarity_service.extract_citation_context(
            citation=target_citation,
            document_text=target_document.extracted_text
        )
        
        # Calculate similarity
        semantic_match = await semantic_similarity_service.calculate_semantic_similarity(
            source_context=source_context,
            target_context=target_context,
            include_context=request.include_context
        )
        
        return SimilarityResponse(
            source_citation_id=semantic_match.source_citation_id,
            target_citation_id=semantic_match.target_citation_id,
            similarity_score=semantic_match.similarity_score,
            context_overlap=semantic_match.context_overlap,
            combined_confidence=semantic_match.combined_confidence,
            match_reason=semantic_match.match_reason,
            semantic_features=semantic_match.semantic_features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity calculation failed: {str(e)}"
        )


@router.post("/matches", response_model=List[SemanticMatchResponse])
async def find_semantic_matches(
    request: SemanticMatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Find semantically similar citations.
    
    Searches for citations that are semantically similar to a source citation,
    useful for finding related cases, duplicate citations, or references to
    the same legal authority expressed differently.
    """
    try:
        # Get source citation
        source_citation = await db.get(Citation, request.source_citation_id)
        if not source_citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source citation {request.source_citation_id} not found"
            )
        
        # Get source document
        source_document = await db.get(Document, source_citation.document_id)
        if not source_document or not source_document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source document text not available"
            )
        
        # Build query for candidate citations
        from sqlalchemy import select
        query = select(Citation)
        
        # If document_id specified, limit to that document
        if request.document_id:
            query = query.where(Citation.document_id == request.document_id)
            # Also get that document's text
            target_document = await db.get(Document, request.document_id)
            if not target_document or not target_document.extracted_text:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Target document text not available"
                )
            document_text = target_document.extracted_text
        else:
            # Use source document for all candidates (same document matching)
            query = query.where(Citation.document_id == source_citation.document_id)
            document_text = source_document.extracted_text
        
        # Execute query
        result = await db.execute(query)
        candidate_citations = result.scalars().all()
        
        # Find semantic matches
        matches = await semantic_similarity_service.find_semantic_matches(
            source_citation=source_citation,
            candidate_citations=candidate_citations,
            document_text=document_text,
            threshold=request.threshold,
            max_matches=request.max_matches
        )
        
        # Format response
        match_responses = []
        for match in matches:
            # Get target citation text
            target_citation = next(
                (c for c in candidate_citations if str(c.id) == match.target_citation_id),
                None
            )
            
            if target_citation:
                match_response = SemanticMatchResponse(
                    source_citation_id=match.source_citation_id,
                    target_citation_id=match.target_citation_id,
                    target_citation_text=target_citation.citation_text,
                    similarity_score=match.similarity_score,
                    context_overlap=match.context_overlap,
                    combined_confidence=match.combined_confidence,
                    match_reason=match.match_reason,
                    semantic_features=match.semantic_features
                )
                match_responses.append(match_response)
        
        return match_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic match search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic match search failed: {str(e)}"
        )


@router.post("/cache/cleanup")
async def cleanup_cache():
    """
    Clean up expired cache entries.
    
    Manually trigger cache cleanup to free memory. This is also
    done automatically based on cache expiry settings.
    """
    try:
        semantic_similarity_service.cleanup_cache()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Cache cleanup completed",
            "embedding_cache_cleared": True,
            "context_cache_cleared": True
        })
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache cleanup failed: {str(e)}"
        )


@router.get("/statistics")
async def get_semantic_statistics():
    """
    Get semantic similarity service statistics.
    
    Returns information about cache usage and service performance.
    """
    try:
        stats = {
            "embedding_cache_size": len(semantic_similarity_service.embedding_cache),
            "context_cache_size": len(semantic_similarity_service.context_cache),
            "cache_expiry_hours": semantic_similarity_service.cache_expiry.total_seconds() / 3600,
            "last_cache_cleanup": semantic_similarity_service.last_cache_cleanup.isoformat(),
            "supported_legal_concepts": list(semantic_similarity_service.legal_concept_patterns.keys()),
            "embedding_dimension": 1536  # OpenAI embedding size
        }
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Failed to get semantic statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )