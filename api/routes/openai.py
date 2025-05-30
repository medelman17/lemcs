"""
OpenAI API integration endpoints.
Provides endpoints for testing, monitoring, and managing OpenAI embedding services.
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from db.database import get_db
from db.models import Citation
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/openai", tags=["openai"])


# Request/Response Models
class EmbeddingTestRequest(BaseModel):
    """Request model for testing embedding creation"""
    text: str = Field(..., description="Text to create embedding for", max_length=8191)
    use_cache: bool = Field(True, description="Whether to use cache for this request")


class EmbeddingTestResponse(BaseModel):
    """Response model for embedding test"""
    text: str
    embedding_dimensions: int
    tokens_used: int
    cached: bool
    processing_time_ms: int
    cost_usd: float
    model: str


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding creation"""
    texts: List[str] = Field(..., description="List of texts to embed", max_items=100)
    use_cache: bool = Field(True, description="Whether to use cache")
    batch_size: Optional[int] = Field(None, description="Override default batch size")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding creation"""
    results: List[EmbeddingTestResponse]
    total_tokens: int
    total_cost_usd: float
    processing_time_ms: int
    cached_count: int


class CitationEmbeddingRequest(BaseModel):
    """Request model for citation embedding creation"""
    citation_ids: List[str] = Field(..., description="List of citation IDs to embed")
    include_context: bool = Field(True, description="Include citation context in embedding")
    force_refresh: bool = Field(False, description="Force refresh even if embeddings exist")


# API Endpoints

@router.get("/health")
async def get_openai_health():
    """
    Check OpenAI service health and connectivity.
    
    Returns connectivity status, configuration, and basic statistics.
    """
    try:
        # Check if OpenAI is configured
        if not settings.OPENAI_API_KEY:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "service": "OpenAI Embedding Service",
                    "status": "unavailable",
                    "message": "OPENAI_API_KEY not configured",
                    "configured": False
                }
            )
        
        # Import and test the service
        from nlp.openai_service import openai_service
        health_info = await openai_service.health_check()
        
        return JSONResponse(content=health_info)
        
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "service": "OpenAI Embedding Service",
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "configured": bool(settings.OPENAI_API_KEY)
            }
        )


@router.get("/statistics")
async def get_openai_statistics():
    """
    Get OpenAI service usage statistics.
    
    Returns detailed statistics about API usage, costs, and performance.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        from nlp.openai_service import openai_service
        stats = openai_service.get_statistics()
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Failed to get OpenAI statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.post("/test-embedding", response_model=EmbeddingTestResponse)
async def test_embedding_creation(request: EmbeddingTestRequest):
    """
    Test embedding creation for a single text.
    
    Useful for testing API connectivity and understanding token usage.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        from nlp.openai_service import openai_service
        
        # Create embedding
        result = await openai_service.create_embedding(
            text=request.text,
            use_cache=request.use_cache
        )
        
        # Calculate cost
        cost_usd = (result.tokens_used / 1000) * 0.00002  # text-embedding-3-small pricing
        
        return EmbeddingTestResponse(
            text=result.text,
            embedding_dimensions=len(result.embedding),
            tokens_used=result.tokens_used,
            cached=result.cached,
            processing_time_ms=result.processing_time_ms,
            cost_usd=cost_usd,
            model=result.model
        )
        
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding test failed: {str(e)}"
        )


@router.post("/batch-embeddings", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    """
    Create embeddings for multiple texts in batch.
    
    Efficient for processing multiple texts with rate limiting and caching.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 texts per batch request"
            )
        
        from nlp.openai_service import openai_service
        
        # Create batch embeddings
        batch_result = await openai_service.create_embeddings_batch(
            texts=request.texts,
            use_cache=request.use_cache,
            batch_size=request.batch_size
        )
        
        # Format individual results
        individual_results = []
        for result in batch_result.results:
            cost_usd = (result.tokens_used / 1000) * 0.00002 if result.tokens_used else 0
            
            individual_results.append(EmbeddingTestResponse(
                text=result.text,
                embedding_dimensions=len(result.embedding),
                tokens_used=result.tokens_used,
                cached=result.cached,
                processing_time_ms=result.processing_time_ms,
                cost_usd=cost_usd,
                model=result.model
            ))
        
        return BatchEmbeddingResponse(
            results=individual_results,
            total_tokens=batch_result.total_tokens,
            total_cost_usd=batch_result.total_cost_usd,
            processing_time_ms=batch_result.processing_time_ms,
            cached_count=batch_result.cached_count
        )
        
    except Exception as e:
        logger.error(f"Batch embedding creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding creation failed: {str(e)}"
        )


@router.post("/citations/embeddings")
async def create_citation_embeddings(
    request: CitationEmbeddingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create embeddings for specific citations.
    
    Creates embeddings for legal citations with legal-specific context.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        # Get citations from database
        from sqlalchemy import select
        
        query = select(Citation).where(Citation.id.in_(request.citation_ids))
        result = await db.execute(query)
        citations = result.scalars().all()
        
        if not citations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No citations found with provided IDs"
            )
        
        # Check which citations already have embeddings (unless force refresh)
        if not request.force_refresh:
            from db.models import CitationEmbedding
            
            existing_query = select(CitationEmbedding).where(
                CitationEmbedding.citation_id.in_(request.citation_ids)
            )
            existing_result = await db.execute(existing_query)
            existing_embeddings = existing_result.scalars().all()
            existing_citation_ids = {e.citation_id for e in existing_embeddings}
            
            # Filter out citations that already have embeddings
            citations = [c for c in citations if c.id not in existing_citation_ids]
            
            if not citations:
                return JSONResponse(content={
                    "message": "All requested citations already have embeddings",
                    "existing_count": len(existing_embeddings),
                    "created_count": 0
                })
        
        # Create embeddings
        from nlp.openai_service import openai_service
        from db.models import CitationEmbedding
        
        embedding_results = await openai_service.create_citation_embeddings(
            citations=citations,
            include_context=request.include_context
        )
        
        # Save to database
        citation_embeddings = []
        for citation, embedding_result in zip(citations, embedding_results):
            citation_embedding = CitationEmbedding(
                citation_id=citation.id,
                embedding=embedding_result.embedding,
                created_at=datetime.utcnow()
            )
            citation_embeddings.append(citation_embedding)
            db.add(citation_embedding)
        
        await db.commit()
        
        # Calculate statistics
        total_tokens = sum(r.tokens_used for r in embedding_results)
        total_cost = (total_tokens / 1000) * 0.00002
        cached_count = sum(1 for r in embedding_results if r.cached)
        
        return JSONResponse(content={
            "message": f"Created embeddings for {len(citation_embeddings)} citations",
            "created_count": len(citation_embeddings),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "cached_count": cached_count,
            "citation_ids": [str(ce.citation_id) for ce in citation_embeddings]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Citation embedding creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation embedding creation failed: {str(e)}"
        )


@router.delete("/cache")
async def clear_embedding_cache():
    """
    Clear the OpenAI embedding cache.
    
    Useful for testing or when embeddings need to be regenerated.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        from nlp.openai_service import openai_service
        
        if not openai_service.redis_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Redis cache not configured"
            )
        
        # Clear embeddings from cache
        async for key in openai_service.redis_client.scan_iter("openai_embedding:*"):
            await openai_service.redis_client.delete(key)
        
        return JSONResponse(content={
            "message": "OpenAI embedding cache cleared",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clearing failed: {str(e)}"
        ) 