"""
Citation Embeddings API endpoints.
Provides REST API for managing citation embeddings including batch processing,
clustering, model refresh, and similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from db.database import get_db
from nlp.citation_embedding_service import citation_embedding_service
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/citation-embeddings", tags=["citation-embeddings"])


# Request/Response Models
class BatchProcessingRequest(BaseModel):
    """Request model for batch embedding processing"""
    force_refresh: bool = Field(False, description="Whether to refresh existing embeddings")
    include_context: bool = Field(True, description="Whether to include legal context")
    max_citations: Optional[int] = Field(None, description="Maximum citations to process (for testing)")


class ModelRefreshRequest(BaseModel):
    """Request model for embedding model refresh"""
    new_model_version: str = Field(..., description="New embedding model version")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to refresh")


class ClusteringRequest(BaseModel):
    """Request model for citation clustering"""
    document_ids: Optional[List[str]] = Field(None, description="Documents to cluster")
    clustering_method: str = Field("kmeans", description="Clustering method: 'kmeans' or 'dbscan'")
    n_clusters: Optional[int] = Field(None, description="Number of clusters for k-means")
    similarity_threshold: float = Field(0.8, description="Similarity threshold for clustering")


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search"""
    citation_id: str = Field(..., description="Target citation ID")
    threshold: float = Field(0.8, description="Minimum similarity threshold")
    max_results: int = Field(10, description="Maximum number of results")


@router.post("/batch-process")
async def batch_process_citations(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Process all citations in batch to create embeddings.
    
    This endpoint processes citations that don't have embeddings yet,
    or refreshes all embeddings if force_refresh is True.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        # Execute batch processing
        result = await citation_embedding_service.process_all_citations_batch(
            db_session=db,
            force_refresh=request.force_refresh,
            include_context=request.include_context,
            max_citations=request.max_citations
        )
        
        return JSONResponse(content={
            "message": f"Batch processing completed",
            "total_citations": result.total_citations,
            "processed_citations": result.processed_citations,
            "skipped_citations": result.skipped_citations,
            "failed_citations": result.failed_citations,
            "total_tokens": result.total_tokens,
            "total_cost_usd": round(result.total_cost_usd, 4),
            "processing_time_ms": result.processing_time_ms,
            "error_count": len(result.errors),
            "errors": result.errors[:10] if result.errors else []  # Limit error list
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.post("/refresh-model")
async def refresh_embeddings_for_model(
    request: ModelRefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh embeddings when the embedding model is updated.
    
    This deletes existing embeddings and recreates them with the new model.
    """
    try:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenAI API not configured"
            )
        
        result = await citation_embedding_service.refresh_embeddings_for_model_update(
            db_session=db,
            new_model_version=request.new_model_version,
            document_ids=request.document_ids
        )
        
        return JSONResponse(content={
            "message": f"Model refresh completed",
            "citations_refreshed": result.citations_refreshed,
            "embeddings_deleted": result.embeddings_deleted,
            "embeddings_created": result.embeddings_created,
            "total_tokens": result.total_tokens,
            "total_cost_usd": round(result.total_cost_usd, 4),
            "processing_time_ms": result.processing_time_ms,
            "new_model_version": request.new_model_version
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model refresh failed: {str(e)}"
        )


@router.post("/cluster")
async def cluster_citations(
    request: ClusteringRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Cluster citations by semantic similarity.
    
    Groups similar citations together using k-means or DBSCAN clustering
    on their embedding vectors.
    """
    try:
        result = await citation_embedding_service.cluster_citations_by_similarity(
            db_session=db,
            document_ids=request.document_ids,
            clustering_method=request.clustering_method,
            n_clusters=request.n_clusters,
            similarity_threshold=request.similarity_threshold
        )
        
        return JSONResponse(content={
            "message": f"Clustering completed using {request.clustering_method}",
            "cluster_count": result.cluster_count,
            "citation_clusters": result.citation_clusters,
            "cluster_metadata": result.cluster_metadata,
            "silhouette_score": result.silhouette_score,
            "inertia": result.inertia,
            "algorithm": request.clustering_method
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Citation clustering failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation clustering failed: {str(e)}"
        )


@router.post("/find-similar")
async def find_similar_citations(
    request: SimilaritySearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Find citations similar to a given citation.
    
    Uses cosine similarity on embedding vectors to find semantically
    similar citations.
    """
    try:
        similar_citations = await citation_embedding_service.find_similar_citations(
            db_session=db,
            citation_id=request.citation_id,
            threshold=request.threshold,
            max_results=request.max_results
        )
        
        return JSONResponse(content={
            "message": f"Found {len(similar_citations)} similar citations",
            "target_citation_id": request.citation_id,
            "threshold": request.threshold,
            "similar_citations": similar_citations
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/statistics")
async def get_embedding_statistics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive statistics about citation embeddings.
    
    Returns information about embedding coverage, recent activity,
    and system configuration.
    """
    try:
        stats = await citation_embedding_service.get_embedding_statistics(db)
        
        return JSONResponse(content={
            "message": "Embedding statistics retrieved successfully",
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get embedding statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedding statistics: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check for citation embedding service.
    
    Returns the status of the embedding service and configuration.
    """
    try:
        # Check if OpenAI is configured
        openai_configured = bool(settings.OPENAI_API_KEY)
        
        # Get service configuration
        service_config = {
            "batch_size": citation_embedding_service.batch_size,
            "max_context_chars": citation_embedding_service.max_context_chars,
            "embedding_model_version": citation_embedding_service.embedding_model_version
        }
        
        health_status = {
            "service": "citation-embeddings",
            "status": "healthy",
            "openai_configured": openai_configured,
            "configuration": service_config
        }
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "service": "citation-embeddings",
                "status": "unhealthy",
                "error": str(e)
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/citations/{citation_id}/context-analysis")
async def analyze_citation_context(
    citation_id: str,
    include_metadata: bool = Query(True, description="Include citation metadata"),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze the context and semantic features of a specific citation.
    
    Returns detailed analysis including legal concepts, entities,
    and contextual information used for embedding generation.
    """
    try:
        from sqlalchemy import select
        from db.models import Citation, Document
        from nlp.semantic_similarity import semantic_similarity_service
        
        # Get citation and its document
        citation_query = (
            select(Citation, Document)
            .join(Document, Citation.document_id == Document.id)
            .where(Citation.id == citation_id)
        )
        result = await db.execute(citation_query)
        citation_pair = result.first()
        
        if not citation_pair:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Citation {citation_id} not found"
            )
        
        citation, document = citation_pair
        
        # Extract context using semantic similarity service
        context = await semantic_similarity_service.extract_citation_context(
            citation=citation,
            document_text=document.extracted_text,
            context_window=500
        )
        
        # Prepare response
        analysis = {
            "citation_id": citation_id,
            "citation_text": citation.citation_text,
            "citation_type": citation.citation_type,
            "context": {
                "surrounding_text": context.surrounding_text[:200] + "..." if len(context.surrounding_text) > 200 else context.surrounding_text,
                "extracted_entities": context.extracted_entities,
                "legal_concepts": context.legal_concepts,
                "case_names": context.case_names,
                "statutory_references": context.statutory_references,
                "position_in_document": context.position_in_document
            }
        }
        
        if include_metadata and citation.doc_metadata:
            analysis["metadata"] = citation.doc_metadata
        
        return JSONResponse(content={
            "message": "Citation context analysis completed",
            "analysis": analysis
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Citation context analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Citation context analysis failed: {str(e)}"
        ) 