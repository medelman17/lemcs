"""
Citation Graph API endpoints.
Provides REST API for building, analyzing, and exporting citation graphs.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from db.database import get_db
from db.models import Document
from nlp.citation_graph import citation_graph_builder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/citation-graph", tags=["citation-graph"])


# Request/Response Models
class GraphBuildRequest(BaseModel):
    """Request model for building citation graphs"""
    document_ids: List[str] = Field(..., description="List of document UUIDs to include in graph")
    include_cross_document: bool = Field(False, description="Include citations from related documents")
    include_embeddings: bool = Field(False, description="Include citation embeddings in graph")


class SingleDocumentGraphRequest(BaseModel):
    """Request model for single document citation graph"""
    document_id: str = Field(..., description="Document UUID to build graph for")
    include_cross_document: bool = Field(False, description="Include citations from related documents")


class GraphExportRequest(BaseModel):
    """Request model for graph export"""
    document_ids: List[str] = Field(..., description="List of document UUIDs")
    format_type: str = Field("json", description="Export format: json, cytoscape, graphml, dot")
    include_cross_document: bool = Field(False, description="Include cross-document relationships")


class GraphAnalysisRequest(BaseModel):
    """Request model for graph analysis"""
    document_ids: List[str] = Field(..., description="List of document UUIDs to analyze")
    include_metrics: bool = Field(True, description="Include network analysis metrics")
    include_statistics: bool = Field(True, description="Include graph statistics")


# API Endpoints

@router.post("/build")
async def build_citation_graph(
    request: GraphBuildRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Build a citation graph from multiple documents.
    
    Creates a graph structure representing citation relationships
    across the specified documents.
    """
    try:
        # Validate documents exist
        for doc_id in request.document_ids:
            document = await db.get(Document, doc_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {doc_id} not found"
                )
        
        # Build the graph
        graph = await citation_graph_builder.build_graph_from_documents(
            document_ids=request.document_ids,
            db_session=db,
            include_embeddings=request.include_embeddings
        )
        
        # Get basic statistics
        stats = graph.get_statistics()
        
        return JSONResponse(content={
            "success": True,
            "graph_id": f"graph_{len(request.document_ids)}_docs",
            "statistics": stats,
            "message": f"Built citation graph with {stats['metadata']['node_count']} citations and {stats['metadata']['edge_count']} relationships"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build citation graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build citation graph: {str(e)}"
        )


@router.post("/build/document/{document_id}")
async def build_document_citation_graph(
    document_id: str,
    include_cross_document: bool = Query(False, description="Include cross-document relationships"),
    db: AsyncSession = Depends(get_db)
):
    """
    Build a citation graph for a single document.
    
    Creates a graph showing all citation relationships within a document,
    optionally including relationships to other documents.
    """
    try:
        # Validate document exists
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Build the graph
        graph = await citation_graph_builder.build_document_citation_graph(
            document_id=document_id,
            db_session=db,
            include_cross_document=include_cross_document
        )
        
        # Get basic statistics
        stats = graph.get_statistics()
        
        return JSONResponse(content={
            "success": True,
            "document_id": document_id,
            "statistics": stats,
            "message": f"Built citation graph with {stats['metadata']['node_count']} citations and {stats['metadata']['edge_count']} relationships"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build document citation graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build document citation graph: {str(e)}"
        )


@router.post("/export")
async def export_citation_graph(
    request: GraphExportRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Export citation graph in specified format for visualization.
    
    Supports multiple export formats:
    - json: D3.js compatible format
    - cytoscape: Cytoscape.js format
    - graphml: GraphML XML format
    - dot: Graphviz DOT format
    """
    try:
        # Validate documents exist
        for doc_id in request.document_ids:
            document = await db.get(Document, doc_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {doc_id} not found"
                )
        
        # Validate format
        supported_formats = ["json", "cytoscape", "graphml", "dot"]
        if request.format_type not in supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {request.format_type}. Supported: {supported_formats}"
            )
        
        # Build the graph
        graph = await citation_graph_builder.build_graph_from_documents(
            document_ids=request.document_ids,
            db_session=db
        )
        
        # Export in requested format
        exported_data = graph.export_for_visualization(request.format_type)
        
        return JSONResponse(content={
            "success": True,
            "format": request.format_type,
            "graph_data": exported_data,
            "metadata": {
                "node_count": graph.metadata["node_count"],
                "edge_count": graph.metadata["edge_count"],
                "document_count": len(request.document_ids)
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export citation graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export citation graph: {str(e)}"
        )


@router.post("/analyze")
async def analyze_citation_graph(
    request: GraphAnalysisRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Perform network analysis on citation graph.
    
    Computes various graph metrics including centrality measures,
    community detection, and citation-specific analytics.
    """
    try:
        # Validate documents exist
        for doc_id in request.document_ids:
            document = await db.get(Document, doc_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document {doc_id} not found"
                )
        
        # Build the graph
        graph = await citation_graph_builder.build_graph_from_documents(
            document_ids=request.document_ids,
            db_session=db
        )
        
        analysis_result = {}
        
        # Include network metrics if requested
        if request.include_metrics:
            analysis_result["network_metrics"] = graph.analyze_network_metrics()
        
        # Include statistics if requested
        if request.include_statistics:
            analysis_result["statistics"] = graph.get_statistics()
        
        # Add summary
        analysis_result["summary"] = {
            "total_citations": graph.metadata["node_count"],
            "total_relationships": graph.metadata["edge_count"],
            "documents_analyzed": len(request.document_ids),
            "relationship_types": list(graph.metadata["relationship_types"])
        }
        
        return JSONResponse(content={
            "success": True,
            "analysis": analysis_result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze citation graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze citation graph: {str(e)}"
        )


@router.get("/citation/{citation_id}/neighbors")
async def get_citation_neighbors(
    citation_id: str,
    direction: str = Query("both", description="Direction: in, out, or both"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get neighboring citations for a specific citation.
    
    Returns citations that reference or are referenced by the given citation.
    """
    try:
        # First, find which document this citation belongs to
        from sqlalchemy import select
        citation_query = select(Citation).where(Citation.id == citation_id)
        citation_result = await db.execute(citation_query)
        citation = citation_result.scalar_one_or_none()
        
        if not citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Citation {citation_id} not found"
            )
        
        # Build graph for the document
        graph = await citation_graph_builder.build_document_citation_graph(
            document_id=str(citation.document_id),
            db_session=db,
            include_cross_document=True
        )
        
        # Get neighbors
        neighbors = graph.get_neighbors(citation_id, direction)
        
        # Format response
        neighbor_data = []
        for neighbor in neighbors:
            neighbor_data.append({
                "citation_id": neighbor.citation_id,
                "citation_text": neighbor.citation_text,
                "citation_type": neighbor.citation_type,
                "document_id": neighbor.document_id,
                "confidence_score": neighbor.confidence_score
            })
        
        return JSONResponse(content={
            "success": True,
            "citation_id": citation_id,
            "direction": direction,
            "neighbors": neighbor_data,
            "neighbor_count": len(neighbor_data)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get citation neighbors: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get citation neighbors: {str(e)}"
        )


@router.get("/citation/{citation_id}/chains")
async def get_citation_chains(
    citation_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum chain depth"),
    db: AsyncSession = Depends(get_db)
):
    """
    Find citation chains starting from a specific citation.
    
    Returns paths of citation relationships that form logical chains.
    """
    try:
        # First, find which document this citation belongs to
        from sqlalchemy import select
        citation_query = select(Citation).where(Citation.id == citation_id)
        citation_result = await db.execute(citation_query)
        citation = citation_result.scalar_one_or_none()
        
        if not citation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Citation {citation_id} not found"
            )
        
        # Build graph for the document
        graph = await citation_graph_builder.build_document_citation_graph(
            document_id=str(citation.document_id),
            db_session=db,
            include_cross_document=True
        )
        
        # Get citation chains
        chains = graph.get_citation_chains(citation_id, max_depth)
        
        # Format response
        chain_data = []
        for chain in chains:
            chain_info = {
                "length": len(chain),
                "citations": [
                    {
                        "citation_id": node.citation_id,
                        "citation_text": node.citation_text,
                        "citation_type": node.citation_type,
                        "position": i
                    }
                    for i, node in enumerate(chain)
                ]
            }
            chain_data.append(chain_info)
        
        return JSONResponse(content={
            "success": True,
            "citation_id": citation_id,
            "max_depth": max_depth,
            "chains": chain_data,
            "chain_count": len(chain_data)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get citation chains: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get citation chains: {str(e)}"
        )


@router.get("/health")
async def citation_graph_health():
    """Health check for citation graph service"""
    try:
        # Test graph creation
        from nlp.citation_graph import CitationGraph
        test_graph = CitationGraph()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "citation_graph",
            "features": [
                "graph_building",
                "network_analysis", 
                "visualization_export",
                "neighbor_discovery",
                "chain_analysis"
            ]
        })
        
    except Exception as e:
        logger.error(f"Citation graph health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service unhealthy: {str(e)}"
        ) 