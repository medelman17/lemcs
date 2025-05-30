"""
Citation Embedding Service for LeMCS.
Provides comprehensive citation embedding functionality including batch processing,
model refresh capabilities, and semantic similarity clustering.
"""
import logging
import asyncio
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_, or_, func, text
from sqlalchemy.orm import selectinload

from db.models import Citation, CitationEmbedding, Document
from nlp.openai_service import openai_service
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingResult:
    """Result of batch embedding processing"""
    total_citations: int
    processed_citations: int
    skipped_citations: int
    failed_citations: int
    total_tokens: int
    total_cost_usd: float
    processing_time_ms: int
    errors: List[str]


@dataclass
class ClusteringResult:
    """Result of citation clustering analysis"""
    cluster_count: int
    citation_clusters: Dict[int, List[str]]  # cluster_id -> citation_ids
    cluster_metadata: Dict[int, Dict[str, Any]]  # cluster_id -> metadata
    silhouette_score: Optional[float]
    inertia: Optional[float]


@dataclass
class RefreshResult:
    """Result of embedding refresh operation"""
    citations_refreshed: int
    embeddings_deleted: int
    embeddings_created: int
    total_tokens: int
    total_cost_usd: float
    processing_time_ms: int


class CitationEmbeddingService:
    """
    Service for managing citation embeddings with advanced features:
    - Batch processing of existing citations
    - Model refresh capabilities  
    - Semantic similarity clustering
    - Context optimization
    """
    
    def __init__(self):
        self.batch_size = 100  # Citations per batch
        self.max_context_chars = 500  # Context window size
        self.clustering_cache = {}  # Cache clustering results
        self.embedding_model_version = "text-embedding-3-small"  # Current model
    
    async def process_all_citations_batch(
        self,
        db_session: AsyncSession,
        force_refresh: bool = False,
        include_context: bool = True,
        max_citations: Optional[int] = None
    ) -> BatchProcessingResult:
        """
        Process all citations in the database to create embeddings in batches
        
        Args:
            db_session: Database session
            force_refresh: Whether to refresh existing embeddings
            include_context: Whether to include legal context in embeddings
            max_citations: Maximum number of citations to process (for testing)
        """
        start_time = datetime.now()
        total_citations = 0
        processed_citations = 0
        skipped_citations = 0
        failed_citations = 0
        total_tokens = 0
        total_cost_usd = 0.0
        errors = []
        
        try:
            # Query citations that need embedding processing
            if force_refresh:
                # Process all citations
                citation_query = select(Citation)
                if max_citations:
                    citation_query = citation_query.limit(max_citations)
            else:
                # Only process citations without embeddings
                citation_query = (
                    select(Citation)
                    .outerjoin(CitationEmbedding)
                    .where(CitationEmbedding.citation_id.is_(None))
                )
                if max_citations:
                    citation_query = citation_query.limit(max_citations)
            
            result = await db_session.execute(citation_query)
            citations_to_process = result.scalars().all()
            total_citations = len(citations_to_process)
            
            if total_citations == 0:
                logger.info("No citations found that need embedding processing")
                return BatchProcessingResult(
                    total_citations=0,
                    processed_citations=0,
                    skipped_citations=0,
                    failed_citations=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    processing_time_ms=0,
                    errors=[]
                )
            
            logger.info(f"Starting batch processing of {total_citations} citations")
            
            # Process in batches
            for i in range(0, total_citations, self.batch_size):
                batch_citations = citations_to_process[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                logger.info(f"Processing batch {batch_num} ({len(batch_citations)} citations)")
                
                try:
                    # Delete existing embeddings if force refresh
                    if force_refresh:
                        citation_ids = [str(c.id) for c in batch_citations]
                        delete_query = delete(CitationEmbedding).where(
                            CitationEmbedding.citation_id.in_(citation_ids)
                        )
                        await db_session.execute(delete_query)
                    
                    # Create embeddings for this batch
                    embedding_results = await openai_service.create_citation_embeddings(
                        citations=batch_citations,
                        include_context=include_context
                    )
                    
                    # Create CitationEmbedding records
                    batch_embeddings = []
                    for citation, embedding_result in zip(batch_citations, embedding_results):
                        if embedding_result.embedding:  # Check if embedding was successful
                            citation_embedding = CitationEmbedding(
                                citation_id=citation.id,
                                embedding=embedding_result.embedding,
                                created_at=datetime.utcnow()
                            )
                            batch_embeddings.append(citation_embedding)
                            db_session.add(citation_embedding)
                            processed_citations += 1
                        else:
                            failed_citations += 1
                            errors.append(f"Failed to create embedding for citation {citation.id}")
                    
                    # Calculate batch statistics
                    batch_tokens = sum(r.tokens_used for r in embedding_results)
                    batch_cost = (batch_tokens / 1000) * 0.00002  # text-embedding-3-small pricing
                    
                    total_tokens += batch_tokens
                    total_cost_usd += batch_cost
                    
                    # Commit this batch
                    await db_session.commit()
                    
                    logger.info(f"Batch {batch_num} completed: {len(batch_embeddings)} embeddings created")
                    
                    # Brief pause between batches to avoid rate limiting
                    if i + self.batch_size < total_citations:
                        await asyncio.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Batch {batch_num} failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    failed_citations += len(batch_citations)
                    
                    # Rollback this batch and continue
                    await db_session.rollback()
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Batch processing completed: {processed_citations}/{total_citations} processed")
            
            return BatchProcessingResult(
                total_citations=total_citations,
                processed_citations=processed_citations,
                skipped_citations=skipped_citations,
                failed_citations=failed_citations,
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                processing_time_ms=int(processing_time),
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    async def refresh_embeddings_for_model_update(
        self,
        db_session: AsyncSession,
        new_model_version: str,
        document_ids: Optional[List[str]] = None
    ) -> RefreshResult:
        """
        Refresh embeddings when the underlying model is updated
        
        Args:
            db_session: Database session
            new_model_version: New embedding model version
            document_ids: Optional list of specific documents to refresh
        """
        start_time = datetime.now()
        
        try:
            # Update model version
            old_model_version = self.embedding_model_version
            self.embedding_model_version = new_model_version
            
            # Query embeddings to refresh
            if document_ids:
                # Refresh embeddings for specific documents
                embeddings_query = (
                    select(CitationEmbedding)
                    .join(Citation)
                    .where(Citation.document_id.in_(document_ids))
                )
            else:
                # Refresh all embeddings
                embeddings_query = select(CitationEmbedding)
            
            result = await db_session.execute(embeddings_query)
            existing_embeddings = result.scalars().all()
            
            if not existing_embeddings:
                logger.info("No embeddings found to refresh")
                return RefreshResult(
                    citations_refreshed=0,
                    embeddings_deleted=0,
                    embeddings_created=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    processing_time_ms=0
                )
            
            # Get citation IDs and delete old embeddings
            citation_ids = [str(e.citation_id) for e in existing_embeddings]
            delete_count = len(existing_embeddings)
            
            delete_query = delete(CitationEmbedding).where(
                CitationEmbedding.citation_id.in_(citation_ids)
            )
            await db_session.execute(delete_query)
            
            # Get citations for re-embedding
            citations_query = select(Citation).where(Citation.id.in_(citation_ids))
            citations_result = await db_session.execute(citations_query)
            citations = citations_result.scalars().all()
            
            # Process refreshed embeddings in batches
            batch_result = await self.process_all_citations_batch(
                db_session=db_session,
                force_refresh=False,  # Already deleted old ones
                include_context=True,
                max_citations=None
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Model refresh completed: {old_model_version} -> {new_model_version}")
            logger.info(f"Refreshed {batch_result.processed_citations} citations")
            
            return RefreshResult(
                citations_refreshed=len(citations),
                embeddings_deleted=delete_count,
                embeddings_created=batch_result.processed_citations,
                total_tokens=batch_result.total_tokens,
                total_cost_usd=batch_result.total_cost_usd,
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            logger.error(f"Embedding refresh failed: {e}")
            raise
    
    async def cluster_citations_by_similarity(
        self,
        db_session: AsyncSession,
        document_ids: Optional[List[str]] = None,
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_threshold: float = 0.8
    ) -> ClusteringResult:
        """
        Cluster citations by semantic similarity for analysis and deduplication
        
        Args:
            db_session: Database session
            document_ids: Optional list of documents to cluster
            clustering_method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters for k-means (auto-detected if None)
            similarity_threshold: Similarity threshold for clustering
        """
        try:
            # Query citations with embeddings
            if document_ids:
                embeddings_query = (
                    select(CitationEmbedding, Citation)
                    .join(Citation)
                    .where(Citation.document_id.in_(document_ids))
                )
            else:
                embeddings_query = (
                    select(CitationEmbedding, Citation)
                    .join(Citation)
                )
            
            result = await db_session.execute(embeddings_query)
            embedding_pairs = result.all()
            
            if len(embedding_pairs) < 2:
                logger.warning("Not enough citations with embeddings for clustering")
                return ClusteringResult(
                    cluster_count=0,
                    citation_clusters={},
                    cluster_metadata={},
                    silhouette_score=None,
                    inertia=None
                )
            
            # Extract embeddings and citation info
            embeddings = []
            citations = []
            citation_ids = []
            
            for embedding_obj, citation_obj in embedding_pairs:
                if embedding_obj.embedding:
                    embeddings.append(np.array(embedding_obj.embedding))
                    citations.append(citation_obj)
                    citation_ids.append(str(citation_obj.id))
            
            if len(embeddings) < 2:
                logger.warning("Not enough valid embeddings for clustering")
                return ClusteringResult(
                    cluster_count=0,
                    citation_clusters={},
                    cluster_metadata={},
                    silhouette_score=None,
                    inertia=None
                )
            
            # Convert to numpy array
            embedding_matrix = np.vstack(embeddings)
            
            # Perform clustering
            if clustering_method == "kmeans":
                cluster_labels, cluster_metadata = await self._perform_kmeans_clustering(
                    embedding_matrix, n_clusters, citations
                )
            elif clustering_method == "dbscan":
                cluster_labels, cluster_metadata = await self._perform_dbscan_clustering(
                    embedding_matrix, similarity_threshold, citations
                )
            else:
                raise ValueError(f"Unsupported clustering method: {clustering_method}")
            
            # Organize results
            citation_clusters = defaultdict(list)
            for citation_id, cluster_label in zip(citation_ids, cluster_labels):
                if cluster_label != -1:  # -1 indicates noise in DBSCAN
                    citation_clusters[cluster_label].append(citation_id)
            
            # Calculate clustering quality metrics
            from sklearn.metrics import silhouette_score
            
            silhouette = None
            inertia = None
            
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                silhouette = silhouette_score(embedding_matrix, cluster_labels)
                
            if clustering_method == "kmeans" and hasattr(cluster_metadata, 'inertia_'):
                inertia = float(cluster_metadata.inertia_)
            
            logger.info(f"Clustering completed: {len(citation_clusters)} clusters, "
                       f"silhouette score: {silhouette:.3f}" if silhouette else "no score")
            
            return ClusteringResult(
                cluster_count=len(citation_clusters),
                citation_clusters=dict(citation_clusters),
                cluster_metadata=cluster_metadata if isinstance(cluster_metadata, dict) else {},
                silhouette_score=silhouette,
                inertia=inertia
            )
            
        except Exception as e:
            logger.error(f"Citation clustering failed: {e}")
            raise
    
    async def _perform_kmeans_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int],
        citations: List[Citation]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform K-means clustering on embeddings"""
        
        # Auto-detect optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(max(2, len(embeddings) // 10), 20)  # Heuristic
        
        n_clusters = min(n_clusters, len(embeddings))  # Can't have more clusters than points
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create cluster metadata
        cluster_metadata = {
            "algorithm": "kmeans",
            "n_clusters": n_clusters,
            "inertia": float(kmeans.inertia_),
            "cluster_centers": kmeans.cluster_centers_.tolist()
        }
        
        # Add cluster descriptions
        for i in range(n_clusters):
            cluster_citations = [citations[j] for j in range(len(citations)) if cluster_labels[j] == i]
            
            # Analyze cluster characteristics
            citation_types = [c.citation_type for c in cluster_citations]
            reporters = [c.reporter for c in cluster_citations if c.reporter]
            
            cluster_metadata[f"cluster_{i}"] = {
                "size": len(cluster_citations),
                "citation_types": list(set(citation_types)),
                "common_reporters": list(set(reporters))
            }
        
        return cluster_labels, cluster_metadata
    
    async def _perform_dbscan_clustering(
        self,
        embeddings: np.ndarray,
        similarity_threshold: float,
        citations: List[Citation]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform DBSCAN clustering on embeddings"""
        
        # Convert similarity threshold to distance
        # For cosine similarity, distance = 1 - similarity
        eps = 1.0 - similarity_threshold
        
        dbscan = DBSCAN(eps=eps, metric='cosine', min_samples=2)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        # Create cluster metadata
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        cluster_metadata = {
            "algorithm": "dbscan",
            "eps": eps,
            "similarity_threshold": similarity_threshold,
            "n_clusters": n_clusters,
            "n_noise": n_noise
        }
        
        # Add cluster descriptions
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise
                continue
                
            cluster_citations = [citations[j] for j in range(len(citations)) if cluster_labels[j] == cluster_id]
            
            citation_types = [c.citation_type for c in cluster_citations]
            reporters = [c.reporter for c in cluster_citations if c.reporter]
            
            cluster_metadata[f"cluster_{cluster_id}"] = {
                "size": len(cluster_citations),
                "citation_types": list(set(citation_types)),
                "common_reporters": list(set(reporters))
            }
        
        return cluster_labels, cluster_metadata
    
    async def get_embedding_statistics(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Get comprehensive statistics about citation embeddings"""
        try:
            # Total citations
            total_citations_query = select(func.count(Citation.id))
            total_citations_result = await db_session.execute(total_citations_query)
            total_citations = total_citations_result.scalar()
            
            # Citations with embeddings
            embedded_citations_query = select(func.count(CitationEmbedding.id))
            embedded_citations_result = await db_session.execute(embedded_citations_query)
            embedded_citations = embedded_citations_result.scalar()
            
            # Embedding coverage by citation type
            coverage_query = (
                select(Citation.citation_type, func.count(Citation.id).label('total'))
                .outerjoin(CitationEmbedding)
                .group_by(Citation.citation_type)
            )
            coverage_result = await db_session.execute(coverage_query)
            coverage_by_type = {row.citation_type: row.total for row in coverage_result}
            
            # Recent embedding creation
            recent_query = (
                select(func.count(CitationEmbedding.id))
                .where(CitationEmbedding.created_at >= datetime.utcnow() - timedelta(days=7))
            )
            recent_result = await db_session.execute(recent_query)
            recent_embeddings = recent_result.scalar()
            
            # Calculate statistics
            coverage_percentage = (embedded_citations / total_citations * 100) if total_citations > 0 else 0
            
            return {
                "total_citations": total_citations,
                "embedded_citations": embedded_citations,
                "coverage_percentage": round(coverage_percentage, 2),
                "embeddings_created_last_7_days": recent_embeddings,
                "coverage_by_citation_type": coverage_by_type,
                "current_model_version": self.embedding_model_version,
                "batch_size": self.batch_size,
                "context_window_chars": self.max_context_chars
            }
            
        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            raise
    
    async def find_similar_citations(
        self,
        db_session: AsyncSession,
        citation_id: str,
        threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find citations similar to a given citation using embeddings
        
        Args:
            db_session: Database session
            citation_id: Target citation ID
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
        """
        try:
            # Get target citation embedding
            target_query = (
                select(CitationEmbedding, Citation)
                .join(Citation)
                .where(CitationEmbedding.citation_id == citation_id)
            )
            target_result = await db_session.execute(target_query)
            target_pair = target_result.first()
            
            if not target_pair:
                logger.warning(f"No embedding found for citation {citation_id}")
                return []
            
            target_embedding, target_citation = target_pair
            target_vector = np.array(target_embedding.embedding).reshape(1, -1)
            
            # Get all other embeddings
            candidates_query = (
                select(CitationEmbedding, Citation)
                .join(Citation)
                .where(CitationEmbedding.citation_id != citation_id)
            )
            candidates_result = await db_session.execute(candidates_query)
            candidates = candidates_result.all()
            
            similarities = []
            
            for candidate_embedding, candidate_citation in candidates:
                candidate_vector = np.array(candidate_embedding.embedding).reshape(1, -1)
                similarity = cosine_similarity(target_vector, candidate_vector)[0, 0]
                
                if similarity >= threshold:
                    similarities.append({
                        "citation_id": str(candidate_citation.id),
                        "citation_text": candidate_citation.citation_text,
                        "citation_type": candidate_citation.citation_type,
                        "similarity_score": float(similarity),
                        "reporter": candidate_citation.reporter,
                        "volume": candidate_citation.volume,
                        "page": candidate_citation.page
                    })
            
            # Sort by similarity score (descending) and limit results
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to find similar citations: {e}")
            raise


# Global service instance
citation_embedding_service = CitationEmbeddingService() 