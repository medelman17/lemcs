"""
Tests for Citation Embedding Service.
Tests batch processing, model refresh, clustering, and similarity search functionality.
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from sqlalchemy import select
from uuid import uuid4

from nlp.citation_embedding_service import (
    CitationEmbeddingService, 
    BatchProcessingResult,
    ClusteringResult,
    RefreshResult,
    citation_embedding_service
)
from db.models import Citation, CitationEmbedding, Document


class TestCitationEmbeddingService:
    """Test suite for Citation Embedding Service"""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for testing"""
        return CitationEmbeddingService()
    
    @pytest.fixture
    def mock_citations(self):
        """Create mock citation objects"""
        citations = []
        for i in range(5):
            citation = MagicMock(spec=Citation)
            citation.id = uuid4()
            citation.citation_text = f"123 F.3d {456 + i}"
            citation.citation_type = "FullCaseCitation"
            citation.reporter = "F.3d"
            citation.volume = "123"
            citation.page = str(456 + i)
            citation.document_id = uuid4()
            citations.append(citation)
        return citations
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embedding vectors"""
        return [np.random.rand(1536).tolist() for _ in range(5)]
    
    @pytest.fixture
    def mock_embedding_results(self, mock_embeddings):
        """Create mock embedding result objects"""
        results = []
        for embedding in mock_embeddings:
            result = MagicMock()
            result.embedding = embedding
            result.tokens_used = 50
            result.cached = False
            results.append(result)
        return results
    
    @pytest.mark.asyncio
    async def test_batch_processing_success(
        self, 
        service, 
        mock_citations, 
        mock_embedding_results
    ):
        """Test successful batch processing of citations"""
        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_citations
        mock_session.commit = AsyncMock()
        
        # Mock OpenAI service
        with patch('nlp.citation_embedding_service.openai_service') as mock_openai:
            mock_openai.create_citation_embeddings.return_value = mock_embedding_results
            
            result = await service.process_all_citations_batch(
                db_session=mock_session,
                force_refresh=False,
                include_context=True,
                max_citations=None
            )
            
            # Verify result
            assert isinstance(result, BatchProcessingResult)
            assert result.total_citations == 5
            assert result.processed_citations == 5
            assert result.failed_citations == 0
            assert result.total_tokens == 250  # 5 * 50 tokens
            assert result.total_cost_usd > 0
            assert len(result.errors) == 0
            
            # Verify OpenAI service was called
            mock_openai.create_citation_embeddings.assert_called_once_with(
                citations=mock_citations,
                include_context=True
            )
            
            # Verify database operations
            assert mock_session.add.call_count == 5  # One add per embedding
            mock_session.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_force_refresh(
        self, 
        service, 
        mock_citations, 
        mock_embedding_results
    ):
        """Test batch processing with force refresh"""
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_citations
        
        with patch('nlp.citation_embedding_service.openai_service') as mock_openai:
            mock_openai.create_citation_embeddings.return_value = mock_embedding_results
            
            result = await service.process_all_citations_batch(
                db_session=mock_session,
                force_refresh=True,
                include_context=True,
                max_citations=3
            )
            
            # Verify delete query was executed for force refresh
            assert mock_session.execute.call_count >= 2  # Query + Delete operations
            assert result.processed_citations == 5
    
    @pytest.mark.asyncio
    async def test_batch_processing_empty_citations(self, service):
        """Test batch processing with no citations"""
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        
        result = await service.process_all_citations_batch(
            db_session=mock_session,
            force_refresh=False,
            include_context=True
        )
        
        assert result.total_citations == 0
        assert result.processed_citations == 0
        assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_failures(
        self, 
        service, 
        mock_citations
    ):
        """Test batch processing with some embedding failures"""
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = mock_citations
        
        # Create mock results with some failures
        mock_results = []
        for i in range(5):
            result = MagicMock()
            result.embedding = np.random.rand(1536).tolist() if i < 3 else None  # 3 success, 2 failures
            result.tokens_used = 50
            mock_results.append(result)
        
        with patch('nlp.citation_embedding_service.openai_service') as mock_openai:
            mock_openai.create_citation_embeddings.return_value = mock_results
            
            result = await service.process_all_citations_batch(
                db_session=mock_session,
                force_refresh=False,
                include_context=True
            )
            
            assert result.processed_citations == 3
            assert result.failed_citations == 2
            assert len(result.errors) == 2
    
    @pytest.mark.asyncio
    async def test_model_refresh_success(
        self, 
        service, 
        mock_citations, 
        mock_embedding_results
    ):
        """Test successful model refresh"""
        mock_session = AsyncMock()
        
        # Mock existing embeddings
        mock_existing_embeddings = []
        for i in range(3):
            embedding = MagicMock()
            embedding.citation_id = mock_citations[i].id
            mock_existing_embeddings.append(embedding)
        
        mock_session.execute.return_value.scalars.return_value.all.side_effect = [
            mock_existing_embeddings,  # First query for existing embeddings
            mock_citations[:3]         # Second query for citations to re-embed
        ]
        
        with patch('nlp.citation_embedding_service.openai_service') as mock_openai:
            mock_openai.create_citation_embeddings.return_value = mock_embedding_results[:3]
            
            # Mock the batch processing method
            with patch.object(service, 'process_all_citations_batch') as mock_batch:
                mock_batch_result = BatchProcessingResult(
                    total_citations=3,
                    processed_citations=3,
                    skipped_citations=0,
                    failed_citations=0,
                    total_tokens=150,
                    total_cost_usd=0.003,
                    processing_time_ms=1000,
                    errors=[]
                )
                mock_batch.return_value = mock_batch_result
                
                result = await service.refresh_embeddings_for_model_update(
                    db_session=mock_session,
                    new_model_version="text-embedding-3-large",
                    document_ids=None
                )
                
                assert isinstance(result, RefreshResult)
                assert result.citations_refreshed == 3
                assert result.embeddings_deleted == 3
                assert result.embeddings_created == 3
                assert service.embedding_model_version == "text-embedding-3-large"
    
    @pytest.mark.asyncio
    async def test_clustering_kmeans(self, service):
        """Test k-means clustering functionality"""
        mock_session = AsyncMock()
        
        # Create mock embedding pairs
        mock_pairs = []
        for i in range(5):
            embedding_obj = MagicMock()
            embedding_obj.embedding = np.random.rand(1536).tolist()
            
            citation_obj = MagicMock()
            citation_obj.id = uuid4()
            citation_obj.citation_type = "FullCaseCitation"
            citation_obj.reporter = "F.3d"
            
            mock_pairs.append((embedding_obj, citation_obj))
        
        mock_session.execute.return_value.all.return_value = mock_pairs
        
        with patch('sklearn.cluster.KMeans') as mock_kmeans:
            # Mock KMeans results
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 2])
            mock_kmeans_instance.inertia_ = 10.5
            mock_kmeans_instance.cluster_centers_ = np.random.rand(3, 1536)
            mock_kmeans.return_value = mock_kmeans_instance
            
            result = await service.cluster_citations_by_similarity(
                db_session=mock_session,
                document_ids=None,
                clustering_method="kmeans",
                n_clusters=3,
                similarity_threshold=0.8
            )
            
            assert isinstance(result, ClusteringResult)
            assert result.cluster_count == 3
            assert len(result.citation_clusters) == 3
            assert result.inertia == 10.5
    
    @pytest.mark.asyncio
    async def test_clustering_dbscan(self, service):
        """Test DBSCAN clustering functionality"""
        mock_session = AsyncMock()
        
        # Create mock embedding pairs
        mock_pairs = []
        for i in range(5):
            embedding_obj = MagicMock()
            embedding_obj.embedding = np.random.rand(1536).tolist()
            
            citation_obj = MagicMock()
            citation_obj.id = uuid4()
            citation_obj.citation_type = "FullCaseCitation"
            citation_obj.reporter = "F.3d"
            
            mock_pairs.append((embedding_obj, citation_obj))
        
        mock_session.execute.return_value.all.return_value = mock_pairs
        
        with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
            # Mock DBSCAN results (clusters 0, 0, 1, 1, -1 where -1 is noise)
            mock_dbscan_instance = MagicMock()
            mock_dbscan_instance.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
            mock_dbscan.return_value = mock_dbscan_instance
            
            result = await service.cluster_citations_by_similarity(
                db_session=mock_session,
                document_ids=None,
                clustering_method="dbscan",
                similarity_threshold=0.8
            )
            
            assert isinstance(result, ClusteringResult)
            assert result.cluster_count == 2  # Clusters 0 and 1, excluding noise (-1)
            assert len(result.citation_clusters) == 2
    
    @pytest.mark.asyncio
    async def test_clustering_insufficient_data(self, service):
        """Test clustering with insufficient data"""
        mock_session = AsyncMock()
        mock_session.execute.return_value.all.return_value = []  # No embeddings
        
        result = await service.cluster_citations_by_similarity(
            db_session=mock_session,
            document_ids=None,
            clustering_method="kmeans"
        )
        
        assert result.cluster_count == 0
        assert len(result.citation_clusters) == 0
        assert result.silhouette_score is None
    
    @pytest.mark.asyncio
    async def test_find_similar_citations(self, service):
        """Test finding similar citations"""
        mock_session = AsyncMock()
        
        # Mock target citation
        target_embedding = MagicMock()
        target_embedding.embedding = np.random.rand(1536).tolist()
        target_citation = MagicMock()
        target_citation.id = uuid4()
        
        # Mock candidate citations
        candidate_pairs = []
        for i in range(3):
            embedding_obj = MagicMock()
            embedding_obj.embedding = np.random.rand(1536).tolist()
            
            citation_obj = MagicMock()
            citation_obj.id = uuid4()
            citation_obj.citation_text = f"Case {i}"
            citation_obj.citation_type = "FullCaseCitation"
            citation_obj.reporter = "F.3d"
            citation_obj.volume = "123"
            citation_obj.page = str(456 + i)
            
            candidate_pairs.append((embedding_obj, citation_obj))
        
        # Setup mock session returns
        mock_session.execute.return_value.first.return_value = (target_embedding, target_citation)
        mock_session.execute.return_value.all.return_value = candidate_pairs
        
        with patch('sklearn.metrics.pairwise.cosine_similarity') as mock_cosine:
            # Mock high similarity scores
            mock_cosine.return_value = np.array([[0.9], [0.85], [0.75]])
            
            result = await service.find_similar_citations(
                db_session=mock_session,
                citation_id=str(target_citation.id),
                threshold=0.8,
                max_results=10
            )
            
            # Should return 2 results (0.9 and 0.85 are above 0.8 threshold)
            assert len(result) == 2
            assert result[0]["similarity_score"] == 0.9  # Highest first
            assert result[1]["similarity_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_get_embedding_statistics(self, service):
        """Test getting embedding statistics"""
        mock_session = AsyncMock()
        
        # Mock various database queries
        mock_session.execute.return_value.scalar.side_effect = [
            100,  # total citations
            75,   # embedded citations
            10    # recent embeddings
        ]
        
        # Mock coverage by type query
        mock_coverage_row = MagicMock()
        mock_coverage_row.citation_type = "FullCaseCitation"
        mock_coverage_row.total = 60
        mock_session.execute.return_value.__iter__ = lambda x: iter([mock_coverage_row])
        
        stats = await service.get_embedding_statistics(mock_session)
        
        assert stats["total_citations"] == 100
        assert stats["embedded_citations"] == 75
        assert stats["coverage_percentage"] == 75.0
        assert stats["embeddings_created_last_7_days"] == 10
        assert "text-embedding-3-small" in stats["current_model_version"]
        assert stats["batch_size"] == 100
    
    @pytest.mark.asyncio
    async def test_find_similar_citations_no_embedding(self, service):
        """Test finding similar citations when target has no embedding"""
        mock_session = AsyncMock()
        mock_session.execute.return_value.first.return_value = None
        
        result = await service.find_similar_citations(
            db_session=mock_session,
            citation_id="nonexistent-id",
            threshold=0.8,
            max_results=10
        )
        
        assert result == []
    
    def test_service_initialization(self, service):
        """Test service initialization with default values"""
        assert service.batch_size == 100
        assert service.max_context_chars == 500
        assert service.embedding_model_version == "text-embedding-3-small"
        assert isinstance(service.clustering_cache, dict)
    
    @pytest.mark.asyncio
    async def test_kmeans_auto_cluster_detection(self, service):
        """Test automatic cluster count detection in k-means"""
        # Test with different embedding counts
        embeddings_50 = np.random.rand(50, 1536)
        citations_50 = [MagicMock() for _ in range(50)]
        
        with patch('sklearn.cluster.KMeans') as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.zeros(50)
            mock_kmeans_instance.inertia_ = 10.0
            mock_kmeans_instance.cluster_centers_ = np.random.rand(5, 1536)
            mock_kmeans.return_value = mock_kmeans_instance
            
            labels, metadata = await service._perform_kmeans_clustering(
                embeddings_50, None, citations_50
            )
            
            # Should auto-detect 5 clusters for 50 embeddings (50 // 10)
            assert mock_kmeans.call_args[1]['n_clusters'] == 5
    
    @pytest.mark.asyncio
    async def test_dbscan_parameter_conversion(self, service):
        """Test DBSCAN parameter conversion from similarity to distance"""
        embeddings = np.random.rand(10, 1536)
        citations = [MagicMock() for _ in range(10)]
        
        with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
            mock_dbscan_instance = MagicMock()
            mock_dbscan_instance.fit_predict.return_value = np.zeros(10)
            mock_dbscan.return_value = mock_dbscan_instance
            
            await service._perform_dbscan_clustering(
                embeddings, 0.8, citations  # 0.8 similarity
            )
            
            # Should convert to distance: 1.0 - 0.8 = 0.2
            assert mock_dbscan.call_args[1]['eps'] == 0.2
            assert mock_dbscan.call_args[1]['metric'] == 'cosine'


@pytest.mark.asyncio
async def test_global_service_instance():
    """Test that the global service instance is properly initialized"""
    assert citation_embedding_service is not None
    assert isinstance(citation_embedding_service, CitationEmbeddingService)
    assert citation_embedding_service.batch_size == 100 