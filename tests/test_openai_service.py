"""
Tests for OpenAI integration service.
Tests embedding creation, rate limiting, caching, and error handling.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import json

from nlp.openai_service import OpenAIService, EmbeddingResult, EmbeddingBatch
from config.settings import settings


class TestOpenAIService:
    """Test suite for OpenAI embedding service"""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512)  # 1536 dimensions
        ]
        mock_response.usage.total_tokens = 10
        return mock_response
    
    @pytest.fixture
    def mock_batch_response(self):
        """Mock OpenAI API batch response"""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 512),
            MagicMock(embedding=[0.4, 0.5, 0.6] * 512)
        ]
        mock_response.usage.total_tokens = 20
        return mock_response
    
    @pytest.fixture
    def service_with_mocked_redis(self):
        """OpenAI service with mocked Redis"""
        with patch('nlp.openai_service.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            
            service = OpenAIService()
            service.redis_client = mock_redis_instance
            return service, mock_redis_instance
    
    @pytest.mark.asyncio
    async def test_create_single_embedding_success(self, mock_openai_response):
        """Test successful single embedding creation"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                # Setup mock
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.return_value = mock_openai_response
                
                service = OpenAIService()
                service.redis_client = None  # Disable cache for test
                
                # Test
                result = await service.create_embedding("test text")
                
                # Verify
                assert result.text == "test text"
                assert len(result.embedding) == 1536
                assert result.tokens_used == 10
                assert result.cached is False
                assert result.model == "text-embedding-3-small"
                
                # Verify API call
                mock_instance.embeddings.create.assert_called_once_with(
                    input=["test text"],
                    model="text-embedding-3-small"
                )
    
    @pytest.mark.asyncio
    async def test_create_embedding_with_cache_hit(self):
        """Test embedding creation with cache hit"""
        service, mock_redis = self.service_with_mocked_redis()
        
        # Setup cache hit
        cached_data = {
            "embedding": [0.7, 0.8, 0.9] * 512,
            "model": "text-embedding-3-small",
            "tokens_used": 5
        }
        mock_redis.get.return_value = json.dumps(cached_data)
        
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            result = await service.create_embedding("cached text")
        
        # Verify cache hit
        assert result.cached is True
        assert len(result.embedding) == 1536
        assert result.tokens_used == 5
        assert service.stats["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_create_embedding_with_cache_miss(self, mock_openai_response):
        """Test embedding creation with cache miss"""
        service, mock_redis = self.service_with_mocked_redis()
        
        # Setup cache miss
        mock_redis.get.return_value = None
        
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.return_value = mock_openai_response
                
                service.client = mock_instance
                
                result = await service.create_embedding("new text")
        
        # Verify cache miss and storage
        assert result.cached is False
        assert service.stats["cache_misses"] == 1
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_embedding_creation(self, mock_batch_response):
        """Test batch embedding creation"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.return_value = mock_batch_response
                
                service = OpenAIService()
                service.redis_client = None  # Disable cache
                
                texts = ["text one", "text two"]
                batch_result = await service.create_embeddings_batch(texts)
                
                # Verify batch result
                assert len(batch_result.results) == 2
                assert batch_result.total_tokens == 20
                assert batch_result.cached_count == 0
                
                # Verify individual results
                for i, result in enumerate(batch_result.results):
                    assert result.text == texts[i]
                    assert len(result.embedding) == 1536
                    assert result.cached is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self):
        """Test rate limiting functionality"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            service = OpenAIService()
            service.rate_limit_requests_per_minute = 2  # Low limit for testing
            service.redis_client = None
            
            # Simulate multiple requests
            current_time = 1000.0
            with patch('time.time', return_value=current_time):
                # First request should pass
                await service._enforce_rate_limits(1, 100)
                assert len(service.request_times) == 1
                
                # Second request should pass
                await service._enforce_rate_limits(1, 100)
                assert len(service.request_times) == 2
                
                # Third request should trigger rate limiting
                with patch('asyncio.sleep') as mock_sleep:
                    await service._enforce_rate_limits(1, 100)
                    mock_sleep.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_citation_embedding_creation(self, mock_batch_response):
        """Test citation-specific embedding creation"""
        # Create mock citations
        mock_citations = [
            MagicMock(
                citation_text="123 F.3d 456",
                doc_metadata={"court": "9th Cir.", "year": "2020"}
            ),
            MagicMock(
                citation_text="789 U.S. 101",
                doc_metadata={"court": "Supreme Court", "year": "2021"}
            )
        ]
        
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.return_value = mock_batch_response
                
                service = OpenAIService()
                service.redis_client = None
                
                results = await service.create_citation_embeddings(
                    citations=mock_citations,
                    include_context=True
                )
                
                # Verify results
                assert len(results) == 2
                
                # Verify API was called with enhanced text
                call_args = mock_instance.embeddings.create.call_args
                texts = call_args[1]['input']
                
                # Should include context
                assert "Court: 9th Cir." in texts[0]
                assert "Year: 2020" in texts[0]
                assert "Court: Supreme Court" in texts[1]
                assert "Year: 2021" in texts[1]
    
    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self):
        """Test error handling when API fails"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.side_effect = Exception("API Error")
                
                service = OpenAIService()
                service.redis_client = None
                
                # Should raise exception and update error stats
                with pytest.raises(Exception, match="API Error"):
                    await service.create_embedding("test text")
                
                assert service.stats["errors"] == 1
    
    @pytest.mark.asyncio
    async def test_statistics_calculation(self):
        """Test statistics calculation"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            service = OpenAIService()
            
            # Manually set some stats
            service.stats.update({
                "total_requests": 5,
                "total_tokens": 100,
                "total_cost_usd": 0.002,
                "cache_hits": 3,
                "cache_misses": 7
            })
            
            stats = service.get_statistics()
            
            # Verify calculated fields
            assert stats["avg_tokens_per_request"] == 20.0
            assert stats["avg_cost_per_request"] == 0.0004
            assert stats["cache_hit_rate"] == 0.3
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_openai_response):
        """Test successful health check"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.return_value = mock_openai_response
                
                service = OpenAIService()
                service.redis_client = None
                
                health = await service.health_check()
                
                assert health["api_healthy"] is True
                assert health["redis_healthy"] is False
                assert health["model"] == "text-embedding-3-small"
    
    @pytest.mark.asyncio
    async def test_health_check_api_failure(self):
        """Test health check with API failure"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            with patch('nlp.openai_service.AsyncOpenAI') as mock_client:
                mock_instance = AsyncMock()
                mock_client.return_value = mock_instance
                mock_instance.embeddings.create.side_effect = Exception("API Down")
                
                service = OpenAIService()
                service.redis_client = None
                
                health = await service.health_check()
                
                assert health["api_healthy"] is False
                assert "API Down" in health["api_message"]
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            service = OpenAIService()
            
            key1 = service._generate_cache_key("test text")
            key2 = service._generate_cache_key("test text")
            key3 = service._generate_cache_key("different text")
            
            # Same text should generate same key
            assert key1 == key2
            
            # Different text should generate different key
            assert key1 != key3
            
            # Should contain model and hash
            assert "text-embedding-3-small" in key1
            assert "openai_embedding:" in key1
    
    def test_service_initialization_without_api_key(self):
        """Test service initialization fails without API key"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', ''):
            with pytest.raises(ValueError, match="OPENAI_API_KEY not configured"):
                OpenAIService()
    
    @pytest.mark.asyncio
    async def test_empty_batch_handling(self):
        """Test handling of empty batch requests"""
        with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
            service = OpenAIService()
            
            result = await service.create_embeddings_batch([])
            
            assert len(result.results) == 0
            assert result.total_tokens == 0
            assert result.total_cost_usd == 0.0
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_graceful_degradation(self):
        """Test graceful degradation when Redis connection fails"""
        with patch('nlp.openai_service.Redis') as mock_redis:
            mock_redis.from_url.side_effect = Exception("Redis connection failed")
            
            with patch('nlp.openai_service.settings.OPENAI_API_KEY', 'test-key'):
                # Should not raise exception, just disable caching
                service = OpenAIService()
                assert service.redis_client is None 