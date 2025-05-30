"""
OpenAI API integration service for LeMCS.
Provides vector embeddings, rate limiting, error handling, and caching.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis
from redis.asyncio import Redis

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: List[float]
    model: str
    tokens_used: int
    cached: bool = False
    processing_time_ms: int = 0


@dataclass
class EmbeddingBatch:
    """Batch of embeddings with metadata"""
    results: List[EmbeddingResult]
    total_tokens: int
    total_cost_usd: float
    processing_time_ms: int
    cached_count: int


class OpenAIService:
    """Service for OpenAI API integration with rate limiting and caching"""
    
    def __init__(self):
        """Initialize the OpenAI service with configuration"""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-3-small"  # 1536 dimensions, cost-effective
        self.max_tokens_per_request = 8191  # Model limit
        self.max_batch_size = 2048  # Batch processing limit
        
        # Rate limiting (OpenAI allows 3000 RPM for embeddings)
        self.rate_limit_requests_per_minute = 2000  # Conservative limit
        self.rate_limit_tokens_per_minute = 1000000  # 1M tokens per minute
        
        # Request tracking for rate limiting
        self.request_times: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []  # (timestamp, tokens)
        
        # Redis for caching (optional)
        self.redis_client: Optional[Redis] = None
        self.cache_ttl = 86400 * 7  # Cache for 7 days
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_waits": 0,
            "errors": 0
        }
        
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = Redis.from_url(settings.REDIS_URL)
            logger.info("Redis cache initialized for OpenAI embeddings")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_client = None
    
    async def create_embedding(
        self, 
        text: str,
        use_cache: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Create a single embedding for the given text
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache for this request
            metadata: Optional metadata for logging
            
        Returns:
            EmbeddingResult with embedding vector and metadata
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache and self.redis_client:
            cached_result = await self._get_cached_embedding(text)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
            self.stats["cache_misses"] += 1
        
        # Rate limiting
        await self._enforce_rate_limits(1, len(text.split()))
        
        try:
            # Make API request
            response = await self._make_embedding_request([text])
            
            # Extract result
            embedding_data = response.data[0]
            embedding = embedding_data.embedding
            tokens_used = response.usage.total_tokens
            
            # Calculate cost (text-embedding-3-small: $0.00002 per 1K tokens)
            cost_usd = (tokens_used / 1000) * 0.00002
            
            # Create result
            result = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                tokens_used=tokens_used,
                cached=False,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            # Cache result
            if use_cache and self.redis_client:
                await self._cache_embedding(text, result)
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += tokens_used
            self.stats["total_cost_usd"] += cost_usd
            
            logger.debug(f"Created embedding for text ({tokens_used} tokens, ${cost_usd:.6f})")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    async def create_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: Optional[int] = None
    ) -> EmbeddingBatch:
        """
        Create embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            batch_size: Override default batch size
            
        Returns:
            EmbeddingBatch with all results and statistics
        """
        start_time = time.time()
        
        if not texts:
            return EmbeddingBatch(
                results=[],
                total_tokens=0,
                total_cost_usd=0.0,
                processing_time_ms=0,
                cached_count=0
            )
        
        batch_size = batch_size or self.max_batch_size
        all_results = []
        total_tokens = 0
        total_cost = 0.0
        cached_count = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for batch items
            batch_results = []
            uncached_texts = []
            uncached_indices = []
            
            if use_cache and self.redis_client:
                for idx, text in enumerate(batch_texts):
                    cached_result = await self._get_cached_embedding(text)
                    if cached_result:
                        batch_results.append((idx, cached_result))
                        cached_count += 1
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(idx)
            else:
                uncached_texts = batch_texts
                uncached_indices = list(range(len(batch_texts)))
            
            # Process uncached texts
            if uncached_texts:
                # Rate limiting for batch
                estimated_tokens = sum(len(text.split()) for text in uncached_texts)
                await self._enforce_rate_limits(1, estimated_tokens)
                
                try:
                    response = await self._make_embedding_request(uncached_texts)
                    
                    # Process results
                    for idx, (embedding_data, text) in enumerate(zip(response.data, uncached_texts)):
                        original_idx = uncached_indices[idx]
                        embedding = embedding_data.embedding
                        
                        result = EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            model=self.model,
                            tokens_used=0,  # Will be calculated from total
                            cached=False,
                            processing_time_ms=0
                        )
                        
                        batch_results.append((original_idx, result))
                        
                        # Cache result
                        if use_cache and self.redis_client:
                            await self._cache_embedding(text, result)
                    
                    # Update statistics
                    batch_tokens = response.usage.total_tokens
                    batch_cost = (batch_tokens / 1000) * 0.00002
                    
                    total_tokens += batch_tokens
                    total_cost += batch_cost
                    
                    self.stats["total_requests"] += 1
                    self.stats["total_tokens"] += batch_tokens
                    self.stats["total_cost_usd"] += batch_cost
                    
                except Exception as e:
                    self.stats["errors"] += 1
                    logger.error(f"Batch embedding failed: {e}")
                    raise
            
            # Sort results by original index and add to all_results
            batch_results.sort(key=lambda x: x[0])
            all_results.extend([result for _, result in batch_results])
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Created {len(all_results)} embeddings ({cached_count} cached, "
                   f"{total_tokens} tokens, ${total_cost:.6f})")
        
        return EmbeddingBatch(
            results=all_results,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            processing_time_ms=processing_time_ms,
            cached_count=cached_count
        )
    
    async def create_citation_embeddings(
        self,
        citations: List[Any],  # Citation objects
        include_context: bool = True
    ) -> List[EmbeddingResult]:
        """
        Create embeddings specifically for legal citations
        
        Args:
            citations: List of Citation objects
            include_context: Whether to include surrounding context
            
        Returns:
            List of EmbeddingResult objects
        """
        if not citations:
            return []
        
        # Prepare citation texts for embedding
        citation_texts = []
        for citation in citations:
            # Base citation text
            text = citation.citation_text
            
            # Add context if available and requested
            if include_context and hasattr(citation, 'doc_metadata'):
                metadata = citation.doc_metadata or {}
                
                # Add court information
                if metadata.get('court'):
                    text += f" Court: {metadata['court']}"
                
                # Add year
                if metadata.get('year'):
                    text += f" Year: {metadata['year']}"
                
                # Add corrected citation if different
                corrected = metadata.get('corrected_citation')
                if corrected and corrected != citation.citation_text:
                    text += f" Corrected: {corrected}"
            
            citation_texts.append(text)
        
        # Create embeddings
        batch_result = await self.create_embeddings_batch(
            citation_texts,
            use_cache=True
        )
        
        return batch_result.results
    
    async def _make_embedding_request(self, texts: List[str]):
        """Make the actual API request with retry logic"""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
        )
        async def _request():
            return await self.client.embeddings.create(
                input=texts,
                model=self.model
            )
        
        return await _request()
    
    async def _enforce_rate_limits(self, requests: int, estimated_tokens: int):
        """Enforce rate limiting before making requests"""
        current_time = time.time()
        
        # Clean old request times (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
        
        # Check request rate limit
        if len(self.request_times) + requests > self.rate_limit_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s for request limit")
                await asyncio.sleep(wait_time)
                self.stats["rate_limit_waits"] += 1
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.rate_limit_tokens_per_minute:
            # Find when we can proceed
            oldest_time = self.token_usage[0][0] if self.token_usage else current_time
            wait_time = 60 - (current_time - oldest_time)
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s for token limit")
                await asyncio.sleep(wait_time)
                self.stats["rate_limit_waits"] += 1
        
        # Record this request
        self.request_times.append(current_time)
        self.token_usage.append((current_time, estimated_tokens))
    
    async def _get_cached_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """Get embedding from cache if available"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(text)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return EmbeddingResult(
                    text=text,
                    embedding=data["embedding"],
                    model=data["model"],
                    tokens_used=data["tokens_used"],
                    cached=True,
                    processing_time_ms=0
                )
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_embedding(self, text: str, result: EmbeddingResult):
        """Cache embedding result"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._generate_cache_key(text)
            cache_data = {
                "embedding": result.embedding,
                "model": result.model,
                "tokens_used": result.tokens_used,
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"openai_embedding:{self.model}:{text_hash}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        stats = self.stats.copy()
        
        if stats["total_requests"] > 0:
            stats["avg_tokens_per_request"] = stats["total_tokens"] / stats["total_requests"]
            stats["avg_cost_per_request"] = stats["total_cost_usd"] / stats["total_requests"]
        
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health and connectivity"""
        try:
            # Test API connectivity with a small request
            test_response = await self.client.embeddings.create(
                input=["test"],
                model=self.model
            )
            
            api_healthy = True
            api_message = "API connectivity OK"
        except Exception as e:
            api_healthy = False
            api_message = f"API error: {e}"
        
        # Check Redis connectivity
        redis_healthy = False
        redis_message = "Redis not configured"
        
        if self.redis_client:
            try:
                await self.redis_client.ping()
                redis_healthy = True
                redis_message = "Redis connectivity OK"
            except Exception as e:
                redis_message = f"Redis error: {e}"
        
        return {
            "service": "OpenAI Embedding Service",
            "api_healthy": api_healthy,
            "api_message": api_message,
            "redis_healthy": redis_healthy,
            "redis_message": redis_message,
            "model": self.model,
            "statistics": self.get_statistics(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global service instance
openai_service = OpenAIService() 